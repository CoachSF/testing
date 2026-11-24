#!/usr/bin/env python
"""Alerts CSV -> enrich against the Betfair PRO store (one alert at a time) -> CSV tables + report.html.

    python scripts/run_analysis.py --alerts alerts.csv [--pro-dir DIR[=Sport] ...] [--coverage-end ISO] [--out output]
--pro-dir takes one or more PRO folders; ``DIR=Sport`` sets the sport of the rows whose manifest has no event_type_id
(default Soccer). --coverage-end (e.g. 2026-11-01T00:00:00Z): alerts kicking off at/after it are "pro_not_published".
Exit code 0 when the report was written, 2 on bad input. An exception while enriching ONE alert becomes a row
with status "error:<ExceptionName>" (message in the ``error`` column); it never kills the run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbroi.pipeline import Config, enrich  # noqa: E402
from sbroi.pro_index import MarketStore, build_event_index_from_manifest, load_manifest  # noqa: E402
from sbroi.report import build_report  # noqa: E402

REQUIRED = ["alert_id", "ts", "sport", "home", "away", "bf_market", "sel_kind"]
COMMISSIONS = (0.0, 0.02, 0.048, 0.05)


def load_alerts(path, limit=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"alerts CSV lacks columns {missing}")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    if "kickoff" in df:
        df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, errors="coerce")
    for c in df.columns:  # blank text cells must be None, not NaN (NaN is truthy inside the matcher)
        if not pd.api.types.is_datetime64_any_dtype(df[c]) and (
                pd.api.types.is_string_dtype(df[c]) or df[c].dtype == object or df[c].isna().all()):
            df[c] = pd.Series([None if pd.isna(v) else v for v in df[c]], index=df.index, dtype=object)
    if limit:
        df = df.head(limit)
    if df.empty:
        raise ValueError("no alerts to analyse")
    return df.reset_index(drop=True)


def parse_pro_dirs(values) -> tuple[list[str], list[str]]:
    """["DIR", "DIR2=Tennis"] -> (["DIR", "DIR2"], ["Soccer", "Tennis"])."""
    pairs = [(v.rpartition("=")[0], v.rpartition("=")[2]) if "=" in v else (v, "") for v in values]
    return [d for d, _ in pairs], [s or "Soccer" for _, s in pairs]


def _peak_mb():
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        k, p = ctypes.windll.kernel32, ctypes.windll.psapi
        k.GetCurrentProcess.restype = wintypes.HANDLE
        p.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(pmc)
        return pmc.PeakWorkingSetSize / 1e6 if p.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb) else None
    except Exception:  # not Windows / no psapi
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--alerts", required=True, help="alerts CSV (columns per sbroi/pipeline.py)")
    ap.add_argument("--pro-dir", nargs="+", default=["data/betfair_pro"], metavar="DIR[=Sport]",
                    help="one or more PRO folders; DIR=Sport = default sport of rows without event_type_id (default Soccer)")
    ap.add_argument("--out", default="output")
    ap.add_argument("--coverage-end", default=None, metavar="ISO",
                    help="end of the published PRO data, e.g. 2026-11-01T00:00:00Z: alerts kicking off at/after it get "
                         "status pro_not_published and are not matched (default: none)")
    ap.add_argument("--commission", type=float, default=0.048)
    ap.add_argument("--label", default=None, help="banner text, e.g. to mark smoke-test output as not real")
    ap.add_argument("--max-cached", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="only the first N alerts (testing)")
    a = ap.parse_args(argv)
    try:
        alerts = load_alerts(a.alerts, a.limit)
        dirs, sports = parse_pro_dirs(a.pro_dir)
        manifest = load_manifest(dirs, sports)
        if manifest.empty:
            raise ValueError(f"no manifest_*.jsonl rows in {', '.join(dirs)}")
        end = None if a.coverage_end is None else pd.Timestamp(a.coverage_end)
        end_ms = None if end is None else int((end.tz_localize("UTC") if end.tzinfo is None else end).timestamp() * 1000)
    except (OSError, ValueError, KeyError) as e:
        print(f"bad input: {type(e).__name__}: {e}")
        return 2
    index = build_event_index_from_manifest(manifest)
    store = MarketStore(dirs[0], manifest, max_cached=a.max_cached)
    cfg = Config(commissions=tuple(sorted({*COMMISSIONS, a.commission})), scalp_commission=a.commission, coverage_end_ms=end_ms)
    parts, t0, n = [], time.time(), len(alerts)
    for i in range(n):
        one = alerts.iloc[[i]]
        try:
            parts.append(enrich(one, store, index, cfg))
        except Exception as e:  # one bad alert must not kill the run
            r = one.iloc[0].to_dict()
            r.update(status=f"error:{type(e).__name__}", error=str(e)[:300])
            parts.append(pd.DataFrame([r]))
        if (i + 1) % 10 == 0:
            print(f"progress: {i + 1}/{n} alerts, {time.time() - t0:.1f}s elapsed, {store.loads} market parses", flush=True)
    enrich_s = time.time() - t0
    enriched = pd.concat(parts, ignore_index=True)
    key = build_report(enriched, a.out, commission=a.commission, label=a.label, n_alerts_input=n)
    peak = _peak_mb()
    print(f"output folder: {Path(a.out).resolve()}")
    print("status counts:", json.dumps(key["status_counts"]))
    print(f"enrich: {enrich_s:.1f}s for {n} alerts ({enrich_s / n:.2f}s/alert), {store.loads} market parses; total {time.time() - t0:.1f}s"
          + (f"; peak working set {peak:.0f} MB" if peak else ""))
    print("key numbers:")
    print(json.dumps(key, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
