#!/usr/bin/env python
"""Raw Telegram history (shootingbets_raw.json) -> alerts.csv (one row per parsed alert) + messages.csv (every message).

    python scripts/build_alerts.py [--in data/telegram/shootingbets_raw.json] [--out data/alerts/alerts.csv]
                                   [--messages-out data/alerts/messages.csv]
Prints coverage and counts (ASCII only, never message texts). Exit 0 when every alert-kind message parsed, 3 when some
did not (both files are still written), 2 on bad input.
When the vendor results file exists (--vendor) its columns (sbroi/vendor.py) are added to alerts.csv and summarised.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sbroi.alerts import build_alerts, coverage, load_messages  # noqa: E402
from sbroi.vendor import join_vendor, load_vendor  # noqa: E402


def say(text="") -> None:
    print(str(text).encode("ascii", "backslashreplace").decode("ascii"))


def _iso(s: pd.Series) -> pd.Series:
    return s.map(lambda t: None if pd.isna(t) else t.isoformat())


def _write(df: pd.DataFrame, path, cols=("ts", "kickoff")) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in cols:
        if c in out:
            out[c] = _iso(out[c])
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _counts(title: str, s: pd.Series) -> None:
    say(title)
    for k, n in s.value_counts().sort_index().items():
        say(f"  {k}: {n}")


def _vendor_summary(al: pd.DataFrame, vendor: pd.DataFrame) -> None:
    say(f"vendor file: {len(vendor)} rows, sum pl_50 / (50 * rows) = {vendor['pl_50'].sum() / (50 * max(len(vendor), 1)):+.4f}")
    j = al[al["v_found"]]
    say("joined to the vendor file, by check_status (joined of all):")
    for k, n in al["check_status"].value_counts().sort_index().items():
        say(f"  {k}: {int((j['check_status'] == k).sum())} of {n}")
    s = j[j["v_same_line"]]
    c = s[s["own_won"].notna()]
    agree = c["own_won"].astype(bool) == c["v_result"].eq("W")
    say(f"own_won vs vendor result, rows on the same line: {int(agree.sum())} agree of {len(c)} compared "
        f"({len(s) - len(c)} not settled by the score, {len(j) - len(s)} on another line)")
    say("  disagree (alert ids): " + (", ".join(str(i) for i in c.loc[~agree, "alert_id"]) or "none"))
    for col in ("own_pl_min", "v_pl_b365"):
        x = al[col].dropna()
        say(f"{col}: n {len(x)}" + (f"  strike {(x > 0).mean():.1%}  ROI {x.mean():+.2%}" if len(x) else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "telegram" / "shootingbets_raw.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "alerts" / "alerts.csv"))
    ap.add_argument("--messages-out", default=str(ROOT / "data" / "alerts" / "messages.csv"))
    ap.add_argument("--vendor", default=str(ROOT / "data" / "vendor_results" / "sb_vendor_results.json"),
                    help="vendor results JSON (skipped silently when the file is absent)")
    a = ap.parse_args(argv)
    try:
        alerts, messages = build_alerts(load_messages(a.inp))
        vendor = load_vendor(a.vendor) if Path(a.vendor).is_file() else None
        if vendor is not None:
            alerts = join_vendor(alerts, vendor)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        say(f"bad input: {type(exc).__name__}: {exc}")
        return 2
    _write(alerts, a.out)
    _write(messages, a.messages_out, cols=("ts",))
    cov = coverage(messages)
    say(f"written: {a.out}\nwritten: {a.messages_out}")
    say("coverage: " + ", ".join(f"{k} {n}" for k, n in cov["kinds"].items()) + f" (messages {cov['n_messages']})")
    say(f"alerts: {cov['n_alert']}  parsed: {cov['n_parsed']}  unparsed: {cov['n_unparsed']}  "
        f"parse_rate: {cov['parse_rate']:.4f}")
    if len(alerts):
        t0, t1 = alerts["ts"].min(), alerts["ts"].max()
        for lbl, t in (("first", t0), ("last", t1)):
            say(f"{lbl} alert: {t:%Y-%m-%d %H:%M:%S} UTC / {t.tz_convert('Australia/Melbourne'):%Y-%m-%d %H:%M:%S} Melbourne")
        al = alerts.assign(scope=alerts["scope"].fillna("-"), bf_market=alerts["bf_market"].fillna("(unmapped)"))
        _counts("by sport:", al["sport_raw"])
        _counts("by mkt_kind:", al["mkt_kind"])
        _counts("by scope:", al["scope"])
        _counts("by sport x mkt_kind x scope:", al["sport_raw"] + " | " + al["mkt_kind"] + " | " + al["scope"])
        _counts("by bf_market:", al["bf_market"])
        _counts("unmapped by map_note:", alerts.loc[alerts["bf_market"].isna(), "map_note"])
        _counts("check_status:", al["check_status"])
        say(f"in_free_window: {int(al['in_free_window'].sum())}  repeats: {int(al['is_repeat'].sum())}  "
            f"edited: {int(al['edited'].sum())}")
        if vendor is not None:
            _vendor_summary(alerts, vendor)
    bad = messages[(messages["kind"] == "alert") & (messages["parse_status"] != "ok")]
    if len(bad):
        say("unparsed alerts (id: reason):")
        for r in bad.itertuples():
            say(f"  {r.id}: {r.parse_status}")
    return 0 if bad.empty else 3


if __name__ == "__main__":
    sys.exit(main())
