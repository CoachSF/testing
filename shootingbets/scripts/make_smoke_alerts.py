#!/usr/bin/env python
"""Pseudo-alerts for PLUMBING tests ONLY (never real results): random soccer events from the PRO manifest.

    python scripts/make_smoke_alerts.py [--n 40] [--seed 7] -> data/smoke/alerts_smoke.csv
One alert per event: MATCH_ODDS (home/away/draw in turn) and OVER_UNDER_25 (over/under in turn), alternately, at a random
20-600 minutes before the off. Only events whose MATCH_ODDS and OVER_UNDER_25 files both exist and are 30-400 kB.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sbroi.pro_index import load_manifest  # noqa: E402
from sbroi.textnorm import split_event_name  # noqa: E402

COLUMNS = ["alert_id", "ts", "sport", "home", "away", "kickoff", "bf_market", "sel_kind", "sel_team", "sel_hc", "ref_odds", "text"]


def make(pro_dir, n=40, seed=7) -> pd.DataFrame:
    pro_dir = Path(pro_dir)
    m = load_manifest(pro_dir)
    m = m[m["bytes"].between(30_000, 400_000)]
    mo, ou = (m[m["market_type"] == t].drop_duplicates("event_id").set_index("event_id") for t in ("MATCH_ODDS", "OVER_UNDER_25"))
    events = sorted(set(mo.index) & set(ou.index))
    rng = random.Random(seed)
    rng.shuffle(events)
    rows = []
    for ev in events:
        a, b = mo.loc[ev], ou.loc[ev]
        sp = split_event_name(a["event_name"]) if isinstance(a["event_name"], str) else None
        if not sp or pd.isna(a["market_time_ms"]) or not all((pro_dir / x["saved"]).is_file() for x in (a, b)):
            continue
        i = len(rows)
        ts = pd.to_datetime(int(a["market_time_ms"]) - rng.uniform(20, 600) * 60_000, unit="ms", utc=True).round("s")
        mkt, kind = (("MATCH_ODDS", ("home", "away", "draw")[(i // 2) % 3]) if i % 2 == 0 else
                     ("OVER_UNDER_25", ("over", "under")[(i // 2) % 2]))
        rows.append({"alert_id": i + 1, "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "sport": "Soccer", "home": sp[0], "away": sp[1],
                     "kickoff": a["market_time"], "bf_market": mkt, "sel_kind": kind, "sel_team": "", "sel_hc": "",
                     "ref_odds": "", "text": "SMOKE"})
        if len(rows) >= n:
            break
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pro-dir", default=str(ROOT / "data" / "betfair_pro"))
    ap.add_argument("--out", default=str(ROOT / "data" / "smoke" / "alerts_smoke.csv"))
    a = ap.parse_args(argv)
    df = make(a.pro_dir, a.n, a.seed)
    if df.empty:
        print("no suitable events in the manifest")
        return 2
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, encoding="utf-8")
    print(f"wrote {len(df)} smoke alerts to {a.out}" + ("" if len(df) == a.n else f" (asked for {a.n})"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
