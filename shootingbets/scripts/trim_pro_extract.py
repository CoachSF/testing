#!/usr/bin/env python
"""Delete every extracted PRO event folder that no alert matched (keeps the manifests).

    python scripts/trim_pro_extract.py --enriched output/enriched.csv --pro-dir DIR[=Sport] [DIR[=Sport] ...] [--commit]

Event folder = <pro dir>/<YYYY-MM-DD>/<eventId>/. Keep set = the non-empty ``match_event_id`` values of the enriched CSV.
Nothing else is touched (manifests, _done_*.json, __pycache__, other folders, plain files in a date folder).
Default is a DRY RUN. Exit 0 ok, 2 bad input (nothing deleted), 3 a kept event is not on disk (nothing deleted),
4 --commit left an event folder that is not in the keep set. After a trim the manifests still list every market, but
only the kept events have files: enrich() then raises MarketFileMissing for an event that was trimmed away.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")
NOTE = "manifests still list every market, but only the kept events have files"


def event_dirs(pro: Path) -> list[Path]:
    """<pro>/<YYYY-MM-DD>/<eventId>/ - a directory that is a child of a date-named child directory of ``pro``."""
    return [e for d in sorted(pro.iterdir()) if d.is_dir() and DATE_DIR.match(d.name)
            for e in sorted(d.iterdir()) if e.is_dir()]


def dir_bytes(path: Path) -> int:
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files)


def keep_ids(csv_path: str) -> set[str]:
    df = pd.read_csv(csv_path, dtype=str)
    if "match_event_id" not in df:
        raise ValueError("no match_event_id column")
    ids = (s.strip() for s in df["match_event_id"].dropna())
    return {s[:-2] if s.endswith(".0") else s for s in ids if s}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enriched", required=True, help="enriched.csv written by run_analysis.py")
    ap.add_argument("--pro-dir", nargs="+", required=True, metavar="DIR[=Sport]", help="PRO folders (=Sport is ignored)")
    ap.add_argument("--commit", action="store_true", help="really delete (default: dry run)")
    a = ap.parse_args(argv)
    try:
        keep = keep_ids(a.enriched)
    except (OSError, ValueError) as e:
        print(f"bad input: {type(e).__name__}: {e}")
        return 2
    if not keep:
        print("bad input: no match_event_id values in the CSV - nothing would be kept")
        return 2
    pros = [Path(v.rpartition("=")[0] if "=" in v else v) for v in a.pro_dir]
    for p in pros:
        if not p.is_dir():
            print(f"bad input: pro dir does not exist: {p}")
            return 2
    events = {p: event_dirs(p) for p in pros}
    missing = sorted(keep - {e.name for evs in events.values() for e in evs})
    if missing:
        print(f"ABORT: {len(missing)} kept event id(s) not found as an event folder in any pro dir "
              f"(wrong CSV for these folders?): {', '.join(missing)}")
        return 3
    tot_k = tot_d = 0
    plan = {}
    for p, evs in events.items():
        kept, gone = [e for e in evs if e.name in keep], [e for e in evs if e.name not in keep]
        kb, db = sum(map(dir_bytes, kept)), sum(map(dir_bytes, gone))
        plan[p] = (kept, gone, db)
        tot_k, tot_d = tot_k + len(kept), tot_d + len(gone)
        print(f"{p}: keep {len(kept)} events {kb / 1e6:.1f} MB | delete {len(gone)} events {db / 1e6:.1f} MB")
    print(f"TOTAL: keep {tot_k} events | delete {tot_d} events {sum(v[2] for v in plan.values()) / 1e6:.1f} MB")
    if not a.commit:
        print("DRY RUN - nothing deleted (add --commit)")
        return 0
    for p, (kept, gone, db) in plan.items():
        for e in gone:
            shutil.rmtree(e)
        for d in {e.parent for e in gone}:
            if not any(d.iterdir()):
                os.rmdir(d)
        (p / "_trimmed.json").write_text(json.dumps({
            "trimmed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "kept_event_ids": sorted({e.name for e in kept}), "deleted_events": len(gone), "deleted_bytes": db,
            "note": NOTE}, indent=1), encoding="utf-8")
    left = [e for p in pros for e in event_dirs(p)]
    bad = sorted({e.name for e in left if e.name not in keep})
    if bad:
        print(f"VERIFY FAILED: {len(bad)} remaining event(s) not in the keep set: {', '.join(bad)}")
        return 4
    print(f"VERIFY remaining events: {len(left)} (all kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
