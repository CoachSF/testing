"""Build the minimal Betfair historical-data download spec from parsed alerts."""
from __future__ import annotations

import json

import pandas as pd


def make_spec(alerts: pd.DataFrame, plan: str = "Basic Plan", pad_days: int = 1) -> dict:
    """One request per sport covering the alert kickoff dates (+/- pad_days), listing
    the events and market types needed. Feed the JSON to scripts/fetch_betfair_basic.py."""
    reqs = []
    for sport, g in alerts.groupby("sport", dropna=True):
        ko = pd.to_datetime(g["kickoff"], utc=True)
        ts = pd.to_datetime(g["ts"], utc=True)
        dates = ko.fillna(ts)
        d0 = (dates.min() - pd.Timedelta(days=pad_days)).date()
        d1 = (dates.max() + pd.Timedelta(days=pad_days)).date()
        mtypes = sorted({"MATCH_ODDS", *[m for m in g["bf_market"].dropna().unique()]})
        events = []
        seen = set()
        for r in g.itertuples(index=False):
            k = pd.Timestamp(r.kickoff) if pd.notna(r.kickoff) else None
            key = (str(r.home).lower(), str(r.away).lower(), k.isoformat() if k is not None else None)
            if key in seen:
                continue
            seen.add(key)
            e = {"home": r.home, "away": r.away}
            if k is not None:
                e["kickoff_utc"] = k.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            events.append(e)
        reqs.append({"sport": sport, "from": d0.isoformat(), "to": d1.isoformat(),
                     "discovery_market_type": "MATCH_ODDS", "market_types": mtypes, "events": events})
    return {"plan": plan, "requests": reqs}


def write_spec(alerts: pd.DataFrame, path: str, **kw) -> dict:
    spec = make_spec(alerts, **kw)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=1, ensure_ascii=False)
    return spec
