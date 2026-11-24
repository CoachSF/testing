"""The vendor's public results page (data/vendor_results/sb_vendor_results.json) scored against the parsed alerts.

The page only lists alerts for which Bet365 offered at least the minimum odds, and its W / L and P/L refer to the BET365
line and odds, which can differ from the alert's own line. ``settle_from_score`` settles the alert's own selection from
the final score instead.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

_MKT = {"Moneyline": "moneyline", "Spread": "handicap", "Totals": "total"}
NEW_COLS = ["v_found", "v_b365_line", "v_b365_odds", "v_score", "v_score_home", "v_score_away", "v_result", "v_pl50",
            "v_same_line", "own_won", "own_pl_min", "v_pl_b365"]


def _norm(s) -> str:
    return " ".join(str(s).casefold().split())


def load_vendor(path) -> pd.DataFrame:
    """One row per vendor result; ``path`` holds the JSON object, or a JSON string that holds it."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if isinstance(d, str):
        d = json.loads(d)
    r = pd.DataFrame(d["rows"], columns=d["columns"])
    sc = r["score"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    return pd.DataFrame({
        "starts": pd.to_datetime(r["starts"], format="%d-%m-%Y %I:%M %p", utc=True), "sport": r["sport"],
        "league": r["league"], "home": r["home"], "away": r["away"], "market": r["market"], "selection": r["selection"],
        "min_odds": num(r["min_odds"]), "b365_line": num(r["b365_line"]), "b365_odds": num(r["b365_value_odds"]),
        "score": r["score"], "score_home": num(sc[0]).astype("Int64"), "score_away": num(sc[1]).astype("Int64"),
        "result": r["result"], "pl_50": num(r["pl_50"].astype(str).str.replace(",", "", regex=False))})


def settle_from_score(mkt_kind, sel_side, line, scope, sh, sa):
    """True (the selection won), False (lost) or None (push, no score, or a DNB draw)."""
    if pd.isna(sh) or pd.isna(sa):
        return None
    if mkt_kind == "moneyline":
        if sh == sa and scope == "dnb":
            return None
        if sel_side == "draw":
            return bool(sh == sa)
        return bool(sh > sa) if sel_side == "home" else bool(sa > sh)
    if pd.isna(line):
        return None
    if mkt_kind == "total":
        t = sh + sa
        return None if t == line else bool((t > line) == (sel_side == "over"))
    if mkt_kind == "handicap":
        m = (sh - sa if sel_side == "home" else sa - sh) + line
        return None if m == 0 else bool(m > 0)
    return None


def join_vendor(alerts: pd.DataFrame, vendor: pd.DataFrame) -> pd.DataFrame:
    """``alerts`` + NEW_COLS. Fixture key = whitespace-normalised casefolded home_raw / away_raw and kickoff == starts;
    of that fixture's vendor rows the one with the alert's market kind and selection side is kept (the one on the
    alert's own line when there are several)."""
    by_fix: dict = {}
    for r in vendor.itertuples(index=False):
        by_fix.setdefault((_norm(r.home), _norm(r.away), int(r.starts.timestamp())), []).append(r)
    rows, own, goals = [], [], []
    for a in alerts.itertuples(index=False):
        c = [r for r in by_fix.get((_norm(a.home_raw), _norm(a.away_raw), int(pd.Timestamp(a.kickoff).timestamp())), [])
             if _MKT.get(r.market) == a.mkt_kind and _norm(r.selection) == _norm(a.sel_side)]
        r = next((r for r in c if a.mkt_kind == "moneyline" or r.b365_line == a.line), c[0] if c else None)
        if r is None:
            rows.append(dict.fromkeys(NEW_COLS, np.nan) | {"v_found": False, "v_same_line": False})
            own.append(None)
            goals.append((None, None))
            continue
        won = settle_from_score(a.mkt_kind, a.sel_side, a.line, a.scope, r.score_home, r.score_away)
        own.append(won)
        goals.append(tuple(None if pd.isna(g) else int(g) for g in (r.score_home, r.score_away)))
        rows.append({"v_found": True, "v_b365_line": r.b365_line, "v_b365_odds": r.b365_odds, "v_score": r.score,
                     "v_result": r.result, "v_pl50": r.pl_50,
                     "v_same_line": bool(a.mkt_kind == "moneyline" or r.b365_line == a.line),
                     "own_pl_min": np.nan if won is None else (a.min_odds - 1.0 if won else -1.0),
                     "v_pl_b365": r.b365_odds - 1.0 if r.result == "W" else -1.0 if r.result == "L" else np.nan})
    new = pd.DataFrame(rows, index=alerts.index, columns=NEW_COLS)
    new["own_won"] = pd.array(own, dtype="boolean")
    new["v_score_home"], new["v_score_away"] = (pd.array([g[i] for g in goals], dtype="Int64") for i in (0, 1))
    return pd.concat([alerts, new], axis=1)
