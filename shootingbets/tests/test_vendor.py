"""sbroi.vendor + scripts/build_alerts.py --vendor, on synthetic result rows (NOT real results)."""
import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from sbroi.vendor import NEW_COLS, join_vendor, load_vendor, settle_from_score
from test_alerts import CLOSE, OPEN, S1, S2, _raw, text

sys.path.insert(0, str(ROOT / "scripts"))
COLS = ["starts", "sport", "league", "home", "away", "market", "selection", "min_odds", "b365_line", "b365_value_odds",
        "score", "result", "pl_50", "date"]


def row(starts="23-11-2026 07:00 PM", home="Bournemouth", away="Leicester City", market="Spread", sel="Away", min_odds="1.89",
        line="-0.5", odds="3.10", score="2-3", result="L", pl="-50.00"):
    return [starts, "Soccer", "England - Premier League Cup U21", home, away, market, sel, min_odds, line, odds, score,
            result, pl, starts[:10]]


def write(path, rows, as_string=False):
    d = {"source": "s", "fetched_utc": "2026-11-24T12:24:58.347Z", "note": "n", "columns": COLS, "rows": rows}
    path.write_text(json.dumps(json.dumps(d) if as_string else d), encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- load_vendor
def test_load_vendor_reads_both_file_shapes(tmp_path):
    rows = [row(), row(starts="24-11-2026 12:30 AM", market="Moneyline", sel="Home", line="-", odds="6.00", score="1-1",
                       result="L", pl="1,234.50"),
            row(market="Totals", sel="Over", line="2.5", score="postponed", result="W", pl="48.50")]
    a, b = load_vendor(write(tmp_path / "a.json", rows)), load_vendor(write(tmp_path / "b.json", rows, as_string=True))
    pd.testing.assert_frame_equal(a, b)                              # the JSON object, or a JSON string holding it
    assert list(a.columns) == ["starts", "sport", "league", "home", "away", "market", "selection", "min_odds", "b365_line",
                               "b365_odds", "score", "score_home", "score_away", "result", "pl_50"]
    assert list(a["starts"]) == [pd.Timestamp("2026-11-23 19:00", tz="UTC"), pd.Timestamp("2026-11-24 00:30", tz="UTC"),
                                 pd.Timestamp("2026-11-23 19:00", tz="UTC")]            # 12:30 AM is 00:30; UTC
    np.testing.assert_allclose(a["b365_line"], [-0.5, np.nan, 2.5])                       # "-" is NaN
    np.testing.assert_allclose(a["b365_odds"], [3.1, 6.0, 3.1])
    np.testing.assert_allclose(a["min_odds"], [1.89] * 3)
    np.testing.assert_allclose(a["pl_50"], [-50.0, 1234.5, 48.5])                         # thousands comma
    assert list(a["score_home"].astype(object).where(a["score_home"].notna(), None)) == [2, 1, None]
    assert list(a["score_away"].astype(object).where(a["score_away"].notna(), None)) == [3, 1, None]
    assert str(a["score_home"].dtype) == "Int64" and list(a["result"]) == ["L", "L", "W"]
    assert list(a["market"]) == ["Spread", "Moneyline", "Totals"] and a.loc[0, "home"] == "Bournemouth"


# ----------------------------------------------------------------------------- settle_from_score
@pytest.mark.parametrize("kind,side,line,scope,sh,sa,want", [
    ("total", "over", 2.5, None, 2, 1, True), ("total", "under", 2.5, None, 2, 1, False),
    ("total", "under", 3.5, None, 2, 1, True), ("total", "over", 3.5, None, 2, 1, False),
    ("total", "over", 3.0, None, 2, 1, None), ("total", "under", 3.0, None, 2, 1, None),          # push
    ("handicap", "home", -1.5, None, 3, 1, True), ("handicap", "home", -1.5, None, 2, 1, False),
    ("handicap", "home", -1.0, None, 2, 1, None),                                                 # push
    ("handicap", "away", 1.5, None, 2, 1, True), ("handicap", "away", -0.5, None, 1, 1, False),
    ("handicap", "home", 0.5, None, 1, 1, True), ("handicap", "away", -1.5, "sets", 0, 2, True),
    ("moneyline", "home", None, None, 2, 1, True), ("moneyline", "away", None, None, 2, 1, False),
    ("moneyline", "away", None, None, 1, 2, True), ("moneyline", "home", None, None, 1, 1, False),
    ("moneyline", "draw", None, None, 1, 1, True), ("moneyline", "draw", None, None, 2, 1, False),
    ("moneyline", "home", None, "dnb", 1, 1, None), ("moneyline", "away", None, "dnb", 0, 0, None),   # draw no bet: stake back
    ("moneyline", "draw", None, "dnb", 1, 1, None), ("moneyline", "home", None, "dnb", 2, 1, True),
    ("moneyline", "home", None, "ot_incl", 3, 2, True),
    ("total", "over", 2.5, None, None, 1, None), ("moneyline", "home", None, None, 1, np.nan, None),   # no score
    ("moneyline", "home", None, None, pd.NA, pd.NA, None), ("total", "over", np.nan, None, 2, 1, None),
    ("spread", "home", -1.5, None, 2, 1, None)])
def test_settle_from_score(kind, side, line, scope, sh, sa, want):
    assert settle_from_score(kind, side, line, scope, sh, sa) is want


# ----------------------------------------------------------------------------- join_vendor
def A(i, mkt, side, line, minodds, home="Bournemouth", away="Leicester City", kick="2026-11-23 19:00", scope=None):
    return {"alert_id": i, "home_raw": home, "away_raw": away, "kickoff": pd.Timestamp(kick, tz="UTC"), "mkt_kind": mkt,
            "sel_side": side, "line": line, "scope": scope, "min_odds": minodds}


def test_join_picks_the_vendor_row_of_the_same_fixture_market_and_selection(tmp_path):
    v = load_vendor(write(tmp_path / "v.json", [                                    # one fixture, final score 2-3
        row(market="Moneyline", sel="Home", line="-", odds="2.50", result="L"),
        row(market="Moneyline", sel="Away", line="-", odds="3.10", result="W"),
        row(market="Spread", sel="Away", line="-0.5", odds="1.95", result="W"),
        row(market="Spread", sel="Away", line="-1.5", odds="2.60", result="L"),
        row(market="Totals", sel="Over", line="2.5", odds="1.90", result="W"),
        row(market="Totals", sel="Under", line="2.5", odds="2.00", result="L"),
        row(home="Foo", away="Bar", starts="23-11-2026 08:00 PM", market="Spread", sel="Home", line="-1.5", odds="1.80",
            score="2-1", result="L")]))
    al = pd.DataFrame([
        A(1, "moneyline", "away", np.nan, 3.0),                                              # -> the Away moneyline row
        A(2, "handicap", "away", -1.5, 2.5),                                                 # two Away spreads: the -1.5 one
        A(3, "handicap", "away", -0.5, 1.89),                                                # ... and here the -0.5 one
        A(4, "handicap", "away", -2.5, 2.0),                                                 # no -2.5 row: the first, other line
        A(5, "total", "under", 2.5, 1.9),
        A(6, "moneyline", "draw", np.nan, 3.4),                                              # no Draw row: not found
        A(7, "moneyline", "home", np.nan, 2.5, kick="2026-11-23 18:59"),                     # other kickoff: not found
        A(8, "moneyline", "home", np.nan, 2.5, home="  BOURNEMOUTH ", away="leicester   city"),   # spacing / case
        A(9, "handicap", "home", -1.0, 1.9, home="Foo", away="Bar", kick="2026-11-23 20:00"),     # push on its own line
        A(10, "total", "over", 2.5, 1.9, home="Foo", away="Bar", kick="2026-11-23 20:00")], index=range(100, 110))
    out = join_vendor(al, v)
    assert list(out.columns) == list(al.columns) + NEW_COLS and list(out.index) == list(range(100, 110))
    assert list(out["v_found"]) == [True, True, True, True, True, False, False, True, True, False]
    assert list(out["v_b365_odds"].fillna(-1)) == [3.1, 2.6, 1.95, 1.95, 2.0, -1, -1, 2.5, 1.8, -1]
    assert list(out["v_b365_line"].fillna(99)) == [99, -1.5, -0.5, -0.5, 2.5, 99, 99, 99, -1.5, 99]
    assert list(out["v_same_line"]) == [True, True, True, False, True, False, False, True, False, False]
    assert list(out["v_score"].fillna("-")) == ["2-3"] * 5 + ["-", "-", "2-3", "2-1", "-"]
    assert list(out["v_score_home"].astype(float).fillna(-1)) == [2, 2, 2, 2, 2, -1, -1, 2, 2, -1]
    assert list(out["v_score_away"].astype(float).fillna(-1)) == [3, 3, 3, 3, 3, -1, -1, 3, 1, -1]
    assert list(out["v_result"].fillna("-")) == ["W", "L", "W", "W", "L", "-", "-", "L", "L", "-"]
    assert list(out["v_pl50"].fillna(0)) == [-50.0] * 5 + [0, 0, -50.0, -50.0, 0]
    # the alert's own selection settled from the score (2-3): away +, away -1.5 lost, away -0.5 won, away -2.5 lost, ...
    assert list(out["own_won"].astype(object).where(out["own_won"].notna(), None)) == [
        True, False, True, False, False, None, None, False, None, None]
    assert str(out["own_won"].dtype) == "boolean"
    np.testing.assert_allclose(out["own_pl_min"], [2.0, -1, 0.89, -1, -1, np.nan, np.nan, -1, np.nan, np.nan])
    np.testing.assert_allclose(out["v_pl_b365"], [2.1, -1, 0.95, 0.95, -1, np.nan, np.nan, -1, -1, np.nan])
    assert list(out["alert_id"]) == list(al["alert_id"]) and out["min_odds"].tolist() == al["min_odds"].tolist()


# ----------------------------------------------------------------------------- scripts/build_alerts.py --vendor
def test_build_alerts_script_adds_and_summarises_the_vendor_columns(tmp_path, capsys):
    ba = importlib.import_module("build_alerts")
    checked = lambda sel: text(sel=sel, tail=("✅Checked: Tritium +1.5 --> 1.90✅",))  # noqa: E731
    bodies = [(OPEN, False), (checked("Tritium +1.5"), False),                                   # id 2: on Chievo v Tritium
              (text(sel="Over 2.5"), False),                                                     # id 3: unchecked, not listed
              (S2.replace("Unchecked", "Checked"), False),                                       # id 4: Cruzeiro v Athletic Club
              (CLOSE, False)]
    raw = tmp_path / "raw.json"
    _raw(raw, bodies)
    vend = write(tmp_path / "v.json", [
        row(starts="03-10-2026 06:45 PM", home="Chievo", away="Tritium", market="Spread", sel="Away", min_odds="1.81",
            line="1.5", odds="1.90", score="0-2", result="W", pl="45.00"),                       # own selection won: agrees
        row(starts="03-10-2026 06:00 PM", home="Cruzeiro", away="Athletic Club", market="Moneyline", sel="Home",
            min_odds="1.62", line="-", odds="1.70", score="1-1", result="W", pl="35.00"),        # own lost, vendor W: disagrees
        row(starts="01-10-2026 06:00 PM", home="Other", away="Fixture", market="Moneyline", sel="Home", line="-",
            odds="2.00", score="1-0", result="W", pl="50.00")])
    out, mout = tmp_path / "a.csv", tmp_path / "m.csv"
    args = ["--in", str(raw), "--out", str(out), "--messages-out", str(mout)]
    assert ba.main(args + ["--vendor", str(vend)]) == 0
    o = capsys.readouterr().out
    assert o.isascii() and "vendor file: 3 rows, sum pl_50 / (50 * rows) = +0.8667" in o     # (45 + 35 + 50) / 150
    assert "  checked: 2 of 2" in o and "  none: 0 of 1" in o
    assert "1 agree of 2 compared (0 not settled by the score, 0 on another line)" in o and "disagree (alert ids): 4" in o
    assert "own_pl_min: n 2  strike 50.0%  ROI -9.50%" in o and "v_pl_b365: n 2  strike 100.0%  ROI +80.00%" in o
    a = pd.read_csv(out).set_index("alert_id")
    assert list(a.columns[-len(NEW_COLS):]) == NEW_COLS
    assert list(a["v_found"]) == [True, False, True] and a.loc[2, "own_pl_min"] == pytest.approx(0.81)
    assert a.loc[4, "own_pl_min"] == -1 and a.loc[4, "v_pl_b365"] == pytest.approx(0.7) and pd.isna(a.loc[3, "own_pl_min"])
    # no vendor file (explicit missing path): silently skipped, no vendor columns, no vendor lines
    assert ba.main(args + ["--vendor", str(tmp_path / "missing.json")]) == 0
    o = capsys.readouterr().out
    assert "vendor" not in o and "v_found" not in pd.read_csv(out).columns
    # a vendor file that is not valid -> exit 2
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert ba.main(args + ["--vendor", str(bad)]) == 2 and "bad input" in capsys.readouterr().out
