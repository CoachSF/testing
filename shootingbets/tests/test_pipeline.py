import bz2

import numpy as np
import pandas as pd
import pytest

from conftest import MIN, T0, to_bytes
from sbroi.betfair_files import load_markets
from sbroi.matching import build_event_index
from sbroi.pipeline import Config, MarketFileMissing, enrich
from test_pro_book import OFF, alerts, pro_lines


def test_enrich_end_to_end(basic_bz2):
    mk = load_markets(basic_bz2)
    idx = build_event_index(mk)
    alerts = pd.DataFrame([
        {"alert_id": 1, "ts": pd.to_datetime(T0, unit="ms", utc=True), "sport": "Soccer",
         "home": "Arsenal FC", "away": "Chelsea FC", "kickoff": pd.to_datetime(T0 + 120 * MIN, unit="ms", utc=True),
         "bf_market": "MATCH_ODDS", "sel_kind": "home", "sel_team": None, "sel_hc": None, "ref_odds": 2.2},
        {"alert_id": 2, "ts": pd.to_datetime(T0, unit="ms", utc=True), "sport": "Soccer",
         "home": "Everton", "away": "Fulham", "kickoff": pd.to_datetime(T0 + 120 * MIN, unit="ms", utc=True),
         "bf_market": "MATCH_ODDS", "sel_kind": "home", "sel_team": None, "sel_hc": None, "ref_odds": 2.0},
    ])
    df = enrich(alerts, mk, idx, Config(commissions=(0.0, 0.05)))
    a = df.set_index("alert_id")
    assert a.loc[1, "status"] == "ok" and a.loc[2, "status"] == "no_event_match"
    assert a.loc[1, "p_l0"] == 2.5 and a.loc[1, "p_l60"] == 2.5 and a.loc[1, "p_close"] == 2.32
    assert a.loc[1, "won"] == 1 and a.loc[1, "back_pl_l0_c5"] == pytest.approx(1.5 * 0.95)
    assert a.loc[1, "lay_pl_l0_c0"] == pytest.approx(-1.5)
    assert a.loc[1, "ref_back_pl"] == pytest.approx(1.2)
    assert a.loc[1, "mins_to_off"] == pytest.approx(120)
    assert a.loc[1, "es_15"] == -5
    assert a.loc[1, "es_240"] is None or np.isnan(a.loc[1, "es_240"])   # beyond the off -> not comparable
    assert a.loc[1, "fx_back_lay_h15_s0"] == pytest.approx((2.5 / 2.4 - 1) * 0.95)
    assert a.loc[1, "tp_back_lay_tp5_slNone_s0_reason"] == "tp"


# ----------------------------------------------------------------------------- unmapped market / double chance / AH ticks
def _pro_dir_with(tmp_path, *specs):
    """Write PRO-style market files (test_pro_book.pro_lines) and return the lazy store + event index."""
    for mid, kw in specs:
        (tmp_path / f"{mid}.bz2").write_bytes(bz2.compress(to_bytes(pro_lines(mid=mid, **kw))))
    mk = load_markets(str(tmp_path))
    return mk, build_event_index(mk)


def test_double_chance_kinds_swap_with_the_event(tmp_path):
    mk, idx = _pro_dir_with(tmp_path, ("1.800000001", dict(market_type="DOUBLE_CHANCE", teams=("Home or Draw", "Draw or Away"))))
    al = alerts(("Chelsea FC", "Arsenal FC"), ("Chelsea FC", "Arsenal FC"), ("Arsenal FC", "Chelsea FC"),
                ("Arsenal FC", "Chelsea FC"))                       # the first two list the teams the other way round
    al["bf_market"] = "DOUBLE_CHANCE"
    al["sel_kind"] = ["home_or_draw", "draw_or_away", "home_or_draw", "draw_or_away"]
    df = enrich(al, mk, idx, Config(commissions=(0.0,)))
    assert list(df["status"]) == ["ok"] * 4 and list(df["match_swapped"]) == [True, True, False, False]
    # "Chelsea or draw" is Betfair's "Draw or Away" (Chelsea is Betfair's away side); un-swapped alerts keep their kind
    assert list(df["bf_runner"]) == ["Draw or Away", "Home or Draw", "Home or Draw", "Draw or Away"]


def _women_dir(tmp_path):
    return _pro_dir_with(tmp_path, ("1.800000001", dict(name="Arsenal (W) v Chelsea (W)", teams=("Arsenal (W)", "Chelsea (W)"))))


def test_alt_names_are_tried_when_the_tagged_names_find_no_event(tmp_path):
    mk, idx = _women_dir(tmp_path)
    al = alerts(*[("x", "y")] * 5)
    al["home"], al["away"] = ["Arsenal", "Arsenal (W)", "Arsenal", "Arsenal", "Arsenal"], ["Chelsea", "Chelsea (W)"] + ["Chelsea"] * 3
    al["home_alt"] = [None, "Arsenal", "Arsenal (W)", np.nan, ""]
    al["away_alt"] = [None, "Chelsea", "Chelsea (W)", np.nan, ""]
    df = enrich(al, mk, idx, Config(commissions=(0.0,)))
    # 0: no alt names, plain names miss the (W) event (the marker penalty); 1: tagged names match first; 2: the plain
    # names miss, the alt names (carrying the marker) match; 3 and 4: NaN / empty alt names are no alt names
    assert list(df["status"]) == ["no_event_match", "ok", "ok", "no_event_match", "no_event_match"]
    assert list(df["match_names"].fillna("-")) == ["-", "primary", "alt", "-", "-"]
    assert df.loc[2, "match_event_id"] == "34000001" and df.loc[2, "match_score"] == 100.0
    assert pd.isna(df.loc[0, "match_event_id"]) and df.loc[0, "match_score"] < 78    # the failed primary try is what is reported


def test_tennis_gets_a_wider_kickoff_window_and_the_window_is_per_sport(tmp_path):
    mk, idx = _pro_dir_with(tmp_path, ("1.800000001", dict(name="Rafael Nadal v Roger Federer", teams=("Rafael Nadal", "Roger Federer"))))
    ev = idx["34000001"]
    al = alerts(*[("Rafael Nadal", "Roger Federer")] * 2)
    al["kickoff"] = pd.to_datetime(OFF - 90 * MIN, unit="ms", utc=True)               # Betfair says 90 minutes later
    al["sport"] = "Tennis"
    ev.sport = "Tennis"
    assert Config().kickoff_window_by_sport == {"Tennis": 720.0} and Config().kickoff_window_min == 20.0
    assert list(enrich(al, mk, idx, Config())["status"]) == ["ok", "ok"]
    for gap_min, status in ((600, "ok"), (719, "ok"), (730, "no_event_match")):     # order-of-play drift up to 12 hours
        late = al.assign(kickoff=pd.to_datetime(OFF - gap_min * MIN, unit="ms", utc=True))
        assert list(enrich(late, mk, idx, Config())["status"]) == [status] * 2, gap_min
    off = enrich(al, mk, idx, Config(kickoff_window_by_sport={}))
    assert list(off["status"]) == ["no_event_match"] * 2 and off["match_reason"].iloc[0] == "no_candidates_in_window"
    ev.sport = "Soccer"
    al["sport"] = "Soccer"
    assert list(enrich(al, mk, idx, Config())["status"]) == ["no_event_match"] * 2          # the wide window is tennis only
    assert list(enrich(al, mk, idx, Config(kickoff_window_by_sport={"Soccer": 120.0}))["status"]) == ["ok", "ok"]


def test_coverage_end_marks_alerts_pro_not_published_without_matching(tmp_path, monkeypatch):
    mk, idx = _pro_dir_with(tmp_path, ("1.800000001", {}))
    al = alerts(*[("Arsenal FC", "Chelsea FC")] * 4)
    al["kickoff"] = [pd.to_datetime(OFF - 60_000, unit="ms", utc=True), pd.to_datetime(OFF, unit="ms", utc=True),
                     pd.to_datetime(OFF + 3_600_000, unit="ms", utc=True), pd.NaT]
    import sbroi.pipeline as P
    calls = []
    real = P.match_event
    monkeypatch.setattr(P, "match_event", lambda *a, **k: calls.append(1) or real(*a, **k))
    df = enrich(al, mk, idx, Config(commissions=(0.0,), coverage_end_ms=OFF))
    assert list(df["status"]) == ["ok", "pro_not_published", "pro_not_published", "ok"]   # at/after the end; NaT is not judged
    assert len(calls) == 2 and df.loc[[1, 2], "match_event_id"].isna().all() and df.loc[[1, 2], "bf_market_id"].isna().all()
    assert len(enrich(al, mk, idx, Config(commissions=(0.0,))).query("status == 'pro_not_published'")) == 0   # default: off


def test_missing_bf_market_is_unmapped_market_after_a_successful_event_match(tmp_path):
    mk, idx = _pro_dir_with(tmp_path, ("1.800000001", {}))
    al = alerts(*[("Arsenal FC", "Chelsea FC")] * 5, ("Nobody FC", "Nowhere FC"))
    al["bf_market"] = pd.Series([None, np.nan, "", "  ", "MATCH_ODDS", None], dtype=object)
    df = enrich(al, mk, idx, Config(commissions=(0.0,)))
    assert list(df["status"]) == ["unmapped_market"] * 4 + ["ok", "no_event_match"]
    u = df.iloc[:4]
    assert u["match_event_id"].notna().all() and (u["bf_event_name"] == "Arsenal v Chelsea").all()
    assert u["match_score"].notna().all() and u["bf_market_id"].isna().all() and u["p_l0"].isna().all()
    assert df.loc[4, "bf_market_id"] == "1.800000001" and df.loc[4, "p_l0"] == 2.5


def test_fine_ladder_follows_the_betting_type_not_the_market_type(tmp_path):
    mk, idx = _pro_dir_with(
        tmp_path, ("1.800000001", {}),
        ("1.800000002", dict(market_type="HANDICAP", betting_type="ASIAN_HANDICAP_DOUBLE_LINE")),   # the US-sport style
        ("1.800000003", dict(market_type="ASIAN_HANDICAP", betting_type="ODDS", event="34000002", name="Everton v Fulham",
                             teams=("Everton", "Fulham"))),                                          # name says AH, ladder is not
        ("1.800000004", dict(market_type="ASIAN_HANDICAP", betting_type="ASIAN_HANDICAP_SINGLE_LINE")))
    al = alerts(("Arsenal FC", "Chelsea FC"), ("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham"), ("Arsenal FC", "Chelsea FC"))
    al["bf_market"] = ["MATCH_ODDS", "HANDICAP", "ASIAN_HANDICAP", "ASIAN_HANDICAP"]
    df = enrich(al, mk, idx, Config(commissions=(0.0,)))
    assert list(df["status"]) == ["ok"] * 4 and list(df["bf_market_id"]) == [f"1.80000000{i}" for i in (1, 2, 3, 4)]
    assert list(df["tick_ladder"]) == ["classic", "fine", "classic", "fine"]
    assert list(df["spread_ticks_l0"].isna()) == [False, True, False, True]
    es = [c for c in df.columns if c.startswith(("es_", "fx_", "tp_"))]
    assert es and list(df[es].notna().any(axis=1)) == [True, False, True, False]


def test_asian_handicap_has_no_tick_columns_but_keeps_every_price_column(tmp_path):
    mk, idx = _pro_dir_with(tmp_path, ("1.800000001", {}),
                            ("1.800000002", dict(market_type="HANDICAP", betting_type="ASIAN_HANDICAP_DOUBLE_LINE")))
    al = alerts(("Arsenal FC", "Chelsea FC"), ("Arsenal FC", "Chelsea FC"))
    al["bf_market"] = ["MATCH_ODDS", "HANDICAP"]
    df = enrich(al, mk, idx, Config(commissions=(0.0, 0.05)))
    assert list(df["status"]) == ["ok", "ok"] and list(df["tick_ladder"]) == ["classic", "fine"]
    mo, ah = df.iloc[0], df.iloc[1]
    tickish = [c for c in df.columns if c.startswith(("es_", "fx_", "tp_"))]
    assert tickish and mo[tickish].notna().any() and ah[tickish].isna().all()     # the classic row has them, the fine row none
    assert mo["spread_ticks_l0"] == 2 and np.isnan(ah["spread_ticks_l0"])
    for c in ("bb_l0", "bl_l0", "bb_l0_size", "p_l0", "p_close", "bb_close", "bl_close", "xback_pl_l0_c0", "xlay_pl_l0_c0",
              "xback_pl_l0_c5", "xclose_back_pl_c0", "back_pl_l0_c5", "lay_pl_l0_c0", "xfx_back_lay_h10", "xfx_lay_back_h10",
              "xfx_back_lay_h100000", "ref_back_pl", "mins_to_off"):
        assert ah[c] == pytest.approx(mo[c]), c                                      # same prices, same P/L
    assert ah["xfx_back_lay_h10"] == pytest.approx((2.48 / 2.40 - 1) * 0.95) and ah["xfx_back_lay_h10_reason"] == "time"


def test_a_listed_market_whose_file_is_gone_raises_but_an_absent_type_is_no_market(tmp_path):
    mk, idx = _pro_dir_with(tmp_path, ("1.800000001", {}))
    al = alerts(("Arsenal FC", "Chelsea FC"))
    other = al.assign(bf_market="OVER_UNDER_25")                                  # the event has no market of this type
    assert list(enrich(other, mk, idx, Config(commissions=(0.0,)))["status"]) == ["no_market"]
    del mk["1.800000001"]                                                         # still in the event index (the manifest), file gone
    with pytest.raises(MarketFileMissing, match=r"34000001.*MATCH_ODDS.*1\.800000001.*trimmed"):
        enrich(al, mk, idx, Config(commissions=(0.0,)))
    assert list(enrich(other, mk, idx, Config(commissions=(0.0,)))["status"]) == ["no_market"]   # unchanged by the missing file
