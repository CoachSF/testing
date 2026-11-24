import bz2
import json

import numpy as np
import pytest

from conftest import MIN, T0, basic_market_lines, md, to_bytes
from sbroi import ticks
from sbroi.betfair_files import load_markets, parse_stream_bytes
from sbroi.matching import build_event_index, match_event, resolve_runner
from sbroi.pricing import ltp_at, ltp_path, preoff_close
from sbroi.roi import (Outcome, back_pl, benjamini_hochberg, clv_stats, lay_pl, max_drawdown,
                       outcome_from_status, summarize)
from sbroi.scalping import event_study, fixed_horizon, green_back_lay, green_lay_back, tp_sl
from sbroi.textnorm import name_score, normalize_name, split_event_name


# ------------------------------------------------------------------ ticks
def test_ladder_shape():
    assert ticks.N_TICKS == 350
    assert ticks.LADDER[0] == 1.01 and ticks.LADDER[-1] == 1000.0
    assert ticks.LADDER[99] == 2.0 and ticks.LADDER[100] == 2.02
    assert ticks.is_valid_price(3.05) and not ticks.is_valid_price(3.03)


def test_tick_math():
    assert ticks.move(2.0, 1) == 2.02
    assert ticks.move(2.02, -1) == 2.0
    assert ticks.move(1.01, -5) == 1.01           # clamped
    assert ticks.tick_diff(2.5, 2.3) == -10       # 0.02 steps
    assert ticks.tick_diff(3.0, 3.1) == 2         # 3.05, 3.1
    assert ticks.snap(3.03, "nearest") == 3.05 or ticks.snap(3.03, "nearest") == 3.0
    assert ticks.snap(3.03, "down") == 3.0 and ticks.snap(3.03, "up") == 3.05
    assert ticks.tick_size(5.0) == pytest.approx(0.1)


# ------------------------------------------------------------- stream files
def test_parse_basic(basic_bz2):
    mk = load_markets(basic_bz2)
    m = mk["1.200000001"]
    assert m.data_level == "BASIC"
    assert m.event_name == "Arsenal v Chelsea" and m.sport == "Soccer"
    assert m.in_play_start_ms == T0 + 120 * MIN
    assert m.close_ms == T0 + 240 * MIN
    assert m.off_ms() == T0 + 120 * MIN
    assert m.market_base_rate == 5.0
    ars = m.runners[(1001, 0.0)]
    assert ars.status == "WINNER" and ars.name == "Arsenal"
    assert m.status_at(T0)[0] == "OPEN" and m.status_at(T0 + 130 * MIN) == ("OPEN", True)


def test_tar_and_no_lookahead(basic_tar):
    mk = load_markets(basic_tar)
    assert set(mk) == {"1.200000001", "1.200000002"}
    r = mk["1.200000001"].runners[(1001, 0.0)]
    # at t0 the last trade was the 2.50 at t0-30m; the 2.40 at t0+10m must not leak in
    p = ltp_at(r, T0)
    assert p.price == 2.5 and p.age_ms == 30 * MIN
    assert ltp_at(r, T0 + 10 * MIN).price == 2.4
    assert ltp_at(r, T0 - 31 * MIN) is None
    assert [x[1] for x in ltp_path(r, T0, T0 + 60 * MIN)] == [2.4, 2.3]
    assert preoff_close(mk["1.200000001"], r).price == 2.32


def test_advanced_and_pro_ladders():
    mid = "1.3"
    lines = [
        {"op": "mcm", "pt": T0, "mc": [{"id": mid, "marketDefinition": md()}]},
        {"op": "mcm", "pt": T0 + MIN, "mc": [{"id": mid, "rc": [
            {"id": 1001, "batb": [[0, 2.5, 100], [1, 2.48, 50]], "batl": [[0, 2.52, 80]], "ltp": 2.5}]}]},
        {"op": "mcm", "pt": T0 + 2 * MIN, "mc": [{"id": mid, "rc": [{"id": 1001, "batb": [[0, 2.48, 50], [1, 0, 0]]}]}]},
    ]
    m = parse_stream_bytes(to_bytes(lines))[0]
    assert m.data_level == "ADVANCED"
    r = m.runners[(1001, 0.0)]
    assert r.bb_p == [2.5, 2.48] and r.bl_p == [2.52]
    lines_pro = [
        {"op": "mcm", "pt": T0, "mc": [{"id": mid, "marketDefinition": md()}]},
        {"op": "mcm", "pt": T0 + MIN, "mc": [{"id": mid, "img": True, "rc": [
            {"id": 1001, "atb": [[2.5, 10], [2.48, 5]], "atl": [[2.52, 7], [2.54, 9]]}]}]},
        {"op": "mcm", "pt": T0 + 2 * MIN, "mc": [{"id": mid, "rc": [{"id": 1001, "atb": [[2.5, 0]], "atl": [[2.52, 0]]}]}]},
    ]
    m2 = parse_stream_bytes(to_bytes(lines_pro))[0]
    assert m2.data_level == "PRO"
    r2 = m2.runners[(1001, 0.0)]
    assert r2.bb_p == [2.5, 2.48] and r2.bl_p == [2.52, 2.54]


def test_handicap_runner_keys():
    mid = "1.4"
    d = md(market_type="ASIAN_HANDICAP")
    d["runners"] = [{"id": 1001, "name": "Arsenal", "hc": -0.5, "sortPriority": 1, "status": "ACTIVE"},
                    {"id": 1001, "name": "Arsenal", "hc": -1.0, "sortPriority": 3, "status": "ACTIVE"}]
    lines = [{"op": "mcm", "pt": T0, "mc": [{"id": mid, "marketDefinition": d}]},
             {"op": "mcm", "pt": T0 + MIN, "mc": [{"id": mid, "rc": [{"id": 1001, "hc": -0.5, "ltp": 1.9},
                                                                    {"id": 1001, "hc": -1.0, "ltp": 2.4}]}]}]
    m = parse_stream_bytes(to_bytes(lines))[0]
    assert m.runners[(1001, -0.5)].ltp_p == [1.9] and m.runners[(1001, -1.0)].ltp_p == [2.4]


def test_corrupt_line_tolerated():
    raw = to_bytes(basic_market_lines()) + b"{not json\n"
    assert parse_stream_bytes(raw)[0].market_id == "1.200000001"


# ----------------------------------------------------------------- roi math
def test_pl_with_commission():
    win, lose, void = Outcome(1, 0, 0, "WINNER", True), Outcome(0, 1, 0, "LOSER", True), Outcome(0, 0, 1, "REMOVED", True)
    assert back_pl(3.0, win, 0.05) == pytest.approx(1.9)
    assert back_pl(3.0, lose, 0.05) == -1.0
    assert back_pl(3.0, void, 0.05) == 0.0
    assert lay_pl(3.0, lose, 0.05) == pytest.approx(0.95)
    assert lay_pl(3.0, win, 0.05) == -2.0
    assert outcome_from_status("WINNER").win == 1 and not outcome_from_status(None).known


def test_summary_and_stats():
    pl = np.array([1.0, -1.0, -1.0, 2.0])
    s = summarize(pl, odds=np.array([2.0, 2.0, 2.0, 3.0]), won=np.array([1, 0, 0, 1]), n_boot=2000)
    assert s.roi == pytest.approx(0.25) and s.n == 4
    assert s.roi_ci_lo <= s.roi <= s.roi_ci_hi
    assert max_drawdown(pl) == pytest.approx(2.0)
    c = clv_stats(np.array([2.2, 2.0]), np.array([2.0, 2.0]))
    assert c["ev_vs_close"] == pytest.approx(0.05) and c["pct_beat_close"] == 0.5
    assert benjamini_hochberg([0.001, 0.5, 0.02, 0.9], q=0.10) == [True, False, True, False]


# ----------------------------------------------------------------- scalping
def test_green_formulas():
    assert green_back_lay(3.0, 2.5) == pytest.approx(0.2)
    assert green_back_lay(3.0, 2.5, 0.05) == pytest.approx(0.19)
    assert green_back_lay(2.5, 3.0) == pytest.approx(2.5 / 3 - 1)
    assert green_lay_back(2.5, 3.0) == pytest.approx(1 - 2.5 / 3)


def test_scalp_sims(basic_bz2):
    m = load_markets(basic_bz2)["1.200000001"]
    r = m.runners[(1001, 0.0)]
    off = m.off_ms()
    res = fixed_horizon(r, T0, 15 * MIN, off)          # 2.50 -> 2.40
    assert res.green_pl == pytest.approx(2.5 / 2.4 - 1) and res.ticks_moved == -5
    res_off = fixed_horizon(r, T0, 999 * MIN, off)     # capped just before the off -> 2.32
    assert res_off.exit_reason == "off" and res_off.exit_price == 2.32
    tp = tp_sl(r, T0, tp_ticks=10, sl_ticks=5, max_ms=200 * MIN, off_ms=off)  # target 2.30 reached at t0+30
    assert tp.exit_reason == "tp" and tp.exit_price == 2.3
    tp_through = tp_sl(r, T0, tp_ticks=10, sl_ticks=5, max_ms=200 * MIN, off_ms=off, fill="through")
    assert tp_through.exit_reason == "off"           # never traded below 2.30 pre-off
    lb = tp_sl(r, T0, tp_ticks=3, sl_ticks=4, max_ms=200 * MIN, off_ms=off, side="lay_back")
    assert lb.exit_reason == "sl" and lb.green_pl < 0
    es = event_study(r, T0, [-30 * MIN, 15 * MIN, 60 * MIN, 130 * MIN], off)
    assert es[15 * MIN] == -5 and es[60 * MIN] == -10 and es[130 * MIN] is None and es[-30 * MIN] == 0


# ----------------------------------------------------------------- matching
def test_name_normalisation():
    assert normalize_name("Manchester United FC") == "manchester united"
    assert normalize_name("Man Utd") == "manchester united"
    assert normalize_name("Atlético de Madrid") == "atletico madrid"
    assert "u21" in normalize_name("England U-21")
    assert name_score("Arsenal Women", "Arsenal (W)") == 100.0
    assert name_score("Arsenal", "Arsenal (W)") < 80
    assert split_event_name("Boston Celtics @ Miami Heat") == ("Miami Heat", "Boston Celtics", True)
    assert split_event_name("Arsenal v Chelsea") == ("Arsenal", "Chelsea", False)


def test_match_and_resolve(basic_bz2):
    mk = load_markets(basic_bz2)
    idx = build_event_index(mk)
    em = match_event(idx, "Soccer", "Arsenal FC", "Chelsea FC", kickoff_ms=T0 + 125 * MIN)
    assert em.event_id == "34000001" and em.reason == "ok" and not em.swapped
    em2 = match_event(idx, "Soccer", "Chelsea", "Arsenal", kickoff_ms=T0 + 120 * MIN)
    assert em2.event_id == "34000001" and em2.swapped
    assert match_event(idx, "Soccer", "Arsenal", "Chelsea", kickoff_ms=T0 + 600 * MIN).event_id is None
    assert match_event(idx, "Soccer", "Everton", "Fulham", kickoff_ms=T0 + 120 * MIN).event_id is None
    m = mk["1.200000001"]
    ev = idx["34000001"]
    assert resolve_runner(m, ev, "home").selection_id == 1001
    assert resolve_runner(m, ev, "away").selection_id == 1002
    assert resolve_runner(m, ev, "draw").selection_id == 58805
    assert resolve_runner(m, ev, "team", team="Chelsea FC").selection_id == 1002
