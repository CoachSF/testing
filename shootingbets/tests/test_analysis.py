"""sbroi.analysis on small HAND-BUILT frames: every expected number is worked out independently of the code."""
import json

import numpy as np
import pandas as pd
import pytest

from sbroi import analysis as A
from sbroi import roi

NAN = float("nan")
C, TAG = 0.048, "c4p8"
T = pd.Timestamp("2026-10-01 10:00", tz="UTC")


def bpl(o, won, c=C, void=False):          # back 1 unit of stake: win (o-1) less commission on the win, lose 1
    return 0.0 if void else ((o - 1) * (1 - c) if won else -1.0)


def lpl(o, won, c=C, void=False):          # lay for 1 unit of backer's stake: runner loses -> +1 less commission
    return 0.0 if void else (-(o - 1) if won else (1 - c))


def mkframe(specs, lat=(0, 60), comms=(0.0, C)):
    """Hand-built enriched frame: one dict per alert, defaults overridable, P/L columns derived from the formulas above."""
    from sbroi.pipeline import _ctag
    rows = []
    for i, s in enumerate(specs):
        r = {"alert_id": i, "ts": T + pd.Timedelta(minutes=10 * i), "match_event_id": f"E{i}", "status": "ok",
             "result_known": True, "won": 1.0, "void": 0.0, "in_play_at_alert": False, "p_l0": 2.0, "bb_l0": 1.98,
             "bl_l0": 2.02, "p_close": 2.0, "ref_odds": 2.2}
        r.update({k: v for k, v in s.items() if k in r})
        known = r["status"] == "ok" and r["result_known"] and not np.isnan(r["won"])
        void = known and r["void"] == 1.0
        for L in lat:
            for k in ("p", "bb", "bl"):
                r.setdefault(f"{k}_l{L}", r[f"{k}_l0"])
            r.update({f"{k}_l{L}": s[f"{k}_l{L}"] for k in ("p", "bb", "bl") if f"{k}_l{L}" in s})
            for c in comms:
                t = _ctag(c)
                for col, odds, fn in ((f"back_pl_l{L}_{t}", f"p_l{L}", bpl), (f"lay_pl_l{L}_{t}", f"p_l{L}", lpl),
                                      (f"xback_pl_l{L}_{t}", f"bb_l{L}", bpl), (f"xlay_pl_l{L}_{t}", f"bl_l{L}", lpl)):
                    r[col] = fn(r[odds], r["won"], c, void) if known and not np.isnan(r[odds]) else NAN
        r["ref_back_pl"] = bpl(r["ref_odds"], r["won"], 0.0, void) if known else NAN
        r["close_back_pl_c0"] = bpl(r["p_close"], r["won"], 0.0, void) if known else NAN
        r.update(s)
        rows.append(r)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- b. leg_frame
def test_leg_frame_exact_values_back_lay_ltp_book():
    # row 0 wins, row 1 loses; row 1 is given the EARLIER timestamp to prove the sort by ts
    df = mkframe([
        {"won": 1.0, "p_l0": 2.5, "bb_l0": 2.48, "bl_l0": 2.52, "p_close": 2.0},
        {"won": 0.0, "p_l0": 3.0, "bb_l0": 2.9, "bl_l0": 3.1, "p_close": 3.3, "ts": T - pd.Timedelta(minutes=5)}])
    b = A.leg_frame(df, "back", "ltp")
    assert list(b["alert_id"]) == [1, 0] and b["ts"].is_monotonic_increasing
    assert list(b["odds"]) == [3.0, 2.5] and list(b["won"]) == [0.0, 1.0] and list(b["liability"]) == [1.0, 1.0]
    assert b["pl"].tolist() == pytest.approx([-1.0, 1.5 * 0.952])               # 1.428
    assert b["clv"].tolist() == pytest.approx([3.0 / 3.3 - 1, 2.5 / 2.0 - 1])   # -0.0909, +0.25
    assert list(b["event"]) == ["E1", "E0"] and b.attrs["n_input"] == 2

    lay = A.leg_frame(df, "lay", "ltp")
    assert lay["pl"].tolist() == pytest.approx([0.952, -1.5])
    assert list(lay["won"]) == [1.0, 0.0]                                        # a lay "wins" when the runner loses
    assert lay["liability"].tolist() == pytest.approx([2.0, 1.5])
    assert lay["clv"].tolist() == pytest.approx([3.3 / 3.0 - 1, 2.0 / 2.5 - 1])  # +0.1, -0.2

    bb = A.leg_frame(df, "back", "book")
    assert list(bb["odds"]) == [2.9, 2.48] and bb["pl"].tolist() == pytest.approx([-1.0, 1.48 * 0.952])
    assert bb["clv"].tolist() == pytest.approx([2.9 / 3.3 - 1, 2.48 / 2.0 - 1])
    bl = A.leg_frame(df, "lay", "book")
    assert list(bl["odds"]) == [3.1, 2.52] and bl["pl"].tolist() == pytest.approx([0.952, -1.52])
    assert bl["liability"].tolist() == pytest.approx([2.1, 1.52])
    assert bl["clv"].tolist() == pytest.approx([3.3 / 3.1 - 1, 2.0 / 2.52 - 1])

    ref = A.leg_frame(df, "back", "ref", commission=0.0)                         # bookmaker quote, no commission
    assert list(ref["odds"]) == [2.2, 2.2] and ref["pl"].tolist() == pytest.approx([-1.0, 1.2])
    close = A.leg_frame(df, "back", "close")
    assert list(close["odds"]) == [3.3, 2.0] and close["pl"].tolist() == pytest.approx([-1.0, 1.0])
    assert close["clv"].tolist() == pytest.approx([0.0, 0.0])                    # taking the close = zero CLV

    with pytest.raises(ValueError):
        A.leg_frame(df, "lay", "ref")


def test_leg_frame_uses_the_requested_latency_and_commission_columns():
    df = mkframe([{"p_l0": 2.0, "p_l60": 1.8, "won": 1.0}], comms=(0.0, C, 0.05))
    assert A.leg_frame(df, "back", "ltp", latency=60, commission=0.05)["pl"].iloc[0] == pytest.approx(0.8 * 0.95)
    assert A.leg_frame(df, "back", "ltp", latency=0, commission=0.0)["pl"].iloc[0] == pytest.approx(1.0)
    assert A.leg_missing(df, "back", "ltp", latency=300) == ["p_l300", "back_pl_l300_c4p8"]
    with pytest.raises(KeyError):
        A.leg_frame(df, "back", "ltp", latency=300)


def test_leg_frame_exclusions_are_counted_by_reason_and_not_in_the_leg():
    nan = NAN
    df = mkframe([
        {},                                                                                   # 0 good
        {"status": "no_event_match", "won": nan, "void": nan, "in_play_at_alert": nan, "p_l0": nan,
         "bb_l0": nan, "bl_l0": nan},                                                         # 1 not matched
        {"won": nan, "void": nan, "result_known": False},                                     # 2 result unknown
        {"won": 0.0, "void": 1.0},                                                            # 3 void
        {"in_play_at_alert": True},                                                           # 4 in play at the alert
        {"p_l0": nan},                                                                        # 5 no price
        {"back_pl_l0_c4p8": nan},                                                             # 6 price but no P/L
        {"p_l0": 1.0},                                                                        # 7 price <= 1 -> no price
        {"status": "no_event_match", "in_play_at_alert": True, "p_l0": nan},                  # 8 counted ONCE (first reason)
    ])
    leg = A.leg_frame(df, "back", "ltp")
    assert leg.attrs["excluded"] == {"not_matched": 2, "no_result": 1, "void": 1, "in_play": 1, "no_price": 2,
                                     "no_tradeable_book": 0, "no_pl": 1}
    assert leg.attrs["n_input"] == 9 and list(leg["alert_id"]) == [0]
    assert sum(leg.attrs["excluded"].values()) + len(leg) == 9                                 # nothing vanished silently
    leg2 = A.leg_frame(df, "back", "ltp", preoff_only=False)                                   # in-play row is kept
    assert leg2.attrs["excluded"]["in_play"] == 0 and sorted(leg2["alert_id"]) == [0, 4]
    assert leg2.attrs["excluded"]["not_matched"] == 2


def test_leg_frame_carries_features_but_not_heavy_columns():
    df = mkframe([{}])
    df["sport"], df["fx_back_lay_h5_s0"], df["es_5"] = "Soccer", 0.1, 3
    leg = A.leg_frame(df, "back", "ltp")
    assert list(leg.columns[:9]) == A.LEG_COLS
    assert "sport" in leg and "ref_odds" in leg and "bb_l0" in leg
    assert not [c for c in leg.columns if c.startswith(("fx_", "es_", "back_pl_", "xlay_pl_"))] and "status" not in leg


def test_book_legs_need_a_tradeable_book_and_frames_without_the_column_are_unchanged():
    df = mkframe([{}, {"bb_l0": 1.12, "bl_l0": 2.96}, {"bl_l0": NAN}, {}, {}])
    df["book_ok_l0"] = [True, False, False, None, True]       # 1 wide, 2 one-sided, 3 not True, 4 fine at L = 0
    df["book_ok_l60"] = [True, True, False, True, False]
    back = A.leg_frame(df, "back", "book")
    assert list(back["alert_id"]) == [0, 4] and back.attrs["excluded"]["no_tradeable_book"] == 3
    assert sum(back.attrs["excluded"].values()) + len(back) == 5                       # nothing vanished silently
    lay = A.leg_frame(df, "lay", "book")                                               # row 2 has no lay price at all
    assert list(lay["alert_id"]) == [0, 4] and lay.attrs["excluded"]["no_price"] == 1
    assert lay.attrs["excluded"]["no_tradeable_book"] == 2                             # ... so only rows 1 and 3 are "not tradeable"
    b60 = A.leg_frame(df, "back", "book", latency=60)                                  # each latency reads its own flag
    assert list(b60["alert_id"]) == [0, 1, 3] and b60.attrs["excluded"]["no_tradeable_book"] == 2
    for price in ("ltp", "ref", "close"):                                              # other legs are not touched
        assert len(A.leg_frame(df, "back", price)) == 5
    old = df.drop(columns=["book_ok_l0", "book_ok_l60"])                               # an older frame: as before, no exclusion
    ob = A.leg_frame(old, "back", "book")
    assert list(ob["alert_id"]) == [0, 1, 2, 3, 4] and ob.attrs["excluded"]["no_tradeable_book"] == 0
    h = A.headline(df, commissions=(C,), latencies=(0, 60))
    r = h[(h.side == "back") & (h.price == "book") & (h.latency_s == 0)].iloc[0]
    assert r["n"] == 2 and json.loads(r["excluded"])["no_tradeable_book"] == 3


def test_book_taker_scalps_use_only_tradeable_books():
    df = mkframe([{}, {}, {}, {}])
    df["xfx_back_lay_h5"] = [0.1, 0.2, 0.3, 0.4]
    df["fx_back_lay_h5_s0"] = [0.1, 0.2, 0.3, 0.4]
    df["book_ok_l0"] = [True, False, True, True]
    fixed, _ = A.scalping_tables(df)
    bk, lt = fixed[fixed.basis == "book"].iloc[0], fixed[fixed.basis == "ltp"].iloc[0]
    assert bk["n"] == 3 and bk["mean_green"] == pytest.approx((0.1 + 0.3 + 0.4) / 3) and lt["n"] == 4
    again, _ = A.scalping_tables(df.drop(columns="book_ok_l0"))
    assert again[again.basis == "book"].iloc[0]["n"] == 4


def test_min_odds_best_back_and_the_gap_on_tradeable_books_hand_worked():
    df = mkframe([{"ref_odds": 2.0, "bb_l0": 1.9, "bl_l0": 2.0}, {"ref_odds": 2.0, "bb_l0": 2.0, "bl_l0": 2.1},
                  {"ref_odds": 2.0, "bb_l0": 2.2, "bl_l0": 2.3}, {"ref_odds": 2.0, "bb_l0": NAN, "bl_l0": 2.4},
                  {"status": "no_event_match", "ref_odds": 2.0}])
    df["book_ok_l0"] = [True, True, False, True, True]
    bb = A.min_odds_best_back(df)
    assert bb.columns.tolist() == ["latency_s", "n", "k"] and bb["latency_s"].tolist() == [0, 60, 120, 300]
    assert bb["n"].tolist() == [3, 3, 0, 0] and bb["k"].tolist() == [2, 2, 0, 0]       # 2.0 >= 2.0 counts; row 3 has no back
    g = A.min_odds_gap(df)                                                             # rows 0 and 1 only (2 wide, 3 one-sided)
    assert g["n"] == 2 and g["lay_ge_min"] == 2
    assert g["best_back"] == pytest.approx({"median": -0.025, "min": -0.05, "max": 0.0})
    assert g["mid"] == pytest.approx({"median": 0.0, "min": -0.025, "max": 0.025})
    e = A.min_odds_gap(df.assign(book_ok_l0=False))
    assert e["n"] == 0 and np.isnan(e["best_back"]["median"]) and e["lay_ge_min"] == 0
    assert A.min_odds_best_back(df.iloc[0:0])["n"].tolist() == [0, 0, 0, 0]
    z = A.resting_order(mkframe([{}]))                                                 # no limit columns at all
    assert z["has_columns"] is False and z["placed"] == 1 and z["filled"] == 0 and np.isnan(z["roi"])


# ----------------------------------------------------------------------------- e. summarize_leg
def _lay_leg():
    odds = [2, 2, 3, 4, 5, 1.5]
    runner_won = [0, 0, 1, 0, 0, 1]
    pl = [lpl(o, w) for o, w in zip(odds, runner_won)]
    return pd.DataFrame({
        "alert_id": range(6), "ts": [T + pd.Timedelta(minutes=i) for i in range(6)],
        "event": ["e1", "e1", "e2", "e2", "e3", "e3"], "pl": pl, "odds": [float(o) for o in odds],
        "won": [1.0 - w for w in runner_won], "liability": [o - 1.0 for o in odds],
        "p_close": [NAN] * 6, "clv": [0.1, 0.2, NAN, 0.0, -0.1, 0.3]})


def test_summarize_leg_lay_matches_hand_computation():
    leg = _lay_leg()
    # pl = [.952, .952, -2, .952, .952, -.5]: sum 1.308 ; liability 1+1+2+3+4+.5 = 11.5
    s = A.summarize_leg(leg, "lay", C)
    assert s["n"] == 6 and s["n_events"] == 3
    assert s["roi"] == pytest.approx(1.308 / 6) and s["strike_rate"] == pytest.approx(4 / 6)
    assert s["rol"] == pytest.approx(1.308 / 11.5) and s["avg_odds"] == pytest.approx(17.5 / 6)
    assert s["max_drawdown"] == pytest.approx(2.0) and s["longest_losing_streak"] == 1
    assert s["clv_n"] == 5 and s["clv_mean"] == pytest.approx(0.1) and s["clv_pct_pos"] == pytest.approx(0.6)
    assert s["roi_ci_lo"] <= s["roi"] <= s["roi_ci_hi"] and s["roi_cl_lo"] <= s["roi"] <= s["roi_cl_hi"]
    assert s["clv_ci_lo"] <= 0.1 <= s["clv_ci_hi"] and 0 < s["p_null"] <= 1
    b = A.summarize_leg(leg, "back", C)                                  # backs have no ROL
    assert np.isnan(b["rol"]) and list(s) == A.SUMMARY_FIELDS


def test_summarize_leg_empty_does_not_raise():
    s = A.summarize_leg(_lay_leg().iloc[0:0], "lay", C)
    assert s["n"] == 0 and s["n_events"] == 0 and s["clv_n"] == 0 and list(s) == A.SUMMARY_FIELDS
    assert all(np.isnan(v) for k, v in s.items() if k not in ("n", "n_events", "clv_n"))
    only_nan_clv = _lay_leg().assign(clv=NAN)
    s2 = A.summarize_leg(only_nan_clv, "back", C)
    assert s2["clv_n"] == 0 and np.isnan(s2["clv_mean"]) and np.isnan(s2["clv_ci_lo"])


# ----------------------------------------------------------------------------- c. null_p
def _const_leg(odds, lay_wins, side="lay"):
    n = len(lay_wins)
    pl = [(1 - C) if w else -(odds - 1) for w in lay_wins]                # lay: wins when the runner loses
    return pd.DataFrame({"pl": pl, "odds": [odds] * n, "won": [float(w) for w in lay_wins]})


def test_null_p_lay_extremes():
    assert A.null_p(_const_leg(2.0, [1] * 60), "lay", C) < 0.05                    # 60/60 lay wins at evens
    assert A.null_p(_const_leg(2.0, [0] * 60), "lay", C) > 0.95                    # 60/60 lay losses
    p = A.null_p(_const_leg(2.0, [1] * 50 + [0] * 50), "lay", C)                   # exactly break-even strike rate
    assert 0.3 < p < 0.7
    assert np.isnan(A.null_p(_const_leg(2.0, []), "lay", C))
    # a backer's break-even run is the same kind of event
    back = pd.DataFrame({"pl": [(1 - C) if w else -1.0 for w in [1] * 50 + [0] * 50], "odds": 2.0, "won": 0.5})
    assert 0.3 < A.null_p(back, "back", C) < 0.7


def test_null_p_back_is_roi_null_sim_pvalue_and_chunking_is_exact(monkeypatch):
    leg = pd.DataFrame({"pl": [1.0, -1, -1, 1, -1, -1, -1, 1, -1, -1], "odds": [3.0] * 10, "won": 0.3})
    direct = roi.null_sim_pvalue(leg["odds"].to_numpy(), float(leg["pl"].mean()), 0.0, 2000, 54321)
    assert A.null_p(leg, "back", 0.0, n_sim=2000) == direct
    monkeypatch.setattr(A, "_CAP", 5000)                                           # force 4 chunks of 500 x 10
    chunked = A.null_p(leg, "back", 0.0, n_sim=2000)
    assert 0 < chunked <= 1 and abs(chunked - direct) < 0.06
    lay = _const_leg(3.0, [1, 0, 0, 1, 0, 0, 0, 1, 0, 0])
    monkeypatch.undo()
    full = A.null_p(lay, "lay", C, n_sim=2000)
    monkeypatch.setattr(A, "_CAP", 5000)
    assert A.null_p(lay, "lay", C, n_sim=2000) == full                              # same RNG stream, chunked or not


# ----------------------------------------------------------------------------- d. cluster_ci
def test_cluster_ci_is_wider_than_iid_when_bets_share_two_events():
    pl = np.r_[np.ones(50), -np.ones(50)]
    clusters = np.r_[["A"] * 50, ["B"] * 50]
    lo, hi = A.cluster_ci(pl, clusters)
    s = roi.summarize(pl, np.full(100, 2.0), np.r_[np.ones(50), np.zeros(50)], run_null=False, n_boot=4000)
    assert (hi - lo) > 4 * (s.roi_ci_hi - s.roi_ci_lo) and lo == pytest.approx(-1.0) and hi == pytest.approx(1.0)
    lo2, hi2 = A.cluster_ci(pl, np.arange(100))                                    # every bet its own cluster ~ iid
    assert abs((hi2 - lo2) - (s.roi_ci_hi - s.roi_ci_lo)) < 0.08


def test_cluster_ci_single_cluster_is_nan_and_missing_ids_are_singletons():
    lo, hi = A.cluster_ci([1.0, -1.0, 1.0], ["X", "X", "X"])
    assert np.isnan(lo) and np.isnan(hi)
    assert all(np.isnan(v) for v in A.cluster_ci([], []))
    lo, hi = A.cluster_ci([1.0, -1.0, 1.0, 1.0], [None, None, "X", "X"])           # two None bets are two clusters
    assert np.isfinite(lo) and np.isfinite(hi)


# ----------------------------------------------------------------------------- f. headline
def test_headline_row_count_values_and_missing_columns():
    lat, comms = (0, 60), (0.0, C)
    df = mkframe([{"won": 1.0, "p_l0": 2.5, "bb_l0": 2.4}, {"won": 0.0, "p_l0": 2.5, "bb_l0": 2.4},
                  {"won": 1.0, "p_l0": 2.0}, {"won": 0.0, "p_l0": 2.0}], lat=lat, comms=comms)
    h = A.headline(df, commissions=comms, latencies=lat)
    assert len(h) == 2 * 2 * len(lat) * len(comms) + 2 == 18
    assert {"side", "price", "latency_s", "commission", "excluded", "note"} <= set(h.columns)
    r = h[(h.side == "back") & (h.price == "ltp") & (h.latency_s == 0) & (h.commission == C)].iloc[0]
    # bets: +1.5*0.952, -1, +1*0.952, -1  -> sum 0.38  -> roi 0.095
    assert r["n"] == 4 and r["roi"] == pytest.approx((1.428 - 1 + 0.952 - 1) / 4) and r["note"] == ""
    assert json.loads(r["excluded"]) == dict.fromkeys(A.REASONS, 0)
    lay = h[(h.side == "lay") & (h.price == "book") & (h.latency_s == 60) & (h.commission == 0.0)].iloc[0]
    assert lay["n"] == 4 and not np.isnan(lay["rol"])
    assert set(h[h.price.isin(["ref", "close"])].side) == {"back"} and (h.price == "ref").sum() == (h.price == "close").sum() == 1

    # missing columns -> rows with n = 0 and a note, no exception
    thin = df.drop(columns=["xlay_pl_l60_c4p8", "ref_back_pl", "bl_l0"])
    h2 = A.headline(thin, commissions=comms, latencies=lat)
    assert len(h2) == 18
    miss = h2[h2["note"] == "columns missing"]
    assert len(miss) > 0 and (miss["n"] == 0).all() and (miss["price"] == "ref").any()
    assert (miss[miss.price == "book"]["side"] == "lay").all()                       # every back/book row still computed
    assert h2[(h2.side == "back") & (h2.price == "ltp")]["n"].eq(4).all()


# ----------------------------------------------------------------------------- g. event study
def test_event_study_table_only_ok_preoff_and_values():
    df = mkframe([{}, {}, {}, {}, {"status": "no_event_match"}, {"in_play_at_alert": True}])
    df["es_5"] = [-2, 0, 4, 3, 99, 99]
    df["es_-5"] = [1, 1, 1, None, 99, 99]
    df["es_x"] = 1
    t = A.event_study_table(df)
    assert list(t["offset_min"]) == [-5, 5] and list(t["n"]) == [3, 4]               # the None is not counted
    r = t.set_index("offset_min").loc[5]
    assert r["mean_ticks"] == pytest.approx(5 / 4) and r["median_ticks"] == pytest.approx(1.5)
    assert r["pct_shorter"] == pytest.approx(1 / 4) and r["pct_longer"] == pytest.approx(2 / 4)   # -2 shorter; 4 and 3 longer
    assert r["ci_lo"] <= r["mean_ticks"] <= r["ci_hi"]
    assert len(A.event_study_table(mkframe([{}]))) == 0


# ----------------------------------------------------------------------------- a. features
def test_add_features_bands_gaps_and_skips():
    df = mkframe([{} for _ in range(8)])
    df["bb_l0"] = [1.4, 1.5, 1.8, 2.2, 3.0, 5.0, 10.0, NAN]
    df["p_l0"] = [2.0] * 7 + [2.7]
    df["ref_odds"] = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.7]
    df["mins_to_off"] = [10, 15, 100, 500, 1000, 14.9, 60, 180]
    df["spread_ticks_l0"] = [1, 2, 4, 8, 1, 2, 3, 5]
    df["bb_l0_size"] = [5, 10, 100, 300, 9.9, 49.9, 50, 250]
    df["drop_pct"] = [1, 2, 3, 4, 5, 6, 7, 8]
    f = A.add_features(df)
    assert list(f["odds_band"]) == ["<=1.5", "<=1.5", "1.5-2", "2-2.5", "2.5-3.5", "3.5-6", ">6", "2.5-3.5"]  # last: p_l0 fallback
    assert list(f["mins_to_off_band"]) == ["<15", "15-60", "60-180", "180-720", "720+", "<15", "60-180", "180-720"]
    assert list(f["spread_band"]) == ["<=1", "2", "3-5", ">5", "<=1", "2", "3-5", "3-5"]
    assert list(f["liq_band"]) == ["<10", "10-50", "50-250", "250+", "<10", "10-50", "50-250", "250+"]
    assert list(f["drop_band"]) == ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", "Q4", "Q4"]
    assert f["gap_book"].iloc[:2].tolist() == pytest.approx([-0.3, -0.25]) and np.isnan(f["gap_book"].iloc[7])
    assert f["gap_ltp"].tolist() == pytest.approx([0.0] * 7 + [0.0])
    assert f["hour_utc"].iloc[0] == 10 and f["dow"].iloc[0] == "Thu"               # 2026-10-01 is a Thursday
    assert f.attrs["features_skipped"] == [] and "odds_band" not in df             # original untouched
    thin = A.add_features(df[["alert_id", "ts", "p_l0"]])
    assert thin.attrs["features_skipped"] == ["mins_to_off_band", "gap_book", "gap_ltp", "spread_band", "liq_band", "drop_band"]
    assert "odds_band" in thin and "hour_utc" in thin


# ----------------------------------------------------------------------------- h. scan_segments
def _seg_leg(seed=11):
    rng = np.random.default_rng(seed)
    n = 100
    mk = np.array(["A"] * 50 + ["B"] * 40 + ["C"] * 10)
    wins = np.r_[[1] * 45 + [0] * 5, rng.permutation([1] * 20 + [0] * 20), rng.permutation([1] * 5 + [0] * 5)]
    pl = np.where(wins == 1, 1 * (1 - C), -1.0)
    return pd.DataFrame({
        "alert_id": range(n), "ts": [T + pd.Timedelta(minutes=i) for i in range(n)], "event": [f"E{i // 2}" for i in range(n)],
        "pl": pl, "odds": 2.0, "won": wins.astype(float), "liability": 1.0, "p_close": 2.0, "clv": rng.normal(0, 0.02, n),
        "mk": mk, "x": rng.permutation(np.linspace(0, 1, n)), "flat": [1, 2, 3, 1] * 25, "nan_cat": NAN})


def test_scan_segments_counts_bh_across_the_whole_table_and_finds_the_planted_segment():
    leg = _seg_leg()
    out = A.scan_segments(leg, "back", C, ["mk", "nan_cat", "ghost"], ["x", "flat", "ghost2"], min_n=20)
    # tested: mk==A (50), mk==B (40) [C has 10 < 20] + 4 quantiles x 2 sides on x (each >= 20 rows) = 10
    assert out.attrs["n_tests"] == len(out) == 10
    assert (out["kind"] == "cat").sum() == 2 and (out["kind"] == "num").sum() == 8
    assert set(out.attrs["skipped"]) == {"nan_cat", "ghost", "flat", "ghost2"}
    assert out.attrs["skipped_reasons"]["flat"] == "fewer than 5 distinct values"
    assert out["n"].min() >= 20 and {"rule", "col", "kind", "n", "bh_pass", *A.SUMMARY_FIELDS} <= set(out.columns)
    # BH is the roi.benjamini_hochberg of the p-values of ALL ten rows
    assert out["bh_pass"].tolist() == roi.benjamini_hochberg(out["p_null"].tolist(), q=0.10)
    r = out.set_index("rule")
    assert bool(r.loc["mk == A", "bh_pass"]) and r.loc["mk == A", "roi"] == pytest.approx((45 * 0.952 - 5) / 50)
    assert not bool(r.loc["mk == B", "bh_pass"])                                  # noise does not pass
    assert 0 < r.loc["mk == B", "p_null"] <= 1 and r.loc["mk == B", "n"] == 40


def test_scan_segments_empty_when_nothing_testable():
    out = A.scan_segments(_seg_leg().iloc[:15], "back", C, ["mk"], ["x"], min_n=20)
    assert len(out) == 0 and out.attrs["n_tests"] == 0 and "bh_pass" in out.columns


# ----------------------------------------------------------------------------- i. walk-forward
def test_walk_forward_leg_too_few_bets_and_structure():
    leg = _seg_leg()
    assert A.walk_forward_leg(leg.iloc[:59], ["mk"], ["x"], C) == {"skipped": "too few bets", "n": 59}
    wf = A.walk_forward_leg(leg, ["mk"], ["x"], C, n_folds=4)
    assert wf["n_rules"] == 2 + 8 and len(wf["picks"]) == 3 and "baseline_all_oos" in wf and "oos_n" in wf
    assert wf["baseline_all_oos"]["n"] == 75                                       # folds 1-3 of 4: rows 25..99
    lay = A.walk_forward_leg(leg, ["mk"], ["x"], C, n_folds=4, side="lay")
    assert lay["baseline_all_oos"]["p_null"] is None and "p_null_note" in lay


def test_walk_forward_leg_equals_segments_walk_forward_and_restores_summarize():
    from sbroi import segments as seg
    leg = _seg_leg()
    wf = A.walk_forward_leg(leg, ["mk"], ["x"], C)
    rules = seg.rules_from_levels(leg, ["mk"]) + seg.rules_from_thresholds(leg, ["x"])
    direct = seg.walk_forward(leg, rules, n_folds=4, select_by="roi_ci_lo", commission=C)
    assert seg.summarize is roi.summarize                                          # the bounded stand-in is removed again
    assert {k: v for k, v in wf.items() if k != "n_rules"} == direct               # identical for a small leg
    assert wf["oos"]["p_null"] == direct["oos"]["p_null"] and direct["oos"]["p_null"] is not None


# ----------------------------------------------------------------------------- j. verdict
def test_verdict_positive_needs_all_four_criteria():
    def row(rule, bh=True, cl=0.05, clv=0.01, n=60, roi_=0.2):
        return {"rule": rule, "col": "c", "kind": "cat", "n": n, "roi": roi_, "roi_cl_lo": cl, "p_null": 0.001,
                "bh_pass": bh, "clv_mean": 0.02, "clv_ci_lo": clv}
    seg = pd.DataFrame([row("ok"), row("no_bh", bh=False), row("cl_neg", cl=-0.02), row("clv_neg", clv=-0.01),
                        row("not_picked"), row("oos_neg"), row("lowroi_nobh", bh=False, roi_=-0.1),
                        row("small_nobh", bh=False, n=20), row("bh_small", n=10, roi_=-0.3, cl=-0.5)])
    wf = {"picks": [{"fold": 1, "rule": "ok", "test_n": 10, "test_pl": 2.0}, {"fold": 2, "rule": "ok", "test_n": 10, "test_pl": -0.5},
                    {"fold": 3, "rule": "no_bh", "test_n": 5, "test_pl": 1.0}, {"fold": 1, "rule": "cl_neg", "test_n": 5, "test_pl": 1.0},
                    {"fold": 2, "rule": "clv_neg", "test_n": 5, "test_pl": 1.0}, {"fold": 3, "rule": "oos_neg", "test_n": 5, "test_pl": -3.0},
                    {"fold": 3, "rule": None}]}
    v = A.verdict(seg, wf, min_n=30).set_index("rule")
    assert list(v.columns) == A.VERDICT_COLS[1:]
    assert set(v.index) == {"ok", "no_bh", "cl_neg", "clv_neg", "not_picked", "oos_neg", "bh_small"}   # lowroi/small_nobh dropped
    assert v["positive"].to_dict() == {"ok": True, "no_bh": False, "cl_neg": False, "clv_neg": False,
                                       "not_picked": False, "oos_neg": False, "bh_small": False}
    assert v.loc["ok", "oos_roi"] == pytest.approx(1.5 / 20) and bool(v.loc["ok", "oos_selected"])
    assert not v.loc["not_picked", "oos_selected"] and np.isnan(v.loc["not_picked", "oos_roi"])
    assert v.loc["oos_neg", "oos_roi"] == pytest.approx(-0.6)
    assert A.verdict(seg, {"skipped": "too few bets", "n": 10}, min_n=30)["positive"].sum() == 0   # no wf -> never positive
    e = A.verdict(seg.iloc[0:0], wf)
    assert len(e) == 0 and list(e.columns) == A.VERDICT_COLS


# ----------------------------------------------------------------------------- k. scalping
def test_scalping_tables_means_shares_and_to_off_label():
    df = mkframe([{}, {}, {}, {}, {"status": "no_event_match"}, {"in_play_at_alert": True}])
    df["fx_back_lay_h5_s0"] = [0.10, -0.05, 0.20, NAN, 99.0, 99.0]
    df["fx_back_lay_h5_s1"] = [0.0, 0.0, 0.0, 0.0, 99.0, 99.0]
    df["fx_lay_back_h100000_s0"] = [0.02, 0.04, -0.01, -0.01, 99.0, 99.0]
    df["xfx_back_lay_h5"] = [-0.02, -0.02, NAN, NAN, 99.0, 99.0]
    df["xfx_back_lay_h5_reason"] = ["time"] * 6                                  # companion column is not a metric
    df["tp_back_lay_tp2_slNone_s0"] = [0.02, 0.02, -0.03, 0.0, 99.0, 99.0]
    df["tp_back_lay_tp2_slNone_s0_reason"] = ["tp", "tp", "sl", "off", "tp", "tp"]
    fixed, tpsl = A.scalping_tables(df)
    assert len(fixed) == 4 and list(fixed.columns) == A.FIXED_COLS
    r = fixed[(fixed.basis == "ltp") & (fixed.side == "back_lay") & (fixed.pen == 0)].iloc[0]
    assert (r["n"], r["n_missing"], r["horizon_min"], r["horizon"]) == (3, 1, 5, "5")
    assert r["mean_green"] == pytest.approx(0.25 / 3) and r["median"] == pytest.approx(0.10)
    assert r["pct_profitable"] == pytest.approx(2 / 3) and r["pct_loss"] == pytest.approx(1 / 3)
    assert r["ci_lo"] <= r["mean_green"] <= r["ci_hi"]
    z = fixed[(fixed.pen == 1)].iloc[0]
    assert z["pct_profitable"] == 0 and z["pct_loss"] == 0 and z["mean_green"] == 0           # zeros are neither
    off = fixed[fixed.horizon_min == 100000].iloc[0]
    assert off["horizon"] == "to_off" and off["side"] == "lay_back" and off["mean_green"] == pytest.approx(0.01)
    bk = fixed[fixed.basis == "book"].iloc[0]
    assert np.isnan(bk["pen"]) and bk["n"] == 2 and bk["n_missing"] == 2 and bk["mean_green"] == pytest.approx(-0.02)
    assert len(tpsl) == 1 and list(tpsl.columns) == A.TPSL_COLS
    t = tpsl.iloc[0]
    assert (t["side"], t["tp"], t["sl"], t["pen"], t["n"], t["n_missing"]) == ("back_lay", 2, "none", 0, 4, 0)
    assert t["mean_green"] == pytest.approx(0.01 / 4) and t["pct_profitable"] == pytest.approx(0.5)
    assert (t["reason_tp"], t["reason_sl"], t["reason_off"], t["reason_time"], t["reason_no_exit"], t["reason_other"]) == \
        (0.5, 0.25, 0.25, 0.0, 0.0, 0.0)
    e1, e2 = A.scalping_tables(mkframe([{}]))
    assert len(e1) == len(e2) == 0 and list(e1.columns) == A.FIXED_COLS


# ----------------------------------------------------------------------------- l. matching
def test_match_report_review_reasons_summary_and_seeded_spot():
    rows = []
    for i in range(30):
        rows.append({"alert_id": i, "ts": T, "sport": "Soccer", "home": f"H{i}", "away": f"A{i}", "kickoff": T,
                     "bf_market": "MATCH_ODDS", "sel_kind": "home", "text": f"msg {i}", "status": "ok", "match_reason": "ok",
                     "match_score": 100.0, "match_dt_min": 0.0, "match_swapped": False, "bf_event_name": f"H{i} v A{i}",
                     "bf_runner": f"H{i}", "bf_hc": 0.0, "bf_off": T})
    rows[1].update(status="no_event_match", match_reason="low_score(min=40,avg=50)", match_score=50.0)
    rows[2].update(match_reason="ambiguous")
    rows[3].update(match_score=80.0)
    rows[4].update(match_dt_min=-7.5)
    rows[5].update(match_swapped=True)
    rows[6].update(status="error:ValueError", match_score=np.nan, match_dt_min=np.nan, match_reason=None, error="boom")
    rows[7].update(match_score=85.0, match_dt_min=5.0)                          # boundary values are fine
    rows[8].update(sport="Tennis", bf_market="MATCH_ODDS")
    df = pd.DataFrame(rows)
    summary, review, spot = A.match_report(df)
    why = review.set_index("alert_id")["why"].to_dict()
    assert set(why) == {1, 2, 3, 4, 5, 6}
    assert why[1] == "status=no_event_match; score<85" and why[2] == "ambiguous" and why[3] == "score<85"
    assert why[4] == "|dt|>5min" and why[5] == "swapped" and why[6] == "status=error:ValueError"
    s = summary.set_index(["sport", "bf_market", "status"])
    assert s.loc[("Soccer", "MATCH_ODDS", "ok"), "n"] == 27 and s.loc[("Tennis", "MATCH_ODDS", "ok"), "n"] == 1
    soc = summary[(summary.sport == "Soccer") & (summary.status == "match_rate")].iloc[0]
    assert soc["n"] == 29 and soc["share"] == pytest.approx(27 / 29)               # 29 Soccer alerts, 27 ok
    assert summary[(summary.sport == "Soccer") & (summary.status == "no_event_match")]["n"].iloc[0] == 1
    assert len(spot) == 20 and set(spot["alert_id"]) <= set(df.loc[df.status == "ok", "alert_id"])
    _, _, spot2 = A.match_report(df)
    assert list(spot["alert_id"]) == list(spot2["alert_id"])                        # seeded
    assert list(A.match_report(df, seed=99)[2]["alert_id"]) != list(spot["alert_id"])
    assert {"text", "bf_event_name", "bf_runner", "match_score"} <= set(spot.columns)
    assert len(A.match_report(df.iloc[:5])[2]) == 4                                 # only the ok rows are sampled


# ----------------------------------------------------------------------------- m. curve
def test_pl_curve_is_cumulative_in_time_order():
    leg = pd.DataFrame({"ts": [T + pd.Timedelta(minutes=m) for m in (5, 1, 3)], "pl": [-1.0, 0.5, 2.0]})
    c = A.pl_curve(leg)
    assert list(c["pl"]) == [0.5, 2.0, -1.0] and list(c["cum_pl"]) == [0.5, 2.5, 1.5] and list(c.columns) == ["ts", "pl", "cum_pl"]
