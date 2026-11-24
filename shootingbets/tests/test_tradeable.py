"""Tradeable book (book_ok_l{L}, spread_pct_l0) and the resting back order at Min. Odds, on synthetic PRO markets (NOT real data)."""
import bz2

import numpy as np
import pandas as pd
import pytest

from conftest import MIN, T0, to_bytes
from sbroi import analysis as A
from sbroi.betfair_files import load_markets
from sbroi.matching import build_event_index
from sbroi.pipeline import Config, _book_ok, enrich
from test_pro_book import OFF, alerts, pro_lines

LATS = (0, 60, 120, 300)


def market(tmp_path, mid, extra=(), removed=False, **kw):
    """A pro_lines market plus extra rc lines ``(pt, [rc entries])`` merged in time order; ``removed``: runner 1001 is
    settled REMOVED (a void)."""
    lines = pro_lines(mid=mid, **kw)
    lines += [{"op": "mcm", "clk": str(pt), "pt": pt, "mc": [{"id": mid, "rc": rc}]} for pt, rc in extra]
    lines.sort(key=lambda x: x["pt"])                        # stable: same-ms lines keep their order
    if removed:
        lines[-1]["mc"][0]["marketDefinition"]["runners"][0]["status"] = "REMOVED"
    (tmp_path / f"{mid}.bz2").write_bytes(bz2.compress(to_bytes(lines)))


def load(tmp_path):
    mk = load_markets(str(tmp_path))
    return mk, build_event_index(mk)


def test_book_ok_is_a_two_sided_book_with_the_lay_at_most_20_percent_above_the_back():
    assert Config().max_spread_frac == 0.20 and Config().kickoff_window_by_sport == {"Tennis": 720.0}
    assert _book_ok(2.0, 2.4, 0.2) and _book_ok(2.0, 2.0, 0.2) and _book_ok(2.0, 1.9, 0.2)        # crossed is not wide
    assert _book_ok(1.5, 1.8, 0.2) is True      # 1.8 / 1.5 - 1 is 0.2000000000000002 in floats: exactly 20% must pass
    assert _book_ok(1.5, 1.81, 0.2) is False and _book_ok(1.12, 2.96, 0.2) is False
    assert _book_ok(np.nan, 2.0, 0.2) is False and _book_ok(2.0, np.nan, 0.2) is False and _book_ok(np.nan, np.nan, 0.2) is False
    assert _book_ok(2.0, 2.2, 0.1) is True and _book_ok(2.0, 2.2, 0.0999) is False


def test_book_ok_at_each_latency_one_sided_wide_and_the_config_threshold(tmp_path):
    t = 3 * MIN // 6                                                          # 30 s
    # A: at T0 2.48 / 2.52; from T0+30s the 2.52 lay is gone and the best lay is 3.2 (29% wide); T0+5min a 2.4 lay arrives
    market(tmp_path, "1.900000001", extra=[(T0 + t, [{"id": 1001, "atl": [[2.52, 0], [3.2, 10.0]]}])])
    # B: the only lay is withdrawn before the alert: one-sided at the alert, two-sided again once the 2.4 lay arrives
    market(tmp_path, "1.900000002", extra=[(T0 - 10 * MIN, [{"id": 1001, "atl": [[2.52, 0]]}])], event="34000002",
           name="Everton v Fulham", teams=("Everton", "Fulham"))
    mk, idx = load(tmp_path)
    al = alerts(("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham"))
    cols = [f"book_ok_l{L}" for L in LATS]
    df = enrich(al, mk, idx, Config(commissions=(0.0,)))
    assert list(df["status"]) == ["ok", "ok"]
    assert [bool(v) for v in df.loc[0, cols]] == [True, False, False, True]
    assert [bool(v) for v in df.loc[1, cols]] == [False, False, False, True]
    assert (df.loc[0, ["bb_l0", "bl_l0", "bl_l60", "bl_l300"]].tolist() == [2.48, 2.52, 3.2, 2.4]
            and np.isnan(df.loc[1, "bl_l0"]) and df.loc[1, "bb_l0"] == 2.48)
    assert df.loc[0, "spread_pct_l0"] == pytest.approx(2.52 / 2.48 - 1) and np.isnan(df.loc[1, "spread_pct_l0"])
    wide = enrich(al, mk, idx, Config(commissions=(0.0,), max_spread_frac=0.5))
    assert [bool(v) for v in wide.loc[0, cols]] == [True, True, True, True]          # 29% is within 50%
    assert [bool(v) for v in wide.loc[1, cols]] == [False, False, False, True]       # one-sided stays one-sided
    exact = 2.52 / 2.48 - 1                                                              # the spread at L = 0, to the bit
    assert enrich(al, mk, idx, Config(max_spread_frac=exact)).loc[0, "book_ok_l0"]
    assert not enrich(al, mk, idx, Config(max_spread_frac=exact - 1e-6)).loc[0, "book_ok_l0"]
    assert "book_ok_l0" not in enrich(al, mk, idx, Config(book=False))                  # no book columns without the book


def test_resting_back_order_at_min_odds_trade_window_equality_and_pl(tmp_path):
    ltp = lambda pt, *prices: (pt, [{"id": 1001 + i, "ltp": p} for i, p in enumerate(prices) if p])  # noqa: E731
    market(tmp_path, "1.900000001", extra=[
        ltp(T0, 9.0, 9.0),                  # AT the alert: not strictly after, never counts
        ltp(T0 + 1, 2.6, 4.0),              # 1 ms after: counts
        ltp(OFF - 1, 3.1),                  # 1 ms before the off: counts
        ltp(OFF, 50.0),                     # AT the off: not strictly before
        ltp(OFF + 5 * MIN, 99.0)])          # in play
    market(tmp_path, "1.900000002", extra=[ltp(T0 + 1, 3.0)], removed=True, event="34000002", name="Everton v Fulham",
           teams=("Everton", "Fulham"))
    mk, idx = load(tmp_path)
    refs = [3.1, 3.1000001, 2.2, 9.0, 50.0, 99.0, 3.5, 4.0, 4.0000001, 2.2, 2.0]
    al = alerts(*[("Arsenal FC", "Chelsea FC")] * 10, ("Everton", "Fulham"))
    al["sel_kind"] = ["home"] * 6 + ["away"] * 3 + ["home", "home"]
    al["ref_odds"] = refs
    al.loc[9, "ts"] = pd.to_datetime(OFF + 10 * MIN, unit="ms", utc=True)                # row 9 is placed in play
    df = enrich(al, mk, idx, Config(commissions=(0.0, 0.048)))
    assert list(df["status"]) == ["ok"] * 11 and list(df["in_play_at_alert"]) == [False] * 9 + [True, False]
    assert df["n_trades_after"].iloc[:10].tolist()[:9] == [2] * 6 + [1] * 3 and df.loc[10, "n_trades_after"] == 1
    assert df["max_ltp_after"].iloc[:9].tolist() == [3.1] * 6 + [4.0] * 3 and df.loc[10, "max_ltp_after"] == 3.0
    assert pd.isna(df.loc[9, ["n_trades_after", "max_ltp_after", "limit_fill", "limit_pl_c4p8"]]).all()   # pre-off only
    fill = [None if pd.isna(v) else bool(v) for v in df["limit_fill"]]
    assert fill == [True, False, True, False, False, False, True, True, False, None, True]       # ">=": 3.1 fills 3.1, 4.0 fills 4.0
    pl = df["limit_pl_c4p8"].tolist()
    assert pl[0] == pytest.approx(2.1 * 0.952) and pl[2] == pytest.approx(1.2 * 0.952) and pl[6] == pl[7] == -1.0
    assert df["limit_pl_c0"].iloc[[0, 2]].tolist() == pytest.approx([2.1, 1.2])
    assert all(np.isnan(pl[i]) for i in (1, 3, 4, 5, 8, 9))                              # not filled, or in play
    assert df.loc[10, "void"] == 1.0 and np.isnan(pl[10])                                # filled, but void: no P/L
    r = A.resting_order(df, 0.048)                                                       # hand-worked: rows 0, 2, 6, 7 filled
    assert (r["placed"], r["traded"], r["filled"], r["wins"], r["has_columns"]) == (9, 9, 4, 2, True)
    assert r["strike_rate"] == 0.5 and r["avg_odds"] == pytest.approx((3.1 + 2.2 + 3.5 + 4.0) / 4)
    assert r["roi"] == pytest.approx((2.1 * 0.952 + 1.2 * 0.952 - 2) / 4) and r["roi_ci_lo"] <= r["roi"] <= r["roi_ci_hi"]
    chk = A.min_odds_check(df, 0.048)
    assert len(chk) == 11 and list(chk["limit_fill"].astype(object).where(chk["limit_fill"].notna(), None)) == fill
    assert chk.columns.tolist() == ["alert_id", "ts", "sport", "fixture", "selection_raw", "min_odds", "bb_l0", "bl_l0",
                                    "spread_pct_l0", "book_ok_l0", "bb/min-1", "max_ltp_after", "n_trades_after",
                                    "limit_fill", "won", "void", "limit_pl_c4p8"]
    assert chk["fixture"].iloc[0] == "Arsenal FC v Chelsea FC" and chk["bb/min-1"].iloc[2] == pytest.approx(2.48 / 2.2 - 1)
    assert chk["min_odds"].tolist() == refs
