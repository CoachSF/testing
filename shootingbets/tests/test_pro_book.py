"""Executable order-book pricing (book_at, book_fixed_horizon, enrich x* columns) and the lazy PRO store."""
import bz2
import json
import math
import shutil
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from conftest import MIN, T0, md, to_bytes
from sbroi.betfair_files import RunnerSeries, load_markets, parse_stream_bytes
from sbroi.matching import build_event_index
from sbroi.pipeline import Config, _ctag, enrich
from sbroi.pricing import book_at
from sbroi.pro_index import COLUMNS, EVENT_TYPE_SPORT, MarketStore, build_event_index_from_manifest, load_manifest
from sbroi.scalping import book_fixed_horizon
from sbroi.ticks import tick_diff

NAN = float("nan")
OFF = T0 + 120 * MIN


def pro_lines(mid="1.700000001", market_type="MATCH_ODDS", event="34000001", name="Arsenal v Chelsea",
              teams=("Arsenal", "Chelsea"), open_ms=OFF, betting_type="ODDS"):
    """PRO-style market: runner 1001 (winner) book moves; best back 2.48 / best lay 2.52 at the alert (T0)."""
    def d(**kw):
        x = md(market_type=market_type, event_name=name, open_ms=open_ms, **kw)
        x["eventId"], x["bettingType"] = event, betting_type
        x["runners"][0]["name"], x["runners"][1]["name"] = teams
        return x

    def line(pt, **mc):
        return {"op": "mcm", "clk": str(pt), "pt": pt, "mc": [{"id": mid, **mc}]}

    return [
        line(T0 - 60 * MIN, marketDefinition=d()),
        line(T0 - 30 * MIN, rc=[
            {"id": 1001, "atb": [[2.48, 100.0]], "atl": [[2.52, 80.0]], "ltp": 2.5, "tv": 1000.0},
            {"id": 1002, "atb": [[3.0, 50.0]], "atl": [[3.1, 40.0]], "ltp": 3.05},
            {"id": 58805, "atb": [[3.4, 30.0]], "atl": [[3.5, 20.0]], "ltp": 3.4}]),
        line(T0 + 5 * MIN, rc=[{"id": 1001, "atl": [[2.4, 55.0]]}]),                    # best lay 2.52 -> 2.40
        line(T0 + 8 * MIN, rc=[{"id": 1001, "atb": [[2.48, 0], [2.44, 90.0]]}]),        # best back 2.48 -> 2.44
        line(T0 + 110 * MIN, rc=[{"id": 1001, "atb": [[2.44, 0], [2.3, 40.0]], "atl": [[2.4, 0], [2.34, 25.0]]}]),
        line(OFF, marketDefinition=d(in_play=True)),
        line(OFF + 5 * MIN, rc=[{"id": 1001, "atl": [[1.9, 10.0]]}]),                    # in-play: must not leak
        line(T0 + 240 * MIN, marketDefinition=d(status="CLOSED", in_play=True, settled=True,
                                                runner_status=("WINNER", "LOSER", "LOSER"))),
    ]


def alerts(*pairs):
    return pd.DataFrame([
        {"alert_id": i, "ts": pd.to_datetime(T0, unit="ms", utc=True), "sport": "Soccer", "home": h, "away": a,
         "kickoff": pd.to_datetime(OFF, unit="ms", utc=True), "bf_market": "MATCH_ODDS", "sel_kind": "home",
         "sel_team": None, "sel_hc": None, "ref_odds": 2.2} for i, (h, a) in enumerate(pairs, 1)])


def series(bb=(), bl=()):
    r = RunnerSeries(selection_id=1)
    for side, obs in (("bb", bb), ("bl", bl)):
        for t, p, s in obs:
            getattr(r, f"{side}_t").append(t)
            getattr(r, f"{side}_p").append(p)
            getattr(r, f"{side}_s").append(s)
    return r


# ----------------------------------------------------------------------------- a. book_at
def test_book_at_levels_removal_empty_side_and_no_lookahead():
    lines = [{"op": "mcm", "pt": pt, "mc": [{"id": "1.1", "rc": [{"id": 1001, **rc}]}]} for pt, rc in [
        (1000, {"atb": [[2.0, 10.0], [1.9, 20.0]], "atl": [[2.2, 5.0]]}),
        (2000, {"atb": [[2.0, 0]]}),                  # best level removed -> next level
        (3000, {"atb": [[1.9, 0]], "atl": [[2.2, 0]]}),  # both sides emptied
        (4000, {"atb": [[1.8, 7.0]]})]]
    r = parse_stream_bytes(to_bytes(lines))[0].runners[(1001, 0.0)]
    nan = {"bb": NAN, "bb_size": NAN, "bb_age_ms": NAN, "bl": NAN, "bl_size": NAN, "bl_age_ms": NAN}

    def same(got, want):
        assert got.keys() == want.keys()
        for k in want:
            assert got[k] == pytest.approx(want[k], nan_ok=True), k

    same(book_at(r, 999), nan)                                                # nothing observed yet
    same(book_at(r, 1000), {"bb": 2.0, "bb_size": 10.0, "bb_age_ms": 0, "bl": 2.2, "bl_size": 5.0, "bl_age_ms": 0})
    same(book_at(r, 1500), {"bb": 2.0, "bb_size": 10.0, "bb_age_ms": 500, "bl": 2.2, "bl_size": 5.0, "bl_age_ms": 500})
    same(book_at(r, 1999), {"bb": 2.0, "bb_size": 10.0, "bb_age_ms": 999, "bl": 2.2, "bl_size": 5.0, "bl_age_ms": 999})
    same(book_at(r, 2000), {"bb": 1.9, "bb_size": 20.0, "bb_age_ms": 0, "bl": 2.2, "bl_size": 5.0, "bl_age_ms": 1000})
    same(book_at(r, 3999), nan)                                               # emptied sides stay NaN
    same(book_at(r, 4000), {**nan, "bb": 1.8, "bb_size": 7.0, "bb_age_ms": 0})


def test_img_messages_are_deltas_even_when_a_runner_is_split_across_rc_entries():
    """Real PRO files flag marketDefinition-change messages img=true but their rc are deltas (incl. size-0
    removals), and one runner can be split over two rc entries: clearing the ladder on img wipes live levels."""
    key, mid = (1001, -0.5), "1.5"
    lines = [
        {"op": "mcm", "pt": 1000, "mc": [{"id": mid, "rc": [
            {"id": 1001, "hc": -0.5, "atb": [[2.0, 10.0], [1.9, 5.0]], "atl": [[2.1, 7.0]]}]}]},
        {"op": "mcm", "pt": 2000, "mc": [{"id": mid, "img": True, "marketDefinition": md(), "rc": [
            {"id": 1001, "hc": -0.5, "atb": [[2.0, 0]]},        # best back level removed -> 1.9 must remain
            {"id": 1001, "hc": -0.5, "atl": [[2.12, 3.0]]}]}]},  # behind the best lay 2.1 -> best lay unchanged
    ]
    r = parse_stream_bytes(to_bytes(lines))[0].runners[key]
    b = book_at(r, 2000)
    assert (b["bb"], b["bb_size"], b["bl"], b["bl_size"]) == (1.9, 5.0, 2.1, 7.0)
    assert r.bb_p == [2.0, 1.9] and r.bl_p == [2.1]


# ----------------------------------------------------------------------------- b. book_fixed_horizon
def test_book_fixed_horizon_back_lay_and_lay_back():
    r = series(bb=[(0, 2.48, 100.0), (5 * MIN, 2.60, 10.0)], bl=[(0, 2.52, 80.0), (5 * MIN, 2.40, 55.0)])
    tr = book_fixed_horizon(r, MIN, 10 * MIN, None, "back_lay", 0.05)       # back 2.48, lay 2.40 at t1
    assert (tr.entry_price, tr.exit_price, tr.exit_reason, tr.exit_pt) == (2.48, 2.40, "time", 11 * MIN)
    assert tr.green_pl == pytest.approx((2.48 / 2.40 - 1) * 0.95)
    assert tr.ticks_moved == tick_diff(2.48, 2.40) == -4
    tr = book_fixed_horizon(r, MIN, 10 * MIN, None, "lay_back", 0.05)       # lay 2.52, back 2.60
    assert (tr.entry_price, tr.exit_price) == (2.52, 2.60)
    assert tr.green_pl == pytest.approx((1 - 2.52 / 2.60) * 0.95) and tr.ticks_moved == 4
    r2 = series(bb=[(0, 2.48, 100.0), (5 * MIN, 2.44, 10.0)], bl=[(0, 2.52, 80.0)])
    tr = book_fixed_horizon(r2, MIN, 10 * MIN, None, "lay_back", 0.05)      # losing green-up: no commission
    assert tr.green_pl == pytest.approx(1 - 2.52 / 2.44) and tr.green_pl < 0


def test_book_fixed_horizon_off_clamp_and_failures():
    r = series(bb=[(0, 2.48, 100.0)], bl=[(0, 2.52, 80.0), (5 * MIN, 2.40, 55.0)])
    tr = book_fixed_horizon(r, MIN, 100 * MIN, 8 * MIN, "back_lay")
    assert (tr.exit_reason, tr.exit_pt, tr.exit_price) == ("off", 8 * MIN - 1, 2.40)
    tr = book_fixed_horizon(r, MIN, 100 * MIN, 8 * MIN, "back_lay", stay_preoff=False)
    assert (tr.exit_reason, tr.exit_pt) == ("time", 101 * MIN)
    # entry side empty / never observed
    for rr, side in [(series(bb=[(0, NAN, 0.0)], bl=[(0, 2.52, 80.0)]), "back_lay"),
                     (series(bb=[(0, 2.48, 100.0)], bl=[(0, NAN, 0.0)]), "lay_back"),
                     (series(), "back_lay")]:
        tr = book_fixed_horizon(rr, MIN, 10 * MIN, None, side)
        assert (tr.exit_reason, tr.entry_price, tr.green_pl) == ("no_entry", None, None)
    # exit side empty at t1, and t1 <= t0
    r3 = series(bb=[(0, 2.48, 100.0)], bl=[(0, 2.52, 80.0), (5 * MIN, NAN, 0.0)])
    tr = book_fixed_horizon(r3, MIN, 10 * MIN, None, "back_lay")
    assert (tr.exit_reason, tr.entry_price, tr.exit_price, tr.green_pl) == ("no_exit", 2.48, None, None)
    assert book_fixed_horizon(r, MIN, 10 * MIN, MIN, "back_lay").exit_reason == "no_exit"


# ----------------------------------------------------------------------------- c. _ctag
@pytest.mark.parametrize("c,tag", [(0, "c0"), (0.0, "c0"), (0.02, "c2"), (0.05, "c5"), (0.1, "c10"),
                                   (0.048, "c4p8"), (0.065, "c6p5"), (0.0025, "c0p25")])
def test_ctag(c, tag):
    assert _ctag(c) == tag


# ----------------------------------------------------------------------------- d. enrich
def test_enrich_pro_book_columns(tmp_path):
    p = tmp_path / "1.700000001.bz2"
    p.write_bytes(bz2.compress(to_bytes(pro_lines())))
    mk = load_markets(p)
    cfg = Config(commissions=(0.0, 0.048, 0.05))
    a = enrich(alerts(("Arsenal FC", "Chelsea FC")), mk, build_event_index(mk), cfg).iloc[0]
    assert a["status"] == "ok" and a["bf_data_level"] == "PRO" and a["has_book"] and a["won"] == 1
    assert (a["bb_l0"], a["bb_l0_size"], a["bl_l0"], a["bl_l0_size"]) == (2.48, 100.0, 2.52, 80.0)
    assert (a["bb_l0_age_min"], a["bl_l0_age_min"], a["tv_l0"]) == (30.0, 30.0, 1000.0)
    assert a["spread_ticks_l0"] == tick_diff(2.48, 2.52) == 2 and a["spread_pct_l0"] == pytest.approx(2.52 / 2.48 - 1)
    assert [bool(a[f"book_ok_l{L}"]) for L in (0, 60, 120, 300)] == [True] * 4         # 2.48 / 2.52, and 2.48 / 2.40 at 300 s
    assert (a["bb_l120"], a["bl_l120"]) == (2.48, 2.52)
    assert (a["bb_l300"], a["bl_l300"], a["bl_l300_size"]) == (2.48, 2.40, 55.0)  # observation at exactly t+300s counts
    assert (a["bb_close"], a["bl_close"]) == (2.3, 2.34)                           # last pre-off book, not in-play 1.9
    assert a["xback_pl_l0_c0"] == pytest.approx(1.48)
    assert a["xback_pl_l0_c4p8"] == pytest.approx(1.48 * 0.952) and a["xback_pl_l0_c5"] == pytest.approx(1.48 * 0.95)
    assert a["xlay_pl_l0_c0"] == pytest.approx(-1.52) and a["xlay_pl_l0_c4p8"] == pytest.approx(-1.52)
    assert a["xback_pl_l300_c0"] == pytest.approx(1.48) and a["xlay_pl_l300_c0"] == pytest.approx(-1.40)
    assert a["xclose_back_pl_c0"] == pytest.approx(1.3) and a["xclose_lay_pl_c0"] == pytest.approx(-1.34)
    assert a["xfx_back_lay_h10"] == pytest.approx((2.48 / 2.40 - 1) * 0.95) and a["xfx_back_lay_h10_reason"] == "time"
    assert a["xfx_lay_back_h10"] == pytest.approx(1 - 2.52 / 2.44)                 # best back 2.44 after the removal
    assert a["xfx_back_lay_h100000"] == pytest.approx((2.48 / 2.34 - 1) * 0.95)
    assert a["xfx_back_lay_h100000_reason"] == "off"
    # existing LTP columns unchanged
    assert a["p_l0"] == 2.5 and a["back_pl_l0_c0"] == pytest.approx(1.5) and a["lay_pl_l0_c0"] == pytest.approx(-1.5)
    assert a["back_pl_l0_c4p8"] == pytest.approx(1.5 * 0.952) and a["p_close"] == 2.5
    # switching the book off removes every new column
    b = enrich(alerts(("Arsenal FC", "Chelsea FC")), mk, build_event_index(mk), Config(book=False))
    assert not [c for c in b.columns if c.startswith(("x", "bb_", "bl_", "has_book", "spread_", "tv_", "book_ok"))]
    assert b.iloc[0]["p_l0"] == 2.5


def test_enrich_basic_market_has_no_book(basic_bz2):
    mk = load_markets(basic_bz2)
    a = enrich(alerts(("Arsenal FC", "Chelsea FC")), mk, build_event_index(mk), Config()).iloc[0]
    assert a["status"] == "ok" and a["bf_data_level"] == "BASIC" and not a["has_book"]
    assert not a["book_ok_l0"] and np.isnan(a["spread_pct_l0"])                      # no book: not tradeable, no spread
    assert all(np.isnan(a[k]) for k in ("bb_l0", "bl_l0", "bb_l0_size", "spread_ticks_l0", "tv_l0", "bb_close",
                                        "xback_pl_l0_c0", "xlay_pl_l0_c0"))
    assert a["xfx_back_lay_h10_reason"] == "no_entry" and a["p_l0"] == 2.5


# ----------------------------------------------------------------------------- e. pro_index
def _iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


@pytest.fixture
def pro_dir(tmp_path):
    """Extractor layout: <day>/<eventId>/<marketId>.bz2 + manifest_<month>.jsonl (dup row, garbage line)."""
    specs = [("1.700000001", "MATCH_ODDS", "34000001", "Arsenal v Chelsea", ("Arsenal", "Chelsea"), OFF),
             ("1.700000002", "OVER_UNDER_25", "34000001", "Arsenal v Chelsea", ("Over", "Under"), OFF + 10 * MIN),
             ("1.700000003", "MATCH_ODDS", "34000002", "Everton v Fulham", ("Everton", "Fulham"), OFF),
             ("1.700000004", "OVER_UNDER_25", "34000002", "Everton v Fulham", ("Over", "Under"), OFF - 20 * MIN)]
    rows = []
    for mid, mt, ev, name, teams, t in specs:
        raw = bz2.compress(to_bytes(pro_lines(mid, mt, ev, name, teams, t)))
        rel = f"{_iso(t)[:10]}/{ev}/{mid}.bz2"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(raw)
        rows.append({"market_id": mid, "event_id": ev, "event_name": name, "market_type": mt,
                     "market_time": _iso(t), "country_code": "GB", "bytes": len(raw),
                     "member": f"PRO/2026/Sep/21/{ev}/{mid}.bz2", "saved": rel})
    dup = {**rows[0], "bytes": 5, "saved": "2026-09-21/34000001/missing.bz2"}
    j = lambda rs: "\n".join(json.dumps(r) for r in rs)  # noqa: E731
    (tmp_path / "manifest_2026-09.jsonl").write_text(j([dup, *rows[:2]]) + "\nnot json {\n\n", encoding="utf-8")
    (tmp_path / "manifest_2026-10.jsonl").write_text(j(rows[2:]) + "\n", encoding="utf-8")
    return tmp_path


def test_manifest_and_event_index(pro_dir, tmp_path_factory):
    df = load_manifest(pro_dir)
    assert len(df) == 4 and df["market_id"].is_unique and list(df.columns) == COLUMNS
    a = df.set_index("market_id").loc["1.700000001"]
    assert a["saved"] == "2026-09-21/34000001/1.700000001.bz2" and a["bytes"] > 5  # larger dup row won
    assert a["market_time_ms"] == OFF
    empty = load_manifest(tmp_path_factory.mktemp("empty"))
    assert len(empty) == 0 and list(empty.columns) == COLUMNS
    idx = build_event_index_from_manifest(df)
    assert set(idx) == {"34000001", "34000002"}
    e = idx["34000001"]
    assert (e.home, e.away, e.event_name, e.sport, e.country_code, e.participants) == \
        ("Arsenal", "Chelsea", "Arsenal v Chelsea", "Soccer", "GB", [])
    assert e.open_date_ms == OFF and e.markets == {"MATCH_ODDS": ["1.700000001"], "OVER_UNDER_25": ["1.700000002"]}
    assert idx["34000002"].open_date_ms == OFF  # the MATCH_ODDS time, although another row starts 20 min earlier
    no_mo = pd.DataFrame([{"market_id": "m1", "event_id": "9", "event_name": "A v B", "market_type": "OVER_UNDER_25",
                           "market_time_ms": 5000, "country_code": None},
                          {"market_id": "m2", "event_id": "9", "event_name": "A v B", "market_type": "OVER_UNDER_15",
                           "market_time_ms": 3000, "country_code": None}])
    assert build_event_index_from_manifest(no_mo)["9"].open_date_ms == 3000  # no MATCH_ODDS: earliest row


def _manifest(root, *rows, month="2026-10"):
    root.mkdir(parents=True, exist_ok=True)
    base = {"country_code": "GB", "bytes": 10, "member": "m", "market_time": _iso(OFF), "market_type": "MATCH_ODDS"}
    (root / f"manifest_{month}.jsonl").write_text("\n".join(json.dumps({**base, **r}) for r in rows) + "\n", encoding="utf-8")


def test_manifest_of_several_folders_event_types_and_default_sports(tmp_path):
    a, b = tmp_path / "soccer", tmp_path / "other"
    _manifest(a, dict(market_id="1.1", event_id="E1", event_name="Arsenal v Chelsea", saved="d/E1/1.1.bz2"))   # no event_type_id
    _manifest(b, dict(market_id="1.2", event_id="E2", event_name="Lakers v Celtics", saved="d/E2/1.2.bz2", event_type_id="7522"),
              dict(market_id="1.3", event_id="E3", event_name="Nadal v Federer", saved="d/E3/1.3.bz2", event_type_id=2),  # int id
              dict(market_id="1.4", event_id="E4", event_name="Foo v Bar", saved="d/E4/1.4.bz2"),                         # none
              dict(market_id="1.5", event_id="E5", event_name="Baz v Qux", saved="d/E5/1.5.bz2", event_type_id="999"))
    assert EVENT_TYPE_SPORT["1"] == "Soccer" and EVENT_TYPE_SPORT["7522"] == "Basketball" and len(EVENT_TYPE_SPORT) == 16
    df = load_manifest([a, str(b)], default_sports=["Soccer", "Handball"])
    r = df.set_index("market_id")
    assert list(df.columns) == COLUMNS and len(df) == 5
    assert list(r["root"]) == [str(a)] + [str(b)] * 4 and list(r["default_sport"]) == ["Soccer"] + ["Handball"] * 4
    assert r.loc["1.1", "event_type_id"] is None and r.loc["1.4", "event_type_id"] is None
    assert (r.loc["1.2", "event_type_id"], r.loc["1.3", "event_type_id"], r.loc["1.5", "event_type_id"]) == ("7522", "2", "999")
    sport = lambda d, **kw: {e: x.sport for e, x in build_event_index_from_manifest(d, **kw).items()}  # noqa: E731
    assert sport(df) == {"E1": "Soccer", "E2": "Basketball", "E3": "Tennis", "E4": "Handball", "E5": "eventType:999"}
    assert sport(df.drop(columns="default_sport"), sport="Rugby Union") == {
        "E1": "Rugby Union", "E2": "Basketball", "E3": "Tennis", "E4": "Rugby Union", "E5": "eventType:999"}
    # a dict keys the default sport by folder; folders left out and a single folder default to Soccer
    d2 = load_manifest([a, b], default_sports={str(b): "Handball"})
    assert list(d2.set_index("market_id").loc[["1.1", "1.4"], "default_sport"]) == ["Soccer", "Handball"]
    one = load_manifest(b)
    assert set(one["root"]) == {str(b)} and set(one["default_sport"]) == {"Soccer"} and sport(one)["E4"] == "Soccer"
    # the same market in two folders: the larger file wins and brings its own root
    _manifest(tmp_path / "dup", dict(market_id="1.1", event_id="E1", event_name="Arsenal v Chelsea", saved="x/1.1.bz2", bytes=99))
    assert load_manifest([a, tmp_path / "dup"]).set_index("market_id").loc["1.1", "root"] == str(tmp_path / "dup")


def split_pro_dir(pro_dir, other):
    """Move the 2026-10 manifest and its market files of the ``pro_dir`` fixture into ``other``."""
    man = pro_dir / "manifest_2026-10.jsonl"
    for line in man.read_text(encoding="utf-8").splitlines():
        rel = json.loads(line)["saved"]
        (other / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(pro_dir / rel, other / rel)
    shutil.move(man, other / man.name)


def test_market_store_resolves_each_file_under_its_own_root(pro_dir, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    split_pro_dir(pro_dir, other)
    df = load_manifest([pro_dir, other])
    assert set(df["root"]) == {str(pro_dir), str(other)}
    store = MarketStore(pro_dir, df)                                   # out_dir is only the fallback
    assert all(m in store for m in df["market_id"]) and store["1.700000003"].market_id == "1.700000003"
    assert (other / df.set_index("market_id").loc["1.700000003", "saved"]).is_file()
    got = enrich(alerts(("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham")), store, build_event_index_from_manifest(df),
                 Config(commissions=(0.0,)))
    assert list(got["status"]) == ["ok", "ok"] and list(got["bb_l0"]) == [2.48, 2.48]
    # without a root column every file is looked up under out_dir, as before: the moved markets are not found
    plain = MarketStore(pro_dir, df.drop(columns="root"))
    assert "1.700000001" in plain and "1.700000003" not in plain and plain.get("1.700000003") is None
    assert MarketStore(other, df)["1.700000001"].market_id == "1.700000001"       # root wins over out_dir


def test_market_store_matches_eager_loading_and_is_lazy(pro_dir):
    df = load_manifest(pro_dir)
    al = alerts(("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham"))
    cfg = Config(commissions=(0.0, 0.05))
    store = MarketStore(pro_dir, df)
    lazy = enrich(al, store, build_event_index_from_manifest(df), cfg)
    mk = load_markets(str(pro_dir))
    eager = enrich(al, mk, build_event_index(mk), cfg)
    assert list(lazy["status"]) == ["ok", "ok"] and list(eager["status"]) == ["ok", "ok"]
    cols = ["bb_l0", "bl_l0", "p_l0", "won", "xback_pl_l0_c0", "xfx_back_lay_h10", "bf_market_id"]
    pd.testing.assert_frame_equal(lazy[cols], eager[cols])
    assert list(lazy["bb_l0"]) == [2.48, 2.48] and list(lazy["p_l0"]) == [2.5, 2.5] and list(lazy["won"]) == [1, 1]
    assert store.loads == 2  # only the two MATCH_ODDS files were parsed; the O/U files were never opened


def test_market_store_mapping_and_lru(pro_dir):
    df = load_manifest(pro_dir)
    A, B = "1.700000001", "1.700000003"
    s1 = MarketStore(pro_dir, df, max_cached=1)
    s1[A], s1[B], s1[A]
    assert s1.loads == 3
    s2 = MarketStore(pro_dir, df, max_cached=2)
    s2[A], s2[B], s2[A]
    assert s2.loads == 2 and s2[A].market_id == A and s2[A] is s2[A]
    assert len(s2) == 4 and A in s2 and "1.999" not in s2 and s2.get("1.999") is None and s2.get("1.999", 7) == 7
    with pytest.raises(KeyError):
        s2["1.999"]
    (pro_dir / df.set_index("market_id").loc[B, "saved"]).unlink()  # in the manifest but the file is gone
    assert B not in s2 and s2.get(B) is not None  # `in` is about the file; an already-cached parse is still served
    assert B not in MarketStore(pro_dir, df) and MarketStore(pro_dir, df).get(B) is None
    assert math.isfinite(s2[A].first_pt)
