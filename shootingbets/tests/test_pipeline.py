import numpy as np
import pandas as pd
import pytest

from conftest import MIN, T0
from sbroi.betfair_files import load_markets
from sbroi.matching import build_event_index
from sbroi.pipeline import Config, enrich


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
