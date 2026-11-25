"""sbroi.alerts + scripts/build_alerts.py on synthetic texts shaped like the real alerts (NOT real data)."""
import importlib
import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT
from sbroi import alerts as A
from sbroi.alerts import AlertParseError, build_alerts, classify, coverage, load_messages, map_to_betfair, parse_alert

sys.path.insert(0, str(ROOT / "scripts"))

R = "\U0001f680"
SOCCER, BASKET = "⚽ SOCCER ⚽", "\U0001f3c0 BASKETBALL \U0001f3c0"
HOCKEY, VOLLEY = "\U0001f3d2 HOCKEY \U0001f3d2", "\U0001f3d0 VOLLEYBALL \U0001f3d0"
HANDBALL, TENNIS = "\U0001f93e HANDBALL \U0001f93e", "\U0001f3be TENNIS \U0001f3be"
OPEN = "\U0001f4e2 Attention! \U0001f4e2\nFrom now on you will receive unlimited alerts for a short period of time."
CLOSE = "⚠ Closing unlimited alerts ⚠\nThe free window is over."


def text(sport=SOCCER, sel="Tritium +1.5", odds="1.81", fixture="Chievo vs Tritium ", league="Italy - Serie D",
         kick="2026-10-03 18:45 (UTC)", tail=("Alternatives:", "▪ Tritium +0.5 --> 3.38")):
    """An alert shaped like the real ones (blank line between blocks, trailing space after names); None drops a line."""
    lines = [sport, "", None if sel is None else f"Selection: {sel}", "",
             None if odds is None else f"Min. Odds: {R}{odds}{R}", "", fixture, league, kick, "", *tail]
    return "\n".join(x for x in lines if x is not None)


S1 = text()
S2 = text(sel="Cruzeiro ", odds="1.62", fixture="Cruzeiro vs Athletic Club ", league="Brazil - Mineiro U20",
          kick="2026-10-03 18:00 (UTC)", tail=("❌Unchecked: Cruzeiro U20 - --> 1.62❌",))
S3 = text(sport=BASKET, sel="Over 146.5", odds="1.88", fixture="Monarcas Juana Diaz vs Cariduros de Fajardo ",
          league="Puerto Rico - Superior Nacional Women", kick="2026-10-15 00:00 (UTC)",
          tail=("Alternatives:", "▪ Over 147.5 --> 1.96", "▪ Over 148.5 --> 2.07"))


def utc(s):
    return pd.Timestamp(s, tz="UTC")


REAL_RULES = dict(A.OTHER_RULES)


@pytest.fixture(autouse=True)
def empty_other_rules(monkeypatch):
    """OTHER_RULES is edited by hand; the tests that are not about the table must not depend on what is in it."""
    monkeypatch.setattr(A, "OTHER_RULES", {})


# ----------------------------------------------------------------------------- parse_alert
def test_parse_the_three_real_shapes_exactly():
    assert parse_alert(S1) == {
        "sport_raw": "SOCCER", "sport": "Soccer", "selection_raw": "Tritium +1.5", "min_odds": 1.81, "home": "Chievo",
        "away": "Tritium", "league": "Italy - Serie D", "country": "Italy", "kickoff": utc("2026-10-03 18:45"),
        "alternatives": [{"sel": "Tritium +0.5", "odds": 3.38}], "check_status": "none", "check_sel": None,
        "check_odds": None, "scope": None, "mkt_kind": "handicap", "sel_side": "away", "line": 1.5, "team": "Tritium",
        "age_group": None, "is_women": False, "is_reserve": False}
    assert parse_alert(S2) == {
        "sport_raw": "SOCCER", "sport": "Soccer", "selection_raw": "Cruzeiro", "min_odds": 1.62, "home": "Cruzeiro",
        "away": "Athletic Club", "league": "Brazil - Mineiro U20", "country": "Brazil",
        "kickoff": utc("2026-10-03 18:00"), "alternatives": [], "check_status": "unchecked",
        "check_sel": "Cruzeiro U20", "check_odds": 1.62, "scope": None, "mkt_kind": "moneyline", "sel_side": "home",
        "line": None, "team": "Cruzeiro", "age_group": "U20", "is_women": False, "is_reserve": False}
    assert parse_alert(S3) == {
        "sport_raw": "BASKETBALL", "sport": "Basketball", "selection_raw": "Over 146.5", "min_odds": 1.88,
        "home": "Monarcas Juana Diaz", "away": "Cariduros de Fajardo", "league": "Puerto Rico - Superior Nacional Women",
        "country": "Puerto Rico", "kickoff": utc("2026-10-15 00:00"),
        "alternatives": [{"sel": "Over 147.5", "odds": 1.96}, {"sel": "Over 148.5", "odds": 2.07}],
        "check_status": "none", "check_sel": None, "check_odds": None, "scope": None, "mkt_kind": "total",
        "sel_side": "over", "line": 146.5, "team": None, "age_group": None, "is_women": True, "is_reserve": False}
    assert parse_alert(S1)["kickoff"].tzinfo is not None and issubclass(AlertParseError, ValueError)


@pytest.mark.parametrize("sel", ["Draw", "draw", "X", "Tie", "TIE "])
def test_draw_selection(sel):
    p = parse_alert(text(sel=sel))
    assert (p["mkt_kind"], p["sel_side"], p["team"], p["line"], p["scope"]) == ("moneyline", "draw", None, None, None)


def test_team_names_with_digits_and_integer_lines():
    fx = "Parada 6 vs Projekt Warsaw"
    p = parse_alert(text(sport=VOLLEY, sel="Parada 6 -1.5 (Sets)", fixture=fx))
    assert (p["mkt_kind"], p["team"], p["line"], p["scope"], p["sel_side"]) == ("handicap", "Parada 6", -1.5, "sets", "home")
    p = parse_alert(text(sport=VOLLEY, sel="Projekt Warsaw +2", fixture=fx))
    assert (p["mkt_kind"], p["team"], p["line"], p["sel_side"]) == ("handicap", "Projekt Warsaw", 2.0, "away")
    p = parse_alert(text(sport=VOLLEY, sel="Parada 6", fixture=fx))  # a trailing number WITHOUT a sign is part of the name
    assert (p["mkt_kind"], p["team"], p["line"], p["sel_side"]) == ("moneyline", "Parada 6", None, "home")
    p = parse_alert(text(sel="under 2.5"))
    assert (p["mkt_kind"], p["sel_side"], p["line"], p["team"]) == ("total", "under", 2.5, None)


@pytest.mark.parametrize("sport,sel,fixture,scope,kind,side,team,line", [
    (HOCKEY, "Hockey Unterland Cavaliers  (OT Incl.)", "Hockey Unterland Cavaliers vs Foo", "ot_incl", "moneyline", "home",
     "Hockey Unterland Cavaliers", None),
    (HOCKEY, "Over 5.5 (OT Incl.)", "Foo vs Bar", "ot_incl", "total", "over", None, 5.5),
    (VOLLEY, "Parada 6 -1.5 (Sets)", "Parada 6 vs Foo", "sets", "handicap", "home", "Parada 6", -1.5),
    (TENNIS, "Flavio Cobolli -1.5 (Sets)", "Flavio Cobolli vs Foo Bar", "sets", "handicap", "home", "Flavio Cobolli", -1.5),
    (HANDBALL, "SAG Villa Ballester  (DNB)", "Foo vs SAG Villa Ballester", "dnb", "moneyline", "away", "SAG Villa Ballester", None),
    (SOCCER, "Albirex Niigata  (W)", "Albirex Niigata (W) vs Foo", "w", "moneyline", "home", "Albirex Niigata", None),
    (SOCCER, "Tritium", "Chievo vs Tritium", None, "moneyline", "away", "Tritium", None),
])
def test_scopes(sport, sel, fixture, scope, kind, side, team, line):
    p = parse_alert(text(sport=sport, sel=sel, fixture=fixture))
    assert (p["scope"], p["mkt_kind"], p["sel_side"], p["team"], p["line"]) == (scope, kind, side, team, line)


def test_sports_and_unknown_sport():
    assert [parse_alert(text(sport=s))["sport"] for s in (HOCKEY, VOLLEY, HANDBALL, TENNIS, "⚾ BASEBALL ⚾")] == \
        ["Ice Hockey", "Volleyball", "Handball", "Tennis", "Baseball"]
    p = parse_alert(text(sport="\U0001f3d3 TABLE TENNIS \U0001f3d3"))
    assert (p["sport_raw"], p["sport"]) == ("TABLE TENNIS", "Table Tennis")


def test_check_lines():
    def chk(line):
        p = parse_alert(text(tail=(line,)))
        return p["check_status"], p["check_sel"], p["check_odds"]
    assert chk("✅Checked: Over 170.5 --> 2.15✅") == ("checked", "Over 170.5", 2.15)
    assert chk("❌Unchecked: Parada 6 -2.5 --> 1.90❌") == ("unchecked", "Parada 6 -2.5", 1.90)
    assert chk("❌Unchecked: Cruzeiro U20 - --> 1.62❌") == ("unchecked", "Cruzeiro U20", 1.62)
    assert chk("✅Checked: Albirex Niigata (W) - --> 2.55✅") == ("checked", "Albirex Niigata (W)", 2.55)
    assert chk("✅Checked: Portadown Reserves - --> 2.10✅")[1] == "Portadown Reserves"
    assert parse_alert(text())["check_status"] == "none"


def test_alternatives():
    p = parse_alert(text(tail=("Alternatives:", "▪ Over 147.5 --> 1.96", "▪  Over 148.5   -->   2.07 ")))
    assert p["alternatives"] == [{"sel": "Over 147.5", "odds": 1.96}, {"sel": "Over 148.5", "odds": 2.07}]
    assert parse_alert(text(tail=()))["alternatives"] == []


def test_league_country_age_and_women():
    p = parse_alert(text(league=None))
    assert (p["league"], p["country"]) == (None, None) and p["kickoff"] == utc("2026-10-03 18:45")  # kickoff is not a league
    for league, country in (("West Asia Super League", None), ("ATP Lyon - R16", "ATP Lyon"), ("Brazil - Copa - X", "Brazil")):
        p = parse_alert(text(league=league))
        assert (p["league"], p["country"]) == (league, country)
    assert parse_alert(text(league="Brazil - Copa U-17"))["age_group"] == "U17"
    assert parse_alert(text(league="Brazil - U20 Cup"))["age_group"] == "U20"
    assert parse_alert(text(league="Italy - Serie D"))["age_group"] is None
    e = chr(233)                                                                      # accents are stripped before the test
    for league in ("England - League Cup Women", "Spain - Liga F (W)", "France - Division 1 Feminine", "Ladies League",
                   "Brazil - Brasileiro Feminino", "Spain - Liga FEMENINA", "Wales - Premier Womens League",
                   f"France - Division 1 F{e}minine", f"France - Ligue F{e}minine", "Italy - Serie A Femminile",
                   "Australia - WNBL", "USA - WNBA", "Australia - wnbl cup", "Japan - Nadeshiko Division 2", f"Spain - Liga F{e}mina"):
        assert parse_alert(text(league=league))["is_women"] is True, league
    for league in ("Australia - NBL1", "Australia - WNBLX", "Brazil - S" + e + "rie A", "Spain - La Liga", "USA - NBA"):
        assert parse_alert(text(league=league))["is_women"] is False, league
    assert parse_alert(text(league="Italy - Serie D"))["is_women"] is False


def test_team_matching_rules():
    side = lambda sel, fx: parse_alert(text(sel=sel, fixture=fx))["sel_side"]  # noqa: E731
    assert side("real  MADRID", "Real Madrid vs Barca") == "home"                      # case / whitespace
    assert side("Projekt Warsaw", "Foo vs Projekt Warsaw Volley") == "away"             # substring of one team
    assert side("Projekt Warsaw Volley Team", "Foo vs Projekt Warsaw Volley") == "away"  # contains one team
    for sel, fx in (("Madrid", "Real Madrid vs Atletico Madrid"),   # substring of BOTH -> not guessed
                    ("Zzz", "Real Madrid vs Barca"),                  # neither
                    ("Same", "Same vs Same")):                        # equal to both
        with pytest.raises(AlertParseError) as e:
            side(sel, fx)
        assert str(e.value) == "selection_team_not_in_fixture"


@pytest.mark.parametrize("kw,reason", [
    (dict(sel=None), "no_selection"), (dict(sel="  "), "no_selection"), (dict(sel="(DNB)"), "no_selection"),
    (dict(odds=None), "no_min_odds"), (dict(odds="abc"), "no_min_odds"),
    (dict(fixture=None), "no_fixture"), (dict(fixture="Chievo - Tritium"), "no_fixture"),
    (dict(fixture="vs Tritium"), "no_fixture"), (dict(fixture="Chievo vs"), "no_fixture"),
    (dict(kick=None), "no_kickoff"),
    (dict(kick="2026-10-03 18:45 (CET)"), "kickoff_not_utc"), (dict(kick="2026-10-03 18:45"), "kickoff_not_utc"),
    (dict(kick="2026-13-45 18:45 (UTC)"), "bad_kickoff"),
    (dict(sel="Nobody +1.5"), "selection_team_not_in_fixture"), (dict(sel="Nobody"), "selection_team_not_in_fixture"),
    (dict(tail=("▪ broken",)), "bad_alternative"),
    (dict(tail=("✅Checked: broken✅",)), "bad_check_line"),
])
def test_every_parse_error_reason(kw, reason):
    with pytest.raises(AlertParseError) as e:
        parse_alert(text(**kw))
    assert str(e.value) == reason


def test_a_vs_line_before_min_odds_is_not_the_fixture_and_empty_text():
    t = "\n".join([SOCCER, "A vs B", "Selection: A", "Min. Odds: 1.5", "Italy - Serie D", "2026-10-03 18:45 (UTC)"])
    with pytest.raises(AlertParseError, match="^no_fixture$"):
        parse_alert(t)
    for empty in ("", None, "\n\n"):
        with pytest.raises(AlertParseError, match="^no_selection$"):
            parse_alert(empty)


# ----------------------------------------------------------------------------- map_to_betfair
def P(sport="Soccer", kind="moneyline", scope=None, side="home", team="Foo", line=None):
    return {"sport": sport, "mkt_kind": kind, "scope": scope, "sel_side": side, "team": team, "line": line}


def M(bf_market, sel_kind, sel_team=None, sel_hc=None, map_note=""):
    return {"bf_market": bf_market, "sel_kind": sel_kind, "sel_team": sel_team, "sel_hc": sel_hc, "map_note": map_note}


def unmapped(note):
    return M(None, None, map_note="unmapped:" + note)


def test_map_soccer_moneyline_and_dnb():
    for side in ("home", "away", "draw"):
        team = None if side == "draw" else "Foo"
        assert map_to_betfair(P(side=side, team=team)) == M("MATCH_ODDS", side)        # sel_team is always None
    assert map_to_betfair(P(scope="dnb", side="away")) == M("DRAW_NO_BET", "away")
    assert map_to_betfair(P(scope="w")) == unmapped("no_rule:Soccer:moneyline:w")


def test_map_soccer_handicap():
    h = lambda line, side="home": map_to_betfair(P(kind="handicap", side=side, line=line))  # noqa: E731
    assert h(-0.5) == M("MATCH_ODDS", "home", None, None, "ah-0.5=match_odds")
    assert h(-0.5, "away") == M("MATCH_ODDS", "away", None, None, "ah-0.5=match_odds")
    assert h(0.5) == M("DOUBLE_CHANCE", "home_or_draw", None, None, "ah+0.5=double_chance")
    assert h(0.5, "away") == M("DOUBLE_CHANCE", "draw_or_away", None, None, "ah+0.5=double_chance")
    assert h(-1.5) == M("ASIAN_HANDICAP", "home", None, -1.5) and h(2.5, "away") == M("ASIAN_HANDICAP", "away", None, 2.5)
    for line in (1.0, -2.0, -0.25, 1.75):
        assert h(line) == unmapped("non_half_line")
    assert map_to_betfair(P(kind="handicap", scope="sets", line=-1.5)) == unmapped("no_rule:Soccer:handicap:sets")


def test_map_soccer_total():
    t = lambda line, side="over": map_to_betfair(P(kind="total", side=side, team=None, line=line))  # noqa: E731
    assert t(2.5) == M("OVER_UNDER_25", "over") and t(0.5, "under") == M("OVER_UNDER_05", "under")
    assert t(8.5) == M("OVER_UNDER_85", "over") and t(4.5)["bf_market"] == "OVER_UNDER_45"
    for line in (9.5, 2.0, 2.25, 146.5):
        assert t(line) == unmapped("total_line")


def test_map_tennis():
    assert map_to_betfair(P("Tennis", side="away")) == M("MATCH_ODDS", "away")
    assert map_to_betfair(P("Tennis", "handicap", "sets", line=-1.5)) == unmapped("tennis_set_handicap")
    assert map_to_betfair(P("Tennis", "handicap", None, line=-1.5)) == unmapped("tennis_set_handicap")
    assert map_to_betfair(P("Tennis", "total", None, "over", None, 22.5)) == unmapped("no_rule:Tennis:total:None")
    assert map_to_betfair(P("Tennis", scope="sets")) == unmapped("no_rule:Tennis:moneyline:sets")


def test_map_other_sports_use_the_rules_table(monkeypatch):
    assert A.OTHER_RULES == {}                      # nothing in the table: every other sport is unmapped
    assert map_to_betfair(P("Basketball", "handicap", None, line=-3.5)) == unmapped("no_rule:Basketball:handicap:None")
    assert map_to_betfair(P("Ice Hockey", scope="ot_incl")) == unmapped("no_rule:Ice Hockey:moneyline:ot_incl")
    monkeypatch.setitem(A.OTHER_RULES, ("Basketball", "moneyline", None), "MONEY_LINE")
    monkeypatch.setitem(A.OTHER_RULES, ("Basketball", "handicap", None), "HANDICAP")
    monkeypatch.setitem(A.OTHER_RULES, ("Basketball", "total", None), "TOTAL_POINTS")
    monkeypatch.setitem(A.OTHER_RULES, ("Ice Hockey", "moneyline", "ot_incl"), "MONEY_LINE")
    assert map_to_betfair(P("Basketball", side="away")) == M("MONEY_LINE", "away")
    assert map_to_betfair(P("Basketball", "handicap", None, "home", "Foo", -3.5)) == M("HANDICAP", "home", None, -3.5)
    assert map_to_betfair(P("Basketball", "total", None, "under", None, 160.5)) == M("TOTAL_POINTS", "under", None, 160.5)
    assert map_to_betfair(P("Ice Hockey", scope="ot_incl")) == M("MONEY_LINE", "home")
    assert map_to_betfair(P("Ice Hockey")) == unmapped("no_rule:Ice Hockey:moneyline:None")   # scope is part of the key
    assert map_to_betfair(P("Volleyball")) == unmapped("no_rule:Volleyball:moneyline:None")
    p = parse_alert(S3)                                                      # a real parsed dict feeds it directly
    assert map_to_betfair(p) == M("TOTAL_POINTS", "over", None, 146.5)


def test_other_rules_table_as_measured_on_the_pro_files(monkeypatch):
    assert REAL_RULES == {
        ("Basketball", "moneyline", None): "MATCH_ODDS", ("Basketball", "handicap", None): "HANDICAP",
        ("Basketball", "total", None): "COMBINED_TOTAL", ("Baseball", "moneyline", None): "MATCH_ODDS",
        ("Baseball", "handicap", None): "HANDICAP", ("Baseball", "total", None): "COMBINED_TOTAL",
        ("Ice Hockey", "moneyline", "ot_incl"): "MONEY_LINE", ("Volleyball", "moneyline", None): "MATCH_ODDS"}
    monkeypatch.setattr(A, "OTHER_RULES", REAL_RULES)
    assert map_to_betfair(P("Basketball", "handicap", None, "away", "Foo", 35.5)) == M("HANDICAP", "away", None, 35.5)
    assert map_to_betfair(P("Baseball", "total", None, "under", None, 8.5)) == M("COMBINED_TOTAL", "under", None, 8.5)
    assert map_to_betfair(P("Basketball", side="home")) == M("MATCH_ODDS", "home")
    assert map_to_betfair(P("Ice Hockey", scope="ot_incl", side="away")) == M("MONEY_LINE", "away")
    assert map_to_betfair(P("Volleyball", side="away")) == M("MATCH_ODDS", "away")
    for p in (P("Volleyball", "handicap", "sets", line=-1.5), P("Handball"), P("Handball", scope="dnb"),
              P("Handball", "total", None, "over", None, 55.5), P("Ice Hockey", "total", "ot_incl", "over", None, 5.5)):
        assert map_to_betfair(p)["bf_market"] is None and map_to_betfair(p)["map_note"].startswith("unmapped:no_rule:")


def test_map_the_real_shapes():
    assert map_to_betfair(parse_alert(S1)) == M("ASIAN_HANDICAP", "away", None, 1.5)
    assert map_to_betfair(parse_alert(S2)) == M("MATCH_ODDS", "home")
    assert map_to_betfair(parse_alert(text(sel="Tritium +0.5")))["bf_market"] == "DOUBLE_CHANCE"
    assert map_to_betfair(parse_alert(text(sel="Over 2.5")))["bf_market"] == "OVER_UNDER_25"


# ----------------------------------------------------------------------------- classify
def test_classify_every_kind():
    assert classify(S1) == "alert" and classify(S3, out=False) == "alert"
    assert classify(S1, out=True) == "user" and classify("anything", True) == "user"
    assert classify(OPEN) == "window_open" and classify(CLOSE) == "window_close"
    assert classify("Welcome to ShootingBets, glad to have you") == "onboarding"
    assert classify("Next, you'll receive a verification code by email") == "onboarding"
    assert classify("From now on you'll receive picks according to your subscription") == "onboarding"
    assert classify("hello there") == "other" and classify("") == "other" and classify(None) == "other"
    assert classify("Selection: A only, no odds line") == "other"        # both markers are needed


# ----------------------------------------------------------------------------- load_messages / build_alerts
def msg(i, minutes, body, out=False, edited=False):
    return {"id": i, "ts": utc("2026-10-03 17:00") + pd.Timedelta(minutes=minutes), "text": body, "out": out, "edited": edited}


def test_load_messages(tmp_path):
    p = tmp_path / "raw.json"
    raw = [{"_": "message", "id": 3, "date": 1790000300, "message": "c", "pFlags": {"out": True}},
           {"_": "message", "id": 1, "date": 1790000100, "message": "a", "pFlags": {}, "edit_date": 1790000200},
           {"_": "message", "id": 2, "date": 1790000200, "message": None, "pFlags": {}, "out": True},
           {"_": "message", "id": 4, "date": 1790000400, "pFlags": {}}]
    p.write_text(json.dumps({"chat": {"id": 1}, "messages": raw}), encoding="utf-8")
    got = load_messages(p)
    assert [m["id"] for m in got] == [1, 2, 3, 4]
    assert got[0] == {"id": 1, "ts": pd.Timestamp(1790000100, unit="s", tz="UTC"), "text": "a", "out": False, "edited": True}
    assert [m["out"] for m in got] == [False, True, True, False] and [m["text"] for m in got] == ["a", "", "c", ""]
    assert got[0]["ts"].tzinfo is not None and [m["edited"] for m in got] == [True, False, False, False]
    p.write_text("{}", encoding="utf-8")
    with pytest.raises(KeyError):
        load_messages(p)


def test_build_alerts_windows_repeats_and_coverage():
    S1_upper = text(fixture="CHIEVO vs TRITIUM ")                     # same match as S1 (casefolded), same kickoff
    S1_next_day = text(kick="2026-10-04 18:45 (UTC)")                 # same teams, other kickoff -> not a repeat
    msgs = [msg(1, 0, OPEN), msg(2, 10, S1), msg(3, 20, S2), msg(4, 30, CLOSE), msg(5, 60, S3), msg(6, 120, OPEN),
            msg(7, 125, S1_upper), msg(8, 126, S1, out=True), msg(9, 127, text(fixture=None)),
            msg(10, 130, S1, edited=True), msg(11, 131, "hello"), msg(12, 132, S1_next_day)]
    a, m = build_alerts(list(reversed(msgs)))                         # order of the input does not matter
    assert list(a.columns) == A.ALERT_COLS and list(m.columns) == ["id", "ts", "kind", "parse_status", "text"]
    assert list(a["alert_id"]) == [2, 3, 5, 7, 10, 12]
    assert list(a["window_id"]) == [1, 1, pd.NA, 6, 6, 6] and str(a["window_id"].dtype) == "Int64"
    assert list(a["in_free_window"]) == [True, True, False, True, True, True]
    np.testing.assert_allclose(a["mins_since_window_open"], [10, 20, np.nan, 5, 10, 12])
    assert list(a["repeat_n"]) == [0, 0, 0, 1, 2, 0] and list(a["is_repeat"]) == [False, False, False, True, True, False]
    assert list(a["edited"]) == [False, False, False, False, True, False]
    assert list(a["n_alt"]) == [1, 0, 2, 1, 1, 1] and json.loads(a.loc[0, "alternatives"]) == [{"sel": "Tritium +0.5", "odds": 3.38}]
    assert list(a["ref_odds"]) == list(a["min_odds"]) == [1.81, 1.62, 1.88, 1.81, 1.81, 1.81]
    assert a.loc[0, "mins_to_kickoff"] == pytest.approx(95)           # 17:10 -> 18:45
    assert list(a["bf_market"].fillna("-")) == ["ASIAN_HANDICAP", "MATCH_ODDS", "-", "ASIAN_HANDICAP", "ASIAN_HANDICAP",
                                                "ASIAN_HANDICAP"]
    assert a.loc[2, "map_note"] == "unmapped:no_rule:Basketball:total:None" and a.loc[0, "sel_hc"] == 1.5
    assert a["ts"].dt.tz is not None and a["kickoff"].dt.tz is not None and (a["text"] == [S1, S2, S3, S1_upper, S1, S1_next_day]).all()
    assert list(m["id"]) == list(range(1, 13)) and list(m["text"]) == [x["text"] for x in msgs]
    assert list(m["kind"]) == ["window_open", "alert", "alert", "window_close", "alert", "window_open", "alert", "user",
                               "alert", "alert", "other", "alert"]
    assert list(m["parse_status"]) == ["", "ok", "ok", "", "ok", "", "ok", "", "error:no_fixture", "ok", "", "ok"]
    assert coverage(m) == {"kinds": {"alert": 7, "window_open": 2, "window_close": 1, "onboarding": 0, "user": 1, "other": 1},
                           "n_messages": 12, "n_alert": 7, "n_parsed": 6, "n_unparsed": 1, "parse_rate": 6 / 7}


def test_build_alerts_with_nothing():
    a, m = build_alerts([])
    assert len(a) == 0 and list(a.columns) == A.ALERT_COLS and len(m) == 0
    c = coverage(m)
    assert c["n_alert"] == 0 and c["n_unparsed"] == 0 and np.isnan(c["parse_rate"]) and set(c["kinds"]) == set(A.KINDS)


def _rows(*bodies):
    return build_alerts([msg(i, i, b) for i, b in enumerate(bodies, 1)])[0]


def test_a_league_without_the_word_women_still_tags_the_team_names_the_way_betfair_writes_them():
    from sbroi.textnorm import normalize_name
    a = _rows(text(sel="Over 2.5", fixture="Geelong United vs Southside Flyers ", league="Australia - WNBL"))
    assert a.loc[0, "home"] == "Geelong United (W)" and a.loc[0, "away"] == "Southside Flyers (W)"
    assert a.loc[0, "home_alt"] == "Geelong United" and bool(a.loc[0, "is_women"])
    assert normalize_name(a.loc[0, "home"]) == normalize_name("Geelong United W")           # Betfair's spelling of the same team


def test_is_reserve_from_the_league():
    for league, flag in (("Northern Ireland - Reserve League", True), ("Spain - Liga Reserves", True),
                         ("England - Preserve Cup", False), ("Italy - Serie D", False)):
        assert parse_alert(text(league=league))["is_reserve"] is flag, league


def test_league_tags_are_added_to_the_names_that_lack_them():
    over = dict(sel="Over 2.5")
    a = _rows(
        text(fixture="West Ham United vs Southampton ", league="England - League Cup Women", **over),
        text(fixture="West Ham United (W) vs Southampton W", league="England - League Cup Women", **over),   # already carry it
        text(fixture="Cruzeiro vs Athletic Club U20 ", league="Brazil - Mineiro U20", **over),
        text(fixture="Foo U19 vs Bar", league="Brazil - Copa U-20", **over),                                 # another U-marker
        text(fixture="Linfield vs Glentoran Reserves", league="Northern Ireland - Reserve League", **over),
        text(fixture="Foo vs Bar", league="X - Women U19 Reserves", **over),
        text(fixture="Foo vs Bar", league="Italy - Serie D", **over))
    cols = ["home", "away", "home_raw", "away_raw", "home_alt", "away_alt"]
    got = a[cols].fillna("-").values.tolist()
    assert got == [
        ["West Ham United (W)", "Southampton (W)", "West Ham United", "Southampton", "West Ham United", "Southampton"],
        ["West Ham United (W)", "Southampton W", "West Ham United (W)", "Southampton W", "-", "-"],
        ["Cruzeiro U20", "Athletic Club U20", "Cruzeiro", "Athletic Club U20", "Cruzeiro", "Athletic Club U20"],
        ["Foo U19", "Bar U20", "Foo U19", "Bar", "Foo U19", "Bar"],
        ["Linfield Res", "Glentoran Reserves", "Linfield", "Glentoran Reserves", "Linfield", "Glentoran Reserves"],
        ["Foo (W) U19 Res", "Bar (W) U19 Res", "Foo", "Bar", "Foo", "Bar"],
        ["Foo", "Bar", "Foo", "Bar", "-", "-"]]
    assert list(a["is_reserve"]) == [False, False, False, False, True, True, False]
    assert list(a["is_women"]) == [True, True, False, False, False, True, False]
    assert list(a["repeat_n"]) == [0, 0, 0, 0, 0, 0, 1]                   # repeats are judged on the names as written


# ----------------------------------------------------------------------------- scripts/build_alerts.py
def _raw(path, bodies):
    msgs = [{"_": "message", "id": i, "date": 1790000000 + 60 * i, "message": b, "pFlags": {"out": True} if o else {}}
            for i, (b, o) in enumerate(bodies, 1)]
    path.write_text(json.dumps({"chat": {"id": 1}, "messages": msgs}), encoding="utf-8")


def test_build_alerts_script(tmp_path, capsys):
    ba = importlib.import_module("build_alerts")
    odd = text(sel="Chievo  (ü)")                       # a scope with a non-ASCII letter must still print as ASCII
    bodies = [("Welcome to ShootingBets, SECRET-ONBOARDING", False), (OPEN, False), (S1, False), (odd, False), (S3, False),
              (text(fixture=None), False), ("my own SECRET-USER message", True), (CLOSE, False)]
    raw, out, mout = tmp_path / "raw.json", tmp_path / "o" / "alerts.csv", tmp_path / "o" / "messages.csv"
    _raw(raw, bodies)
    args = ["--in", str(raw), "--out", str(out), "--messages-out", str(mout), "--vendor", str(tmp_path / "no_vendor.json")]
    assert ba.main(args) == 3                                  # one alert did not parse; files are written anyway
    o = capsys.readouterr().out
    assert o.isascii() and "Chievo" not in o and "Tritium" not in o and "SECRET" not in o      # no message text, ASCII only
    assert "alert 4, window_open 1, window_close 1, onboarding 1, user 1, other 0" in o
    assert "alerts: 4  parsed: 3  unparsed: 1" in o and "  6: error:no_fixture" in o and "Melbourne" in o
    assert "SOCCER | handicap | -: 1" in o and "BASKETBALL | total | -: 1" in o and "ASIAN_HANDICAP: 1" in o
    assert "unmapped:no_rule:Soccer:moneyline:\\xfc: 1" in o and "in_free_window: 3  repeats: 1" in o
    a, m = pd.read_csv(out), pd.read_csv(mout)
    assert list(a["alert_id"]) == [3, 4, 5] and list(m["id"]) == list(range(1, 9))
    assert a.loc[0, "ts"].endswith("+00:00") and a.loc[0, "kickoff"] == "2026-10-03T18:45:00+00:00"
    assert list(m["kind"]).count("alert") == 4 and m.loc[m["id"] == 6, "parse_status"].item() == "error:no_fixture"
    assert m.loc[m["id"] == 3, "text"].item() == S1 and m.loc[m["id"] == 7, "text"].item() == bodies[6][0]  # verbatim
    assert a.loc[0, "text"] == S1 and a.loc[0, "bf_market"] == "ASIAN_HANDICAP"
    # every alert parses -> exit 0
    _raw(raw, [b for b in bodies if b[0] != text(fixture=None)])
    assert ba.main(args) == 0 and "unparsed alerts" not in capsys.readouterr().out
    # bad input -> exit 2, nothing new written
    (tmp_path / "notjson.json").write_text("not json", encoding="utf-8")
    (tmp_path / "nomsgs.json").write_text("{}", encoding="utf-8")
    for name in ("missing.json", "notjson.json", "nomsgs.json"):
        assert ba.main(["--in", str(tmp_path / name), "--out", str(tmp_path / "x.csv"), "--messages-out", str(tmp_path / "y.csv")]) == 2
    assert "bad input" in capsys.readouterr().out and not (tmp_path / "x.csv").exists()
