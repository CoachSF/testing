"""Parse ShootingBets Telegram alerts (raw TL messages) into the alert table the pipeline reads.

``load_messages`` reads data/telegram/shootingbets_raw.json, ``classify`` sorts every message into a kind,
``parse_alert`` turns one alert text into fields (raising ``AlertParseError`` instead of guessing),
``map_to_betfair`` gives the Betfair marketType / selection for an alert (``bf_market`` None = no rule yet) and
``build_alerts`` joins all of it with the free-alert-window state into ``(alerts_df, messages_df)``.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .matching import over_under_market_type
from .textnorm import normalize_name, strip_accents

SPORTS = {"SOCCER": "Soccer", "BASKETBALL": "Basketball", "HOCKEY": "Ice Hockey", "TENNIS": "Tennis",
          "VOLLEYBALL": "Volleyball", "HANDBALL": "Handball", "BASEBALL": "Baseball"}
# (sport, mkt_kind, scope) -> Betfair marketType for every sport without a built-in rule (measured on the PRO files:
# US-style sports have a 2-runner MATCH_ODDS "Moneyline", HANDICAP and COMBINED_TOTAL with (name, hc) runners; Ice Hockey's
# overtime-inclusive moneyline is MONEY_LINE; Volleyball only has MATCH_ODDS; Handball only the 3-way MATCH_ODDS: no rule)
OTHER_RULES: dict[tuple[str, str, str | None], str] = {
    ("Basketball", "moneyline", None): "MATCH_ODDS", ("Basketball", "handicap", None): "HANDICAP",
    ("Basketball", "total", None): "COMBINED_TOTAL", ("Baseball", "moneyline", None): "MATCH_ODDS",
    ("Baseball", "handicap", None): "HANDICAP", ("Baseball", "total", None): "COMBINED_TOTAL",
    ("Ice Hockey", "moneyline", "ot_incl"): "MONEY_LINE", ("Volleyball", "moneyline", None): "MATCH_ODDS"}
KINDS = ("alert", "window_open", "window_close", "onboarding", "user", "other")

_NUM = r"\d+(?:\.\d+)?"
_MIN_ODDS = re.compile(r"Min\.\s*Odds:\D*?(" + _NUM + ")")
_KICKOFF = re.compile(r"(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2})(\s*\(UTC\))?")
_ALT = re.compile(r"^▪\s*(.+?)\s*-->\s*(" + _NUM + r")\s*$")
_CHECK_HEAD = re.compile(r"\b(Unchecked|Checked):")
_CHECK = re.compile(r"\b(Unchecked|Checked):\s*(.+?)\s*-->\s*(" + _NUM + ")")
_TOTAL = re.compile(r"^(Over|Under)\s+(" + _NUM + ")$", re.I)
_HANDICAP = re.compile(r"^(.*\S)\s+([+-]" + _NUM + ")$")
_SCOPE = re.compile(r"^(.*?)\s*\(([^()]*)\)\s*$")
_SCOPES = {"ot incl.": "ot_incl", "sets": "sets", "dnb": "dnb"}
_HALF_LINES = {x + 0.5 for x in range(9)}


class AlertParseError(ValueError):
    """The alert text is not understood; ``str(exc)`` is a short snake_case reason."""


def classify(text: str, out: bool = False) -> str:
    text = text or ""
    if out:
        return "user"
    if "Selection:" in text and "Min. Odds:" in text:
        return "alert"
    if "unlimited alerts for a short period" in text:
        return "window_open"
    if "Closing unlimited alerts" in text:
        return "window_close"
    if text.startswith("Welcome to ShootingBets") or "verification code" in text \
            or "receive picks according to your subscription" in text:
        return "onboarding"
    return "other"


def _side(team: str, home: str, away: str) -> str:
    t, h, a = (" ".join(s.casefold().split()) for s in (team, home, away))
    hit = (t == h, t == a)
    if not any(hit):
        hit = (t in h or h in t, t in a or a in t)
    if sum(hit) != 1:
        raise AlertParseError("selection_team_not_in_fixture")
    return "home" if hit[0] else "away"


def parse_alert(text: str) -> dict:
    lines = [ln.strip() for ln in (text or "").split("\n") if ln.strip()]
    if not lines:
        raise AlertParseError("no_selection")
    sport_raw = " ".join(re.findall(r"[^\W\d_]+", lines[0])).upper()
    sel = next((ln.split(":", 1)[1].strip() for ln in lines if ln.startswith("Selection:")), "")
    if not sel:
        raise AlertParseError("no_selection")
    i_odds = next((i for i, ln in enumerate(lines) if _MIN_ODDS.search(ln)), None)
    if i_odds is None:
        raise AlertParseError("no_min_odds")
    min_odds = float(_MIN_ODDS.search(lines[i_odds]).group(1))
    i_fix = next((i for i in range(i_odds + 1, len(lines)) if re.search(r"\s+vs\s+", lines[i])), None)
    teams = [s.strip() for s in re.split(r"\s+vs\s+", lines[i_fix], maxsplit=1)] if i_fix is not None else []
    if len(teams) != 2 or not all(teams):
        raise AlertParseError("no_fixture")
    home, away = teams
    league = lines[i_fix + 1] if i_fix + 1 < len(lines) and not _KICKOFF.match(lines[i_fix + 1]) else None
    country = league.split(" - ", 1)[0] if league and " - " in league else None
    k = _KICKOFF.search("\n".join(lines))
    if k is None:
        raise AlertParseError("no_kickoff")
    if k.group(3) is None:
        raise AlertParseError("kickoff_not_utc")
    try:
        kickoff = pd.Timestamp(f"{k.group(1)} {k.group(2)}", tz="UTC")
    except ValueError:
        raise AlertParseError("bad_kickoff") from None

    alternatives = []
    for ln in lines:
        if ln.startswith("▪"):
            a = _ALT.match(ln)
            if a is None:
                raise AlertParseError("bad_alternative")
            alternatives.append({"sel": a.group(1), "odds": float(a.group(2))})
    check_status, check_sel, check_odds = "none", None, None
    for ln in lines:
        if _CHECK_HEAD.search(ln):
            c = _CHECK.search(ln)
            if c is None:
                raise AlertParseError("bad_check_line")
            check_status, check_sel = c.group(1).lower(), re.sub(r"\s+-$", "", c.group(2))
            check_odds = float(c.group(3))
            break

    core, scope = sel, None
    sc = _SCOPE.match(sel)
    if sc:
        core, inner = sc.group(1).strip(), sc.group(2).strip().lower()
        scope = _SCOPES.get(" ".join(inner.split()), inner)
        if not core:
            raise AlertParseError("no_selection")
    team, line, side = None, None, None
    if (t := _TOTAL.match(core)):
        mkt_kind, side, line = "total", t.group(1).lower(), float(t.group(2))
    elif (h := _HANDICAP.match(core)):
        mkt_kind, team, line = "handicap", h.group(1), float(h.group(2))
    elif core.casefold() in ("draw", "x", "tie"):
        mkt_kind, side = "moneyline", "draw"
    else:
        mkt_kind, team = "moneyline", core
    if team is not None:
        side = _side(team, home, away)
    league_s = league or ""
    age = re.search(r"\bU-?(\d{2})\b", league_s, re.I)
    return {
        "sport_raw": sport_raw, "sport": SPORTS.get(sport_raw, sport_raw.title()), "selection_raw": sel,
        "min_odds": min_odds, "home": home, "away": away, "league": league, "country": country,
        "kickoff": kickoff, "alternatives": alternatives, "check_status": check_status, "check_sel": check_sel,
        "check_odds": check_odds, "scope": scope, "mkt_kind": mkt_kind, "sel_side": side, "line": line,
        "team": team, "age_group": f"U{age.group(1)}" if age else None,
        "is_women": bool(re.search(r"women|ladies|femen|femin|femmin|nadeshiko|\(w\)|\bwnbl\b|\bwnba\b",
                                   strip_accents(league_s), re.I)),
        "is_reserve": bool(re.search(r"\breserves?\b", league_s, re.I))}


def map_to_betfair(p: dict) -> dict:
    """Betfair market + selection. ``sel_team`` is always None: the pipeline targets the matched Betfair event's own
    home / away name, which does not depend on how the alert spells the team."""
    sport, kind, scope, side, line = (p.get(k) for k in ("sport", "mkt_kind", "scope", "sel_side", "line"))

    def yes(market, sel_kind, sel_team=None, sel_hc=None, note=""):
        return {"bf_market": market, "sel_kind": sel_kind, "sel_team": sel_team, "sel_hc": sel_hc, "map_note": note}

    def no(why):
        return yes(None, None, note="unmapped:" + why)

    if sport == "Soccer":
        if kind == "moneyline" and scope in (None, "dnb"):
            return yes("MATCH_ODDS" if scope is None else "DRAW_NO_BET", side)
        if kind == "handicap" and scope is None:
            if line == -0.5:
                return yes("MATCH_ODDS", side, note="ah-0.5=match_odds")
            if line == 0.5:
                return yes("DOUBLE_CHANCE", "home_or_draw" if side == "home" else "draw_or_away",
                           note="ah+0.5=double_chance")
            return yes("ASIAN_HANDICAP", side, None, line) if abs(line) % 1 == 0.5 else no("non_half_line")
        if kind == "total" and scope is None:
            return yes(over_under_market_type(line), side) if line in _HALF_LINES else no("total_line")
    elif sport == "Tennis":
        if kind == "moneyline" and scope is None:
            return yes("MATCH_ODDS", side)
        if kind == "handicap":
            return no("tennis_set_handicap")
    market = OTHER_RULES.get((sport, kind, scope))
    if market is None:
        return no(f"no_rule:{sport}:{kind}:{scope}")
    return yes(market, side) if kind == "moneyline" else yes(market, side, None, line)


def load_messages(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)["messages"]
    out = [{"id": int(m["id"]), "ts": pd.Timestamp(int(m["date"]), unit="s", tz="UTC"), "text": m.get("message") or "",
            "out": bool(m.get("out") or (m.get("pFlags") or {}).get("out")), "edited": bool(m.get("edit_date"))}
           for m in raw]
    return sorted(out, key=lambda m: m["id"])


ALERT_COLS = ["alert_id", "ts", "text", "sport_raw", "sport", "selection_raw", "min_odds", "home", "away", "home_raw",
              "away_raw", "home_alt", "away_alt", "league", "country", "kickoff", "alternatives", "n_alt", "check_status",
              "check_sel", "check_odds", "scope", "mkt_kind", "sel_side", "line", "team", "age_group", "is_women",
              "is_reserve", "ref_odds", "mins_to_kickoff",
              "window_id", "in_free_window", "mins_since_window_open", "edited", "repeat_n", "is_repeat",
              "bf_market", "sel_kind", "sel_team", "sel_hc", "map_note"]


def _tagged(name: str, women: bool, age: str | None, reserve: bool) -> str:
    """Betfair carries women's / youth / reserve markers in the team name, the alerts only in the league line: add the
    marker the name lacks (same normalisation as textnorm: the women / res tokens, u plus two digits)."""
    toks = normalize_name(name).split()
    if women and "women" not in toks:
        name += " (W)"
    if age and not any(re.fullmatch(r"u\d\d", t) for t in toks):
        name += f" {age}"
    if reserve and "res" not in toks:
        name += " Res"
    return name


def build_alerts(messages) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, mrows, seen = [], [], {}
    win_id = win_ts = None
    for m in sorted(messages, key=lambda m: m["id"]):
        kind, status = classify(m["text"], m["out"]), ""
        if kind == "window_open":
            win_id, win_ts = m["id"], m["ts"]
        elif kind == "window_close":
            win_id = win_ts = None
        elif kind == "alert":
            try:
                p = parse_alert(m["text"])
            except AlertParseError as exc:
                status = f"error:{exc}"
            else:
                status = "ok"
                names = [_tagged(p[k], p["is_women"], p["age_group"], p["is_reserve"]) for k in ("home", "away")]
                changed = names != [p["home"], p["away"]]
                key = (p["home"].casefold(), p["away"].casefold(), p["kickoff"])
                repeat_n = seen[key] = seen.get(key, -1) + 1
                rows.append({
                    "alert_id": m["id"], "ts": m["ts"], "text": m["text"], **p, "home_raw": p["home"], "away_raw": p["away"],
                    "home": names[0], "away": names[1], "home_alt": p["home"] if changed else None,
                    "away_alt": p["away"] if changed else None,
                    "alternatives": json.dumps(p["alternatives"], ensure_ascii=False), "n_alt": len(p["alternatives"]),
                    "ref_odds": p["min_odds"], "mins_to_kickoff": (p["kickoff"] - m["ts"]).total_seconds() / 60,
                    "window_id": win_id, "in_free_window": win_id is not None,
                    "mins_since_window_open": (m["ts"] - win_ts).total_seconds() / 60 if win_id is not None else float("nan"),
                    "edited": m["edited"], "repeat_n": repeat_n, "is_repeat": repeat_n > 0, **map_to_betfair(p)})
        mrows.append({"id": m["id"], "ts": m["ts"], "kind": kind, "parse_status": status, "text": m["text"]})
    alerts = pd.DataFrame(rows, columns=ALERT_COLS)
    alerts["window_id"] = alerts["window_id"].astype("Int64")
    return alerts, pd.DataFrame(mrows, columns=["id", "ts", "kind", "parse_status", "text"])


def coverage(messages_df: pd.DataFrame) -> dict:
    counts = messages_df["kind"].value_counts()
    n_alert = int(counts.get("alert", 0))
    n_parsed = int((messages_df["parse_status"] == "ok").sum())
    return {"kinds": {k: int(counts.get(k, 0)) for k in KINDS}, "n_messages": int(len(messages_df)),
            "n_alert": n_alert, "n_parsed": n_parsed, "n_unparsed": n_alert - n_parsed,
            "parse_rate": n_parsed / n_alert if n_alert else float("nan")}
