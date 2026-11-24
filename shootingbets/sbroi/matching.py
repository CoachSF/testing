"""Map an alert (sport, teams, kickoff, market, selection) to a Betfair market/runner."""
from __future__ import annotations

from dataclasses import dataclass, field

from .betfair_files import MarketRecord, RunnerSeries
from .textnorm import name_score, split_event_name, surname_score

INDIVIDUAL_SPORTS = {"Tennis", "Darts", "Snooker", "Boxing", "Mixed Martial Arts", "Golf"}
DRAW_NAMES = {"the draw", "draw", "tie"}


@dataclass
class EventInfo:
    event_id: str
    sport: str | None
    event_name: str | None
    open_date_ms: int | None
    country_code: str | None
    home: str | None
    away: str | None
    participants: list[str] = field(default_factory=list)
    markets: dict[str, list[str]] = field(default_factory=dict)  # market_type -> [market_id]


def build_event_index(markets: dict[str, MarketRecord]) -> dict[str, EventInfo]:
    idx: dict[str, EventInfo] = {}
    for mid, m in markets.items():
        if not m.event_id:
            continue
        ev = idx.get(m.event_id)
        if ev is None:
            home = away = None
            if m.event_name:
                sp = split_event_name(m.event_name)
                if sp:
                    home, away, _ = sp
            ev = idx[m.event_id] = EventInfo(
                event_id=m.event_id, sport=m.sport, event_name=m.event_name,
                open_date_ms=m.open_date_ms or m.market_time_ms, country_code=m.country_code,
                home=home, away=away,
            )
        ev.markets.setdefault(m.market_type or "UNKNOWN", []).append(mid)
        if m.market_type in ("MATCH_ODDS", "MONEY_LINE") and not ev.participants:
            rs = sorted(m.runners.values(), key=lambda r: (r.sort_priority or 99))
            ev.participants = [r.name for r in rs if r.name and r.name.strip().lower() not in DRAW_NAMES]
    return idx


@dataclass
class EventMatch:
    event_id: str | None
    score: float
    home_score: float
    away_score: float
    runner_up_score: float
    swapped: bool
    time_diff_min: float | None
    n_candidates: int
    reason: str


def _pair_scores(sport: str | None, a_home: str, a_away: str, ev: EventInfo) -> tuple[float, float, float, float]:
    """Return (home_score, away_score, swapped_home, swapped_away)."""
    f = surname_score if sport in INDIVIDUAL_SPORTS else name_score
    cands_home = [x for x in (ev.home, ev.participants[0] if len(ev.participants) > 0 else None) if x]
    cands_away = [x for x in (ev.away, ev.participants[1] if len(ev.participants) > 1 else None) if x]
    if not cands_home or not cands_away:
        return (0.0, 0.0, 0.0, 0.0)
    hs = max(f(a_home, c) for c in cands_home)
    as_ = max(f(a_away, c) for c in cands_away)
    shs = max(f(a_home, c) for c in cands_away)
    sas = max(f(a_away, c) for c in cands_home)
    return hs, as_, shs, sas


def match_event(index: dict[str, EventInfo], sport: str | None, home: str, away: str,
                kickoff_ms: int | None = None, alert_ms: int | None = None,
                window_min: float = 20.0, min_side: float = 60.0, min_avg: float = 78.0,
                allow_swap: bool = True) -> EventMatch:
    cands = []
    for ev in index.values():
        if sport and ev.sport and ev.sport != sport:
            continue
        if ev.open_date_ms is None:
            continue
        if kickoff_ms is not None:
            dt = abs(ev.open_date_ms - kickoff_ms) / 60_000
            if dt > window_min:
                continue
        elif alert_ms is not None:
            # unknown kickoff: event must start between 6h before (in-play) and 10 days after the alert
            if not (alert_ms - 6 * 3_600_000 <= ev.open_date_ms <= alert_ms + 10 * 86_400_000):
                continue
            dt = (ev.open_date_ms - alert_ms) / 60_000
        else:
            dt = None
        cands.append((ev, dt))
    scored = []
    for ev, dt in cands:
        hs, as_, shs, sas = _pair_scores(sport, home, away, ev)
        direct = (min(hs, as_), (hs + as_) / 2, hs, as_, False)
        swapped = (min(shs, sas), (shs + sas) / 2, shs, sas, True)
        best = max(direct, swapped, key=lambda x: (x[0], x[1])) if allow_swap else direct
        scored.append((best, ev, dt))
    if not scored:
        return EventMatch(None, 0.0, 0.0, 0.0, 0.0, False, None, 0, "no_candidates_in_window")
    scored.sort(key=lambda x: (x[0][1], x[0][0], -(abs(x[2]) if x[2] is not None else 0)), reverse=True)
    (mn, avg, hs, as_, sw), ev, dt = scored[0]
    runner_up = scored[1][0][1] if len(scored) > 1 else 0.0
    ok = mn >= min_side and avg >= min_avg
    reason = "ok" if ok else f"low_score(min={mn:.0f},avg={avg:.0f})"
    if ok and runner_up >= avg - 3 and len(scored) > 1:
        reason = "ambiguous"
    return EventMatch(ev.event_id if ok else None, avg, hs, as_, runner_up, sw,
                      None if dt is None else float(dt), len(scored), reason)


# ---------------------------------------------------------------------------
# Market / runner resolution
# ---------------------------------------------------------------------------

def over_under_market_type(line: float, half: str | None = None) -> str:
    code = f"{int(round(line * 10)):02d}"
    if half in ("1H", "FH", "first_half"):
        return f"FIRST_HALF_GOALS_{code}"
    return f"OVER_UNDER_{code}"


def find_markets(ev: EventInfo, market_type: str) -> list[str]:
    return list(ev.markets.get(market_type, []))


def resolve_runner(m: MarketRecord, ev: EventInfo, kind: str, team: str | None = None,
                   handicap: float | None = None, sport: str | None = None) -> RunnerSeries | None:
    """kind: home | away | draw | over | under | yes | no | team | home_or_draw | draw_or_away | home_or_away"""
    runners = list(m.runners.values())
    if handicap is not None:
        runners = [r for r in runners if abs(r.handicap - handicap) < 1e-9]
    by_name = lambda r: (r.name or "").strip().lower()  # noqa: E731
    if kind == "draw":
        hits = [r for r in runners if by_name(r) in DRAW_NAMES]
        return hits[0] if hits else None
    if kind in ("over", "under", "yes", "no"):
        hits = [r for r in runners if by_name(r).startswith(kind)]
        return hits[0] if len(hits) == 1 else None
    f = surname_score if sport in INDIVIDUAL_SPORTS else name_score
    if kind in ("home", "away", "team"):
        target = team
        if target is None:
            target = ev.home if kind == "home" else ev.away
            if not target and ev.participants:
                target = ev.participants[0 if kind == "home" else 1] if len(ev.participants) > 1 else None
        if not target:
            return None
        cand = [r for r in runners if by_name(r) not in DRAW_NAMES]
        scored = sorted(((f(target, r.name), r) for r in cand if r.name), key=lambda x: x[0], reverse=True)
        if scored and scored[0][0] >= 70 and (len(scored) == 1 or scored[0][0] - scored[1][0] >= 5):
            return scored[0][1]
        # fall back on sort priority (1 = home, 2 = away) for 2/3-way markets
        if kind in ("home", "away") and m.market_type in ("MATCH_ODDS", "MONEY_LINE", "DRAW_NO_BET"):
            want = 1 if kind == "home" else 2
            sp = [r for r in cand if r.sort_priority == want]
            return sp[0] if len(sp) == 1 else None
        return None
    if kind in ("home_or_draw", "draw_or_away", "home_or_away"):
        names = {"home_or_draw": ("home or draw", "1x"), "draw_or_away": ("draw or away", "x2"),
                 "home_or_away": ("home or away", "12")}[kind]
        hits = [r for r in runners if by_name(r) in names]
        return hits[0] if hits else None
    return None
