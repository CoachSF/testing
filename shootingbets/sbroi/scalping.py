"""Trading-out ("scalping") simulations around an alert time t0.

Two directions:
  * back_lay  : back at t0, lay later at a shorter price  -> profits if price shortens
  * lay_back  : lay at t0, back later at a longer price   -> profits if price drifts

Green-up P/L (equal profit on every outcome), per 1 unit of the *opening* stake:
  back B then lay at L  : S*(B/L - 1)
  lay  L0 then back at B1: S*(1 - L0/B1)
Commission is charged on the positive net market profit.

Execution realism with BASIC (LTP-only) data is limited: we assume trades at LTP
and model the bid/ask spread with tick penalties (``entry_pen`` / ``exit_pen``).
Resting take-profit orders are assumed filled when LTP *reaches* (mode 'touch')
or trades *through* (mode 'through', more conservative) the target.
"""
from __future__ import annotations

from dataclasses import dataclass

from .betfair_files import RunnerSeries
from .pricing import ltp_at, ltp_path
from .ticks import move, tick_diff


def green_back_lay(p_back: float, p_lay: float, commission: float = 0.0) -> float:
    g = p_back / p_lay - 1.0
    return g * (1.0 - commission) if g > 0 else g


def green_lay_back(p_lay: float, p_back: float, commission: float = 0.0) -> float:
    g = 1.0 - p_lay / p_back
    return g * (1.0 - commission) if g > 0 else g


def _entry(p0: float, side: str, pen: int) -> float:
    # backing: worse = lower odds; laying: worse = higher odds
    return move(p0, -pen) if side == "back_lay" else move(p0, +pen)


def _market_exit(p: float, side: str, pen: int) -> float:
    # closing a back with a lay: worse = higher odds; closing a lay with a back: worse = lower
    return move(p, +pen) if side == "back_lay" else move(p, -pen)


def _green(side: str, p_open: float, p_close: float, commission: float) -> float:
    if side == "back_lay":
        return green_back_lay(p_open, p_close, commission)
    return green_lay_back(p_open, p_close, commission)


@dataclass
class TradeResult:
    side: str
    entry_price: float | None
    exit_price: float | None
    exit_reason: str          # tp | sl | time | off | no_entry | no_exit
    exit_pt: int | None
    green_pl: float | None    # per unit opening stake, after commission
    ticks_moved: int | None   # signed ladder move entry->exit (negative = shortened)


def fixed_horizon(r: RunnerSeries, t0: int, horizon_ms: int, off_ms: int | None,
                  side: str = "back_lay", commission: float = 0.0,
                  entry_pen: int = 0, exit_pen: int = 0,
                  stay_preoff: bool = True) -> TradeResult:
    """Open at t0, close at t0+horizon (or just before the off if earlier)."""
    p0 = ltp_at(r, t0)
    if p0 is None:
        return TradeResult(side, None, None, "no_entry", None, None, None)
    t1 = t0 + horizon_ms
    reason = "time"
    if stay_preoff and off_ms is not None and t1 >= off_ms:
        t1 = off_ms - 1
        reason = "off"
    if t1 <= t0:
        return TradeResult(side, p0.price, None, "no_exit", None, None, None)
    p1 = ltp_at(r, t1)
    if p1 is None:
        return TradeResult(side, p0.price, None, "no_exit", None, None, None)
    e = _entry(p0.price, side, entry_pen)
    x = _market_exit(p1.price, side, exit_pen)
    return TradeResult(side, e, x, reason, t1, _green(side, e, x, commission), tick_diff(p0.price, p1.price))


def tp_sl(r: RunnerSeries, t0: int, tp_ticks: int, sl_ticks: int | None, max_ms: int,
          off_ms: int | None, side: str = "back_lay", commission: float = 0.0,
          entry_pen: int = 0, exit_pen: int = 0, fill: str = "touch",
          stay_preoff: bool = True) -> TradeResult:
    """Open at t0 with a resting take-profit ``tp_ticks`` away; optional stop-loss
    ``sl_ticks`` against; otherwise exit at t0+max_ms or just before the off."""
    p0 = ltp_at(r, t0)
    if p0 is None:
        return TradeResult(side, None, None, "no_entry", None, None, None)
    e = _entry(p0.price, side, entry_pen)
    t_end = t0 + max_ms
    end_reason = "time"
    if stay_preoff and off_ms is not None and t_end >= off_ms:
        t_end = off_ms - 1
        end_reason = "off"
    if side == "back_lay":
        target = move(e, -tp_ticks)
        stop = move(e, +sl_ticks) if sl_ticks is not None else None
        hit_tp = (lambda p: p <= (move(target, -1) if fill == "through" else target))
        hit_sl = (lambda p: stop is not None and p >= stop)
    else:
        target = move(e, +tp_ticks)
        stop = move(e, -sl_ticks) if sl_ticks is not None else None
        hit_tp = (lambda p: p >= (move(target, +1) if fill == "through" else target))
        hit_sl = (lambda p: stop is not None and p <= stop)
    for pt, p in ltp_path(r, t0, t_end):
        if hit_tp(p):
            return TradeResult(side, e, target, "tp", pt, _green(side, e, target, commission), tick_diff(e, target))
        if hit_sl(p):
            x = _market_exit(p, side, exit_pen)
            return TradeResult(side, e, x, "sl", pt, _green(side, e, x, commission), tick_diff(e, p))
    p1 = ltp_at(r, t_end)
    if p1 is None:
        return TradeResult(side, e, None, "no_exit", None, None, None)
    x = _market_exit(p1.price, side, exit_pen)
    return TradeResult(side, e, x, end_reason, t_end, _green(side, e, x, commission), tick_diff(e, p1.price))


def event_study(r: RunnerSeries, t0: int, offsets_ms: list[int], off_ms: int | None,
                stay_preoff: bool = True) -> dict[int, int | None]:
    """Signed tick move from LTP(t0) to LTP(t0+offset) for each offset (may be negative).

    Offsets landing at/after the off are returned as None when stay_preoff=True
    (pre-off and in-play prices are not comparable)."""
    p0 = ltp_at(r, t0)
    out: dict[int, int | None] = {}
    for off in offsets_ms:
        t = t0 + off
        if p0 is None or (stay_preoff and off_ms is not None and t >= off_ms):
            out[off] = None
            continue
        p = ltp_at(r, t)
        out[off] = None if p is None else tick_diff(p0.price, p.price)
    return out
