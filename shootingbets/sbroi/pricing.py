"""Point-in-time price lookups on a MarketRecord runner (no look-ahead).

BASIC data only carries the last traded price (LTP). We use the LTP in force at
time t (the most recent trade at or before t) as the proxy for the price you
could have taken at t. With ADVANCED/PRO data, best back / best lay at t are
available too and should be preferred for execution realism.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .betfair_files import MarketRecord, RunnerSeries


@dataclass
class PricePoint:
    price: float
    pt: int          # publish time (ms) of the observation used
    age_ms: int      # t - pt: how stale the observation is at the query time


def _last_at_or_before(ts: list[int], vals: list[float], t_ms: int) -> PricePoint | None:
    i = bisect.bisect_right(ts, t_ms) - 1
    if i < 0:
        return None
    v = vals[i]
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return PricePoint(price=float(v), pt=ts[i], age_ms=t_ms - ts[i])


def ltp_at(r: RunnerSeries, t_ms: int) -> PricePoint | None:
    return _last_at_or_before(r.ltp_t, r.ltp_p, t_ms)


def best_back_at(r: RunnerSeries, t_ms: int) -> PricePoint | None:
    return _last_at_or_before(r.bb_t, r.bb_p, t_ms)


def best_lay_at(r: RunnerSeries, t_ms: int) -> PricePoint | None:
    return _last_at_or_before(r.bl_t, r.bl_p, t_ms)


def traded_volume_at(r: RunnerSeries, t_ms: int) -> float | None:
    p = _last_at_or_before(r.tv_t, r.tv_v, t_ms)
    return None if p is None else p.price


def ltp_path(r: RunnerSeries, t0_ms: int, t1_ms: int) -> list[tuple[int, float]]:
    """LTP observations with t0 < pt <= t1, in time order."""
    i0 = bisect.bisect_right(r.ltp_t, t0_ms)
    i1 = bisect.bisect_right(r.ltp_t, t1_ms)
    return list(zip(r.ltp_t[i0:i1], r.ltp_p[i0:i1]))


def preoff_close(m: MarketRecord, r: RunnerSeries) -> PricePoint | None:
    """Last LTP strictly before the off (the exchange 'closing' price)."""
    off = m.off_ms()
    if off is None:
        return None
    return ltp_at(r, off - 1)


def back_price_at(m: MarketRecord, r: RunnerSeries, t_ms: int, prefer: str = "auto") -> PricePoint | None:
    """Price you could back at, at time t.

    prefer='auto': best available back if ladder data exists, else LTP.
    prefer='ltp' : always LTP.
    """
    if prefer != "ltp" and r.bb_t:
        p = best_back_at(r, t_ms)
        if p is not None:
            return p
    return ltp_at(r, t_ms)


def lay_price_at(m: MarketRecord, r: RunnerSeries, t_ms: int, prefer: str = "auto") -> PricePoint | None:
    if prefer != "ltp" and r.bl_t:
        p = best_lay_at(r, t_ms)
        if p is not None:
            return p
    return ltp_at(r, t_ms)
