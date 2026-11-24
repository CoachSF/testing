"""Betfair Exchange price ladder (decimal odds) utilities.

Betfair only accepts prices on a fixed ladder of 350 ticks between 1.01 and 1000.
Prices are handled internally as integer hundredths to avoid float drift.

    1.01 - 2.00   step 0.01
    2.02 - 3.00   step 0.02
    3.05 - 4.00   step 0.05
    4.1  - 6.0    step 0.1
    6.2  - 10.0   step 0.2
    10.5 - 20.0   step 0.5
    21   - 30     step 1
    32   - 50     step 2
    55   - 100    step 5
    110  - 1000   step 10
"""
from __future__ import annotations

import bisect
import math

# (inclusive upper bound, increment) in hundredths of a unit of decimal odds
_BANDS = (
    (200, 1),
    (300, 2),
    (400, 5),
    (600, 10),
    (1000, 20),
    (2000, 50),
    (3000, 100),
    (5000, 200),
    (10000, 500),
    (100000, 1000),
)


def _build_ladder() -> list[int]:
    ladder: list[int] = []
    lower = 100
    for upper, inc in _BANDS:
        p = lower + inc
        while p <= upper:
            ladder.append(p)
            p += inc
        lower = upper
    return ladder


LADDER_H: tuple[int, ...] = tuple(_build_ladder())
LADDER: tuple[float, ...] = tuple(h / 100 for h in LADDER_H)
N_TICKS = len(LADDER_H)  # 350
MIN_PRICE = LADDER[0]
MAX_PRICE = LADDER[-1]

assert N_TICKS == 350, N_TICKS


def _to_h(price: float) -> int:
    return int(round(float(price) * 100))


def is_valid_price(price: float) -> bool:
    h = _to_h(price)
    i = bisect.bisect_left(LADDER_H, h)
    return i < N_TICKS and LADDER_H[i] == h


def price_to_tick(price: float, mode: str = "nearest") -> int:
    """Index (0..349) of ``price`` on the ladder.

    mode: 'nearest' (ties go to the lower tick), 'down' (largest tick <= price),
    'up' (smallest tick >= price). Out-of-range prices are clamped.
    """
    if price is None or (isinstance(price, float) and math.isnan(price)):
        raise ValueError("price is None/NaN")
    h = _to_h(price)
    if h <= LADDER_H[0]:
        return 0
    if h >= LADDER_H[-1]:
        return N_TICKS - 1
    i = bisect.bisect_left(LADDER_H, h)
    if LADDER_H[i] == h:
        return i
    lo, hi = i - 1, i  # LADDER_H[lo] < h < LADDER_H[hi]
    if mode == "down":
        return lo
    if mode == "up":
        return hi
    if mode != "nearest":
        raise ValueError(f"unknown mode {mode!r}")
    return lo if (h - LADDER_H[lo]) <= (LADDER_H[hi] - h) else hi


def tick_to_price(tick: int) -> float:
    return LADDER[max(0, min(N_TICKS - 1, int(tick)))]


def snap(price: float, mode: str = "nearest") -> float:
    """Snap an arbitrary decimal price onto the Betfair ladder."""
    return tick_to_price(price_to_tick(price, mode))


def move(price: float, n_ticks: int) -> float:
    """Price ``n_ticks`` away on the ladder (negative = shorter/lower odds). Clamped."""
    return tick_to_price(price_to_tick(price) + int(n_ticks))


def tick_diff(p_from: float, p_to: float) -> int:
    """Signed ladder distance from ``p_from`` to ``p_to``. Negative => price shortened."""
    return price_to_tick(p_to) - price_to_tick(p_from)


def tick_size(price: float) -> float:
    """Increment of the ladder band containing ``price`` (the gap to the next tick up)."""
    t = price_to_tick(price)
    if t >= N_TICKS - 1:
        return (LADDER_H[-1] - LADDER_H[-2]) / 100
    return (LADDER_H[t + 1] - LADDER_H[t]) / 100
