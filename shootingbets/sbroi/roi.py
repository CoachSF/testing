"""Settlement, P/L, ROI and the statistics used to judge whether an edge is real."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .betfair_files import RunnerSeries


@dataclass(frozen=True)
class Outcome:
    win: float    # share of the stake that wins
    lose: float   # share of the stake that loses
    void: float   # share refunded
    status: str
    known: bool


def outcome_from_status(status: str | None, market_closed: bool = True) -> Outcome:
    s = (status or "").upper()
    if s in ("WINNER", "PLACED"):
        return Outcome(1.0, 0.0, 0.0, s, True)
    if s == "LOSER":
        return Outcome(0.0, 1.0, 0.0, s, True)
    if s in ("REMOVED", "REMOVED_VACANT"):
        return Outcome(0.0, 0.0, 1.0, s, True)
    return Outcome(0.0, 0.0, 0.0, s or "UNKNOWN", False)


def runner_outcome(r: RunnerSeries, market_closed: bool) -> Outcome:
    if not market_closed:
        return Outcome(0.0, 0.0, 0.0, "NOT_CLOSED", False)
    return outcome_from_status(r.status, market_closed)


# ---------------------------------------------------------------------------
# P/L (per 1 unit of backer's stake)
# ---------------------------------------------------------------------------

def back_pl(odds: float, oc: Outcome, commission: float = 0.0) -> float:
    gross = oc.win * (odds - 1.0) - oc.lose
    return gross * (1.0 - commission) if gross > 0 else gross


def lay_pl(odds: float, oc: Outcome, commission: float = 0.0) -> float:
    """Lay of 1 unit backer's stake (liability = odds - 1)."""
    gross = oc.lose * 1.0 - oc.win * (odds - 1.0)
    return gross * (1.0 - commission) if gross > 0 else gross


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def max_drawdown(pl_in_time_order: np.ndarray) -> float:
    if len(pl_in_time_order) == 0:
        return 0.0
    eq = np.cumsum(pl_in_time_order)
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float(np.max(peak - eq))


def longest_losing_streak(pl_in_time_order: np.ndarray) -> int:
    best = cur = 0
    for v in pl_in_time_order:
        if v < 0:
            cur += 1
            best = max(best, cur)
        elif v > 0:
            cur = 0
    return best


def bootstrap_roi(pl: np.ndarray, stake: np.ndarray, n_boot: int = 10000,
                  seed: int = 12345) -> np.ndarray:
    n = len(pl)
    if n == 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    s = stake[idx].sum(axis=1)
    p = pl[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return p / s


def null_sim_pvalue(odds: np.ndarray, observed_roi: float, commission: float = 0.0,
                    n_sim: int = 20000, seed: int = 54321, margin: float = 0.0) -> float:
    """P(ROI >= observed) if every bet won with probability (1/odds)*(1-margin).

    margin=0 treats the exchange price you took as fair (zero-edge null).
    """
    n = len(odds)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    pwin = np.clip((1.0 / odds) * (1.0 - margin), 0, 1)
    wins = rng.random((n_sim, n)) < pwin
    gross_win = (odds - 1.0) * (1.0 - commission)
    pl = np.where(wins, gross_win, -1.0).sum(axis=1)
    roi = pl / n
    return float((np.sum(roi >= observed_roi) + 1) / (n_sim + 1))


@dataclass
class Summary:
    n: int
    stake: float
    pl: float
    roi: float
    roi_ci_lo: float
    roi_ci_hi: float
    p_boot_le0: float
    p_null: float
    strike_rate: float
    avg_odds: float
    max_drawdown: float
    longest_losing_streak: int

    def as_dict(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in self.__dict__.items()}


def summarize(pl: np.ndarray, odds: np.ndarray, won: np.ndarray, stake: np.ndarray | None = None,
              commission: float = 0.0, n_boot: int = 10000, run_null: bool = True) -> Summary:
    pl = np.asarray(pl, dtype=float)
    odds = np.asarray(odds, dtype=float)
    won = np.asarray(won, dtype=float)
    stake = np.ones_like(pl) if stake is None else np.asarray(stake, dtype=float)
    n = len(pl)
    if n == 0:
        nan = float("nan")
        return Summary(0, 0.0, 0.0, nan, nan, nan, nan, nan, nan, nan, 0.0, 0)
    tot_stake = float(stake.sum())
    tot_pl = float(pl.sum())
    roi = tot_pl / tot_stake if tot_stake else float("nan")
    boots = bootstrap_roi(pl, stake, n_boot=n_boot)
    lo, hi = (float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))) if len(boots) else (float("nan"),) * 2
    p_le0 = float(np.mean(boots <= 0)) if len(boots) else float("nan")
    p_null = null_sim_pvalue(odds, roi, commission) if run_null else float("nan")
    return Summary(
        n=n, stake=tot_stake, pl=tot_pl, roi=roi, roi_ci_lo=lo, roi_ci_hi=hi,
        p_boot_le0=p_le0, p_null=p_null,
        strike_rate=float(np.mean(won)), avg_odds=float(np.mean(odds)),
        max_drawdown=max_drawdown(pl), longest_losing_streak=longest_losing_streak(pl),
    )


def clv_stats(p_taken: np.ndarray, p_close: np.ndarray) -> dict:
    """Closing-line value: did you get a bigger price than the exchange's closing price?

    'ev_vs_close' = mean(p_taken/p_close - 1) is the expected ROI (pre-commission)
    *if* the closing price is an unbiased probability estimate - a far less noisy
    edge estimate than realised ROI on a few hundred bets.
    """
    p_taken = np.asarray(p_taken, dtype=float)
    p_close = np.asarray(p_close, dtype=float)
    ok = np.isfinite(p_taken) & np.isfinite(p_close) & (p_close > 1) & (p_taken > 1)
    a, c = p_taken[ok], p_close[ok]
    if len(a) == 0:
        return {"n": 0}
    ratio = a / c
    ev = ratio - 1.0
    rng = np.random.default_rng(777)
    boots = ev[rng.integers(0, len(ev), size=(5000, len(ev)))].mean(axis=1)
    return {
        "n": int(len(a)),
        "ev_vs_close": float(ev.mean()),
        "ev_ci_lo": float(np.percentile(boots, 2.5)),
        "ev_ci_hi": float(np.percentile(boots, 97.5)),
        "mean_log_ratio": float(np.log(ratio).mean()),
        "pct_beat_close": float(np.mean(ratio > 1.0)),
        "pct_equal_close": float(np.mean(ratio == 1.0)),
        "median_ratio": float(np.median(ratio)),
    }


def benjamini_hochberg(pvals: list[float], q: float = 0.10) -> list[bool]:
    """Return which hypotheses survive BH FDR control at level q."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (pvals[i] if pvals[i] == pvals[i] else 1.0))
    passed = [False] * m
    k_max = 0
    for rank, i in enumerate(order, start=1):
        p = pvals[i] if pvals[i] == pvals[i] else 1.0
        if p <= q * rank / m:
            k_max = rank
    for rank, i in enumerate(order, start=1):
        if rank <= k_max:
            passed[i] = True
    return passed
