"""Analysis layer on top of ``pipeline.enrich``: legs (bets), statistics, segments, scalping, matching review.

Pure functions, DataFrames in / DataFrames out.  Conventions (owner's standing rules):
  * P/L is per 1 unit of the BACKER's stake for backs and lays; lays also report ROL = sum(pl) / sum(odds - 1).
  * Void results are excluded from every ROI / strike-rate / count and the exclusions are counted by reason.
  * Nothing observed after the alert time selects or prices an entry (legs read the ``*_l{latency}`` columns).
  * Bets on one match are not independent: CIs are given i.i.d. and clustered by event.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from . import roi as _roi
from . import segments as _seg
from .pipeline import _ctag

REAL_COMMISSION = 0.048
REASONS = ("not_matched", "no_result", "void", "in_play", "no_price", "no_pl")
NAN = float("nan")

# (side, price) -> (odds column, P/L column)
_LEG_COLS = {
    ("back", "ltp"): ("p_l{L}", "back_pl_l{L}_{c}"),
    ("lay", "ltp"): ("p_l{L}", "lay_pl_l{L}_{c}"),
    ("back", "book"): ("bb_l{L}", "xback_pl_l{L}_{c}"),
    ("lay", "book"): ("bl_l{L}", "xlay_pl_l{L}_{c}"),
    ("back", "ref"): ("ref_odds", "ref_back_pl"),
    ("back", "close"): ("p_close", "close_back_pl_c0"),
}
_BASE = ("status", "ts", "alert_id", "match_event_id", "won", "void")
_HEAVY = ("back_pl_", "lay_pl_", "xback_pl_", "xlay_pl_", "xclose_", "close_back_pl", "ref_back_pl",
          "fx_", "xfx_", "tp_", "es_")
LEG_COLS = ["alert_id", "ts", "event", "pl", "odds", "won", "liability", "p_close", "clv"]
SUMMARY_FIELDS = ["n", "n_events", "roi", "roi_ci_lo", "roi_ci_hi", "roi_cl_lo", "roi_cl_hi", "p_null",
                  "strike_rate", "avg_odds", "max_drawdown", "longest_losing_streak", "rol",
                  "clv_n", "clv_mean", "clv_ci_lo", "clv_ci_hi", "clv_pct_pos"]
_CAP = 8_000_000  # max random cells per simulation chunk


def _to_ms(ts: pd.Series) -> np.ndarray:
    """Epoch milliseconds of a tz-aware / naive-UTC timestamp series (independent of the datetime unit)."""
    t = pd.to_datetime(ts, utc=True)
    return ((t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)).to_numpy(float)


# --------------------------------------------------------------------------------------------- a. features
def _cut(x: pd.Series, bins, labels, right=True) -> pd.Series:
    return pd.cut(x, bins=bins, labels=labels, right=right).astype(object)


def add_features(enriched: pd.DataFrame) -> pd.DataFrame:
    df = enriched.copy()
    skipped: list[str] = []
    inf = np.inf

    def num(c):
        return pd.to_numeric(df[c], errors="coerce")

    if "bb_l0" in df or "p_l0" in df:
        s = num("bb_l0") if "bb_l0" in df else pd.Series(np.nan, index=df.index)
        if "p_l0" in df:
            s = s.fillna(num("p_l0"))
        df["odds_band"] = _cut(s, [-inf, 1.5, 2, 2.5, 3.5, 6, inf], ["<=1.5", "1.5-2", "2-2.5", "2.5-3.5", "3.5-6", ">6"])
    else:
        skipped.append("odds_band")
    if "mins_to_off" in df:
        df["mins_to_off_band"] = _cut(num("mins_to_off"), [-inf, 15, 60, 180, 720, inf],
                                      ["<15", "15-60", "60-180", "180-720", "720+"], right=False)
    else:
        skipped.append("mins_to_off_band")
    ref = num("ref_odds").where(lambda s: s > 0) if "ref_odds" in df else None
    for name, col in (("gap_book", "bb_l0"), ("gap_ltp", "p_l0")):
        if ref is not None and col in df:
            df[name] = num(col) / ref - 1
        else:
            skipped.append(name)
    if "ts" in df:
        t = pd.to_datetime(df["ts"], utc=True)
        df["hour_utc"] = t.dt.hour
        df["dow"] = t.dt.day_name().str[:3].astype(object)
    else:
        skipped += ["hour_utc", "dow"]
    if "spread_ticks_l0" in df:
        df["spread_band"] = _cut(num("spread_ticks_l0"), [-inf, 1, 2, 5, inf], ["<=1", "2", "3-5", ">5"])
    else:
        skipped.append("spread_band")
    if "bb_l0_size" in df:
        df["liq_band"] = _cut(num("bb_l0_size"), [-inf, 10, 50, 250, inf], ["<10", "10-50", "50-250", "250+"], right=False)
    else:
        skipped.append("liq_band")
    if "drop_pct" in df:
        try:
            q = pd.qcut(num("drop_pct"), 4, labels=False, duplicates="drop")
        except ValueError:
            q = pd.Series(np.nan, index=df.index)
        df["drop_band"] = q.map(lambda v: f"Q{int(v) + 1}" if pd.notna(v) else None).astype(object)
    else:
        skipped.append("drop_band")
    df.attrs["features_skipped"] = skipped
    return df


# --------------------------------------------------------------------------------------------- b. legs
def leg_columns(side: str, price: str, latency: int = 0, commission: float = REAL_COMMISSION) -> tuple[str, str]:
    """(odds column, P/L column) of a leg in the enriched frame."""
    try:
        o, p = _LEG_COLS[(side, price)]
    except KeyError:
        raise ValueError(f"unsupported leg side={side!r} price={price!r}") from None
    return o.format(L=int(latency)), p.format(L=int(latency), c=_ctag(commission))


def leg_missing(enriched: pd.DataFrame, side: str, price: str, latency: int = 0,
                commission: float = REAL_COMMISSION, preoff_only: bool = True) -> list[str]:
    need = [*_BASE, *leg_columns(side, price, latency, commission)] + (["in_play_at_alert"] if preoff_only else [])
    return [c for c in need if c not in enriched.columns]


def leg_frame(enriched: pd.DataFrame, side: str, price: str, latency: int = 0,
              commission: float = REAL_COMMISSION, preoff_only: bool = True) -> pd.DataFrame:
    """One row per bet that survives the exclusions; ``attrs["excluded"]`` counts every excluded row by reason."""
    miss = leg_missing(enriched, side, price, latency, commission, preoff_only)
    if miss:
        raise KeyError(f"columns missing: {miss}")
    ocol, pcol = leg_columns(side, price, latency, commission)
    df = enriched
    num = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy(float)  # noqa: E731
    won, void, odds, pl = num("won"), num("void"), num(ocol), num(pcol)
    known = ~np.isnan(won)
    if "result_known" in df:
        known &= df["result_known"].eq(True).to_numpy()
    alive = np.ones(len(df), bool)
    excl = dict.fromkeys(REASONS, 0)
    for reason, bad in (
        ("not_matched", df["status"].ne("ok").to_numpy()),
        ("no_result", ~known),
        ("void", np.nan_to_num(void) > 0),
        ("in_play", df["in_play_at_alert"].eq(True).to_numpy() if preoff_only else np.zeros(len(df), bool)),
        ("no_price", ~(np.isfinite(odds) & (odds > 1))),
        ("no_pl", ~np.isfinite(pl)),
    ):
        excl[reason] = int((alive & bad).sum())
        alive &= ~bad
    d = df[alive].reset_index(drop=True)
    o, p, w = odds[alive], pl[alive], won[alive]
    pc = pd.to_numeric(d["p_close"], errors="coerce").to_numpy(float) if "p_close" in d else np.full(len(d), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        clv = np.where(np.isfinite(pc) & (pc > 1), (o / pc - 1) if side == "back" else (pc / o - 1), np.nan)
    computed = pd.DataFrame({
        "alert_id": d["alert_id"].to_numpy(), "ts": pd.to_datetime(d["ts"], utc=True),
        "event": d["match_event_id"].to_numpy(), "pl": p, "odds": o,
        "won": w if side == "back" else 1.0 - w,
        "liability": np.ones(len(d)) if side == "back" else o - 1.0, "p_close": pc, "clv": clv}, index=d.index)
    skip = {*LEG_COLS, "won", "void", "match_event_id", "status"}
    carry = [c for c in d.columns if c not in skip and not str(c).startswith(_HEAVY)]
    out = pd.concat([computed, d[carry]], axis=1).sort_values("ts", kind="stable").reset_index(drop=True)
    out.attrs.update(excluded=excl, n_input=int(len(df)), side=side, price=price, latency=int(latency),
                     commission=commission)
    return out


# --------------------------------------------------------------------------------------------- c, d. nulls & CIs
def _null_back(odds, obs, commission, n_sim, seed):
    n = len(odds)
    if n * n_sim <= _CAP:
        return _roi.null_sim_pvalue(odds, obs, commission, n_sim, seed)
    step, ge, done, i = max(100, _CAP // n), 0, 0, 0
    while done < n_sim:  # chunked so a big leg never allocates n_sim x n at once; counts are recovered exactly
        m = min(step, n_sim - done)
        p = _roi.null_sim_pvalue(odds, obs, commission, m, seed + i)
        ge += int(round(p * (m + 1))) - 1
        done, i = done + m, i + 1
    return (ge + 1) / (n_sim + 1)


def null_p(leg: pd.DataFrame, side: str, commission: float, n_sim: int = 20000, seed: int = 54321) -> float:
    """P(ROI >= observed) if every selection won with probability 1/odds (the price taken is fair)."""
    n = len(leg)
    if n == 0:
        return NAN
    odds, obs = leg["odds"].to_numpy(float), float(leg["pl"].mean())
    if side == "back":
        return _null_back(odds, obs, commission, n_sim, seed)
    rng = np.random.default_rng(seed)
    pwin, win_pl, lose_pl = 1.0 / odds, -(odds - 1.0), 1.0 - commission
    step, ge, done = max(100, _CAP // n), 0, 0
    while done < n_sim:
        m = min(step, n_sim - done)
        sim = np.where(rng.random((m, n)) < pwin, win_pl, lose_pl).sum(axis=1) / n
        ge += int((sim >= obs - 1e-12).sum())
        done += m
    return (ge + 1) / (n_sim + 1)


def cluster_ci(pl, clusters, n_boot: int = 4000, seed: int = 2024) -> tuple[float, float]:
    """95% bootstrap CI of the mean, resampling whole clusters (events) with replacement."""
    pl = np.asarray(pl, float)
    if len(pl) == 0:
        return NAN, NAN
    codes = pd.factorize(pd.Series(np.asarray(clusters, dtype=object)))[0].astype(np.int64)
    neg = codes < 0  # missing cluster id: each such bet is its own cluster
    codes[neg] = codes.max() + 1 + np.arange(int(neg.sum()))
    k = int(codes.max()) + 1
    if k < 2:
        return NAN, NAN
    s, c = np.bincount(codes, weights=pl, minlength=k), np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    step, parts = max(1, 4_000_000 // k), []
    for lo in range(0, n_boot, step):
        idx = rng.integers(0, k, size=(min(step, n_boot - lo), k))
        parts.append(s[idx].sum(axis=1) / c[idx].sum(axis=1))
    m = np.concatenate(parts)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# --------------------------------------------------------------------------------------------- e. summary
def _empty_summary() -> dict:
    d = dict.fromkeys(SUMMARY_FIELDS, NAN)
    d.update(n=0, n_events=0, clv_n=0)
    return d


def summarize_leg(leg: pd.DataFrame, side: str, commission: float, n_sim: int = 20000) -> dict:
    n = len(leg)
    if n == 0:
        return _empty_summary()
    pl, odds, won = (leg[c].to_numpy(float) for c in ("pl", "odds", "won"))
    s = _roi.summarize(pl, odds, won, commission=commission, n_boot=int(np.clip(15_000_000 // n, 500, 4000)),
                       run_null=False)
    ev = leg["event"].to_numpy(dtype=object)
    cl_lo, cl_hi = cluster_ci(pl, ev)
    clv = leg["clv"].to_numpy(float)
    ok = np.isfinite(clv)
    c_lo, c_hi = cluster_ci(clv[ok], ev[ok]) if ok.any() else (NAN, NAN)
    return {
        "n": n, "n_events": int(pd.Series(ev).nunique(dropna=False)), "roi": s.roi,
        "roi_ci_lo": s.roi_ci_lo, "roi_ci_hi": s.roi_ci_hi, "roi_cl_lo": cl_lo, "roi_cl_hi": cl_hi,
        "p_null": null_p(leg, side, commission, n_sim=n_sim), "strike_rate": s.strike_rate, "avg_odds": s.avg_odds,
        "max_drawdown": s.max_drawdown, "longest_losing_streak": s.longest_losing_streak,
        "rol": float(pl.sum() / leg["liability"].sum()) if side == "lay" else NAN,
        "clv_n": int(ok.sum()), "clv_mean": float(clv[ok].mean()) if ok.any() else NAN,
        "clv_ci_lo": c_lo, "clv_ci_hi": c_hi, "clv_pct_pos": float((clv[ok] > 0).mean()) if ok.any() else NAN}


# --------------------------------------------------------------------------------------------- f. headline
def headline(enriched: pd.DataFrame, commissions=(0.0, 0.02, 0.048, 0.05), latencies=(0, 60, 120, 300)) -> pd.DataFrame:
    specs = [(sd, pr, lat, c) for sd in ("back", "lay") for pr in ("ltp", "book") for lat in latencies for c in commissions]
    specs += [("back", "ref", 0, 0.0), ("back", "close", 0, 0.0)]
    rows = []
    for side, price, lat, c in specs:
        base = {"side": side, "price": price, "latency_s": lat if price in ("ltp", "book") else NAN, "commission": c}
        if leg_missing(enriched, side, price, lat, c):
            rows.append({**base, **_empty_summary(), "excluded": "{}", "note": "columns missing"})
            continue
        leg = leg_frame(enriched, side, price, lat, c)
        rows.append({**base, **summarize_leg(leg, side, c), "excluded": json.dumps(leg.attrs["excluded"]),
                     "note": "" if len(leg) else "no bets after exclusions"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------- g. event study
def _eligible(enriched: pd.DataFrame) -> pd.DataFrame:
    """Alerts matched to a Betfair runner and placed before the off."""
    m = enriched["status"].eq("ok")
    if "in_play_at_alert" in enriched:
        m &= ~enriched["in_play_at_alert"].eq(True)
    return enriched[m]


def event_study_table(enriched: pd.DataFrame) -> pd.DataFrame:
    cols = ["offset_min", "n", "mean_ticks", "median_ticks", "ci_lo", "ci_hi", "pct_shorter", "pct_longer"]
    e = _eligible(enriched)
    found = sorted((int(m[1]), c) for c in e.columns if (m := re.fullmatch(r"es_(-?\d+)", c)))
    rows = []
    for off, c in found:
        v = pd.to_numeric(e[c], errors="coerce")
        ok = v.notna().to_numpy()
        x = v.to_numpy(float)[ok]
        if not len(x):
            rows.append(dict.fromkeys(cols, NAN) | {"offset_min": off, "n": 0})
            continue
        lo, hi = cluster_ci(x, e["match_event_id"].to_numpy(dtype=object)[ok])
        rows.append({"offset_min": off, "n": int(len(x)), "mean_ticks": float(x.mean()), "median_ticks": float(np.median(x)),
                     "ci_lo": lo, "ci_hi": hi, "pct_shorter": float((x < 0).mean()), "pct_longer": float((x > 0).mean())})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------------- h. segments
def _prep(leg: pd.DataFrame, num_cols) -> pd.DataFrame:
    leg = leg.copy()
    for c in num_cols:
        if c in leg:
            leg[c] = pd.to_numeric(leg[c], errors="coerce")
    return leg


def scan_segments(leg: pd.DataFrame, side: str, commission: float, cat_cols, num_cols, min_n: int = 20) -> pd.DataFrame:
    leg = _prep(leg, num_cols)
    cands, skipped, why = [], [], {}
    for kind, cols in (("cat", cat_cols), ("num", num_cols)):
        for c in cols:
            if c not in leg or leg[c].notna().sum() == 0:
                skipped.append(c)
                why[c] = "missing" if c not in leg else "all NaN"
                continue
            rules = _seg.rules_from_levels(leg, [c], min_n=min_n) if kind == "cat" else _seg.rules_from_thresholds(leg, [c])
            if kind == "num" and not rules:
                skipped.append(c)
                why[c] = "fewer than 5 distinct values"
            cands += [(r, c, kind) for r in rules]
    rows, seen = [], set()
    for r, c, kind in cands:
        if r.name in seen:  # tied quantiles give the same rule twice: test it once
            continue
        seen.add(r.name)
        sub = leg[r.fn(leg).fillna(False).astype(bool)]
        if len(sub) >= min_n:
            rows.append({"rule": r.name, "col": c, "kind": kind, **summarize_leg(sub, side, commission, n_sim=5000)})
    out = pd.DataFrame(rows, columns=["rule", "col", "kind", *SUMMARY_FIELDS])
    out["bh_pass"] = _roi.benjamini_hochberg(out["p_null"].fillna(1.0).tolist(), q=0.10)
    out = out.sort_values("p_null", kind="stable").reset_index(drop=True)
    out.attrs.update(n_tests=int(len(out)), skipped=skipped, skipped_reasons=why)
    return out


# --------------------------------------------------------------------------------------------- i. walk-forward
def _bounded_summarize(side):
    """roi.summarize with the same results for small legs but bounded memory for large ones (segments.walk_forward
    draws 10000 x n bootstrap and 20000 x n null cells in one go). The null is the back null: skipped for lays."""
    def f(pl, odds, won, stake=None, commission=0.0, n_boot=10000, run_null=True):
        s = _roi.summarize(pl, odds, won, stake, commission, int(min(n_boot, max(500, 15_000_000 // max(len(pl), 1)))), run_null=False)
        if run_null:
            s.p_null = _null_back(np.asarray(odds, float), s.roi, commission, 20000, 54321) if side != "lay" and len(pl) else NAN
        return s
    return f


def walk_forward_leg(leg: pd.DataFrame, cat_cols, num_cols, commission: float, n_folds: int = 4, side: str | None = None) -> dict:
    if len(leg) < 60:
        return {"skipped": "too few bets", "n": len(leg)}
    side = side or leg.attrs.get("side")
    leg = _prep(leg, num_cols)
    ok = lambda c: c in leg and leg[c].notna().any()  # noqa: E731
    rules = _seg.rules_from_levels(leg, [c for c in cat_cols if ok(c)]) + _seg.rules_from_thresholds(leg, [c for c in num_cols if ok(c)])
    orig, _seg.summarize = _seg.summarize, _bounded_summarize(side)
    try:
        res = _seg.walk_forward(leg, rules, n_folds=n_folds, select_by="roi_ci_lo", commission=commission)
    finally:
        _seg.summarize = orig
    res["n_rules"] = len(rules)
    if side == "lay":  # the walk-forward null is the BACK null: not valid for lays, so it is withheld (None)
        res["p_null_note"] = "null p withheld: the walk-forward null is the back formula"
    return res


# --------------------------------------------------------------------------------------------- j. verdict
VERDICT_COLS = ["rule", "n", "roi", "roi_cl_lo", "p_null", "bh_pass", "clv_mean", "clv_ci_lo",
                "oos_selected", "oos_roi", "positive"]


def verdict(segments_df: pd.DataFrame, wf: dict, min_n: int = 30) -> pd.DataFrame:
    if segments_df is None or len(segments_df) == 0:
        return pd.DataFrame(columns=VERDICT_COLS)
    s = segments_df
    keep = s[s["bh_pass"].astype(bool) | ((s["roi"] > 0) & (s["n"] >= min_n))]
    agg: dict[str, list] = {}
    for p in (wf or {}).get("picks", []):
        if p.get("rule") is not None:
            a = agg.setdefault(p["rule"], [0, 0.0])
            a[0] += p.get("test_n") or 0
            a[1] += p.get("test_pl") or 0.0
    rows = []
    for r in keep.itertuples(index=False):
        a = agg.get(r.rule)
        oos = a[1] / a[0] if a and a[0] > 0 else NAN
        pos = bool(r.bh_pass) and r.roi_cl_lo > 0 and r.clv_ci_lo > 0 and oos > 0  # NaN compares False
        rows.append({"rule": r.rule, "n": r.n, "roi": r.roi, "roi_cl_lo": r.roi_cl_lo, "p_null": r.p_null,
                     "bh_pass": bool(r.bh_pass), "clv_mean": r.clv_mean, "clv_ci_lo": r.clv_ci_lo,
                     "oos_selected": a is not None, "oos_roi": oos, "positive": bool(pos)})
    out = pd.DataFrame(rows, columns=VERDICT_COLS)
    return out.sort_values(["positive", "roi"], ascending=[False, False], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------------- k. scalping
_FIX_LTP = re.compile(r"fx_(back_lay|lay_back)_h(\d+)_s(\d+)")
_FIX_BOOK = re.compile(r"xfx_(back_lay|lay_back)_h(\d+)")
_TPSL = re.compile(r"tp_(back_lay|lay_back)_tp(\d+)_sl(None|\d+)_s(\d+)")
FIXED_COLS = ["basis", "side", "horizon_min", "horizon", "pen", "n", "n_missing", "mean_green", "ci_lo", "ci_hi",
              "median", "pct_profitable", "pct_loss"]
_REASONS = ("tp", "sl", "time", "off", "no_exit")
TPSL_COLS = ["side", "tp", "sl", "pen", "n", "n_missing", "mean_green", "ci_lo", "ci_hi", "pct_profitable",
             *[f"reason_{r}" for r in _REASONS], "reason_other"]


def _green_stats(e: pd.DataFrame, col: str) -> dict:
    v = pd.to_numeric(e[col], errors="coerce")
    ok = v.notna().to_numpy()
    x = v.to_numpy(float)[ok]
    d = {"n": int(len(x)), "n_missing": int(len(e) - len(x))}
    if not len(x):
        return d | dict.fromkeys(("mean_green", "ci_lo", "ci_hi", "median", "pct_profitable", "pct_loss"), NAN)
    lo, hi = cluster_ci(x, e["match_event_id"].to_numpy(dtype=object)[ok])
    return d | {"mean_green": float(x.mean()), "ci_lo": lo, "ci_hi": hi, "median": float(np.median(x)),
                "pct_profitable": float((x > 0).mean()), "pct_loss": float((x < 0).mean())}


def scalping_tables(enriched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    e = _eligible(enriched)
    fixed, tpsl = [], []
    for c in e.columns:
        if m := _FIX_LTP.fullmatch(c):
            basis, side, h, pen = "ltp", m[1], int(m[2]), int(m[3])
        elif m := _FIX_BOOK.fullmatch(c):
            basis, side, h, pen = "book", m[1], int(m[2]), NAN
        elif m := _TPSL.fullmatch(c):
            r = e[c + "_reason"] if c + "_reason" in e else pd.Series(None, index=e.index, dtype=object)
            share = {f"reason_{k}": float((r == k).mean()) if len(e) else NAN for k in _REASONS}
            share["reason_other"] = float((~r.isin(_REASONS)).mean()) if len(e) else NAN
            tpsl.append({"side": m[1], "tp": int(m[2]), "sl": "none" if m[3] == "None" else m[3], "pen": int(m[4]),
                         **{k: v for k, v in _green_stats(e, c).items() if k in TPSL_COLS}, **share})
            continue
        else:
            continue
        fixed.append({"basis": basis, "side": side, "horizon_min": h, "horizon": "to_off" if h == 100000 else str(h),
                      "pen": pen, **_green_stats(e, c)})
    f = pd.DataFrame(fixed, columns=FIXED_COLS)
    f = f.sort_values(["basis", "side", "pen", "horizon_min"], kind="stable").reset_index(drop=True)
    t = pd.DataFrame(tpsl, columns=TPSL_COLS)
    t = t.sort_values(["side", "pen", "tp", "sl"], kind="stable").reset_index(drop=True)
    return f, t


# --------------------------------------------------------------------------------------------- l. matching
def _col(df: pd.DataFrame, c: str, default=None) -> pd.Series:
    return df[c] if c in df else pd.Series(default, index=df.index)


def match_report(enriched: pd.DataFrame, seed: int = 1234) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = enriched
    g = pd.DataFrame({"sport": _col(df, "sport", "(none)").fillna("(none)"),
                      "bf_market": _col(df, "bf_market", "(none)").fillna("(none)"), "status": df["status"]})
    keys = ["sport", "bf_market"]
    t = g.groupby([*keys, "status"], dropna=False).size().rename("n").reset_index()
    t["share"] = t["n"] / t.groupby(keys, dropna=False)["n"].transform("sum")
    r = t.assign(ok=t["n"].where(t["status"] == "ok", 0)).groupby(keys, dropna=False).agg(n=("n", "sum"), ok=("ok", "sum")).reset_index()
    r["status"], r["share"] = "match_rate", r["ok"] / r["n"]
    summary = pd.concat([t, r[[*keys, "status", "n", "share"]]], ignore_index=True)

    n = lambda c: pd.to_numeric(_col(df, c), errors="coerce")  # noqa: E731
    flags = {"ambiguous": _col(df, "match_reason").eq("ambiguous"), "score<85": n("match_score") < 85,
             "|dt|>5min": n("match_dt_min").abs() > 5, "swapped": _col(df, "match_swapped").eq(True)}
    bad = df["status"].ne("ok")
    mask = bad | pd.concat(flags.values(), axis=1).any(axis=1)
    why = pd.Series("", index=df.index)
    why[bad] = "status=" + df.loc[bad, "status"].astype(str)
    for k, f in flags.items():
        why = why.where(~f, why + np.where(why == "", "", "; ") + k)
    keep = ["alert_id", "ts", "sport", "home", "away", "kickoff", "bf_market", "sel_kind", "status", "error",
            "match_reason", "match_score", "match_dt_min", "match_swapped", "bf_event_name"]
    review = df.loc[mask, [c for c in keep if c in df]].assign(why=why[mask])
    spot_cols = ["alert_id", "ts", "text", "home", "away", "kickoff", "bf_market", "sel_kind", "sel_hc",
                 "bf_event_name", "bf_runner", "bf_hc", "bf_off", "match_score"]
    ok = df[df["status"].eq("ok")]
    spot = ok.sample(min(20, len(ok)), random_state=seed)[[c for c in spot_cols if c in df]]
    return summary, review.reset_index(drop=True), spot.reset_index(drop=True)


# --------------------------------------------------------------------------------------------- m. curve
def pl_curve(leg: pd.DataFrame) -> pd.DataFrame:
    d = leg.sort_values("ts", kind="stable")[["ts", "pl"]].reset_index(drop=True)
    d["cum_pl"] = d["pl"].cumsum()
    return d
