"""Strategy discovery on enriched alert/bet tables, with out-of-sample checks.

Expected DataFrame columns:
    ts      - alert time (sortable; pandas Timestamp UTC)
    pl      - P/L per 1 unit stake for the bet (after commission)
    odds    - decimal odds taken
    won     - 1/0 (or fraction for split results)
plus any feature columns used for segmentation, and optionally
    p_close - exchange closing price (for CLV)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .roi import benjamini_hochberg, clv_stats, summarize


def _summ(df: pd.DataFrame, commission: float, run_null: bool = True) -> dict:
    s = summarize(df["pl"].to_numpy(), df["odds"].to_numpy(), df["won"].to_numpy(),
                  commission=commission, n_boot=4000, run_null=run_null).as_dict()
    if "p_close" in df:
        c = clv_stats(df["odds"].to_numpy(), df["p_close"].to_numpy())
        s.update({f"clv_{k}": v for k, v in c.items() if k != "n"})
        s["clv_n"] = c.get("n", 0)
    return s


def segment_table(df: pd.DataFrame, by: str | list[str], min_n: int = 20,
                  commission: float = 0.0) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, dropna=False, observed=True):
        if len(g) < min_n:
            continue
        r = _summ(g.sort_values("ts"), commission)
        r["segment"] = key if isinstance(key, tuple) else (key,)
        r["by"] = tuple(by) if isinstance(by, list) else (by,)
        rows.append(r)
    out = pd.DataFrame(rows)
    if len(out):
        out["bh_pass_q10"] = benjamini_hochberg(out["p_null"].fillna(1.0).tolist(), q=0.10)
        out = out.sort_values("roi", ascending=False)
    return out


def threshold_scan(df: pd.DataFrame, col: str, thresholds: list[float], side: str = "ge",
                   min_n: int = 20, commission: float = 0.0) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        g = df[df[col] >= th] if side == "ge" else df[df[col] <= th]
        if len(g) < min_n:
            continue
        r = _summ(g.sort_values("ts"), commission)
        r.update({"col": col, "side": side, "threshold": th})
        rows.append(r)
    return pd.DataFrame(rows)


@dataclass
class Rule:
    name: str
    fn: Callable[[pd.DataFrame], pd.Series]  # boolean mask


def rules_from_levels(df: pd.DataFrame, cols: list[str], min_n: int = 15) -> list[Rule]:
    rules = []
    for c in cols:
        for lvl, cnt in df[c].value_counts(dropna=True).items():
            if cnt >= min_n:
                rules.append(Rule(f"{c} == {lvl}", (lambda d, c=c, lvl=lvl: d[c] == lvl)))
    return rules


def rules_from_thresholds(df: pd.DataFrame, cols: list[str], quantiles=(0.2, 0.4, 0.6, 0.8)) -> list[Rule]:
    rules = []
    for c in cols:
        vals = df[c].dropna()
        if vals.nunique() < 5:
            continue
        for q in quantiles:
            th = float(vals.quantile(q))
            rules.append(Rule(f"{c} >= {th:.4g}", (lambda d, c=c, th=th: d[c] >= th)))
            rules.append(Rule(f"{c} <= {th:.4g}", (lambda d, c=c, th=th: d[c] <= th)))
    return rules


def walk_forward(df: pd.DataFrame, rules: list[Rule], n_folds: int = 4, min_train_n: int = 25,
                 select_by: str = "roi_ci_lo", commission: float = 0.0) -> dict:
    """Expanding-window walk-forward: for fold k pick the rule that looked best on
    folds < k (by ``select_by``), bet it on fold k. Returns out-of-sample results."""
    d = df.sort_values("ts").reset_index(drop=True)
    edges = np.linspace(0, len(d), n_folds + 1).astype(int)
    oos_parts, picks = [], []
    for k in range(1, n_folds):
        train = d.iloc[: edges[k]]
        test = d.iloc[edges[k]: edges[k + 1]]
        best, best_val = None, -np.inf
        for rule in rules:
            tr = train[rule.fn(train).fillna(False).astype(bool)]
            if len(tr) < min_train_n:
                continue
            s = summarize(tr["pl"].to_numpy(), tr["odds"].to_numpy(), tr["won"].to_numpy(),
                          commission=commission, n_boot=1000, run_null=False)
            val = getattr(s, select_by) if select_by != "clv" else (
                clv_stats(tr["odds"].to_numpy(), tr["p_close"].to_numpy()).get("ev_ci_lo", -np.inf)
                if "p_close" in tr else -np.inf)
            if val is not None and np.isfinite(val) and val > best_val:
                best, best_val = rule, val
        if best is None:
            picks.append({"fold": k, "rule": None})
            continue
        te = test[best.fn(test).fillna(False).astype(bool)]
        picks.append({"fold": k, "rule": best.name, "train_score": float(best_val), "test_n": int(len(te)),
                      "test_pl": float(te["pl"].sum()) if len(te) else 0.0,
                      "test_roi": float(te["pl"].sum() / len(te)) if len(te) else None})
        oos_parts.append(te)
    oos = pd.concat(oos_parts) if oos_parts else d.iloc[0:0]
    res = {"picks": picks, "oos_n": int(len(oos))}
    if len(oos):
        res["oos"] = summarize(oos["pl"].to_numpy(), oos["odds"].to_numpy(), oos["won"].to_numpy(),
                               commission=commission).as_dict()
    # baseline: bet everything in the same OOS folds
    base = d.iloc[edges[1]:]
    res["baseline_all_oos"] = summarize(base["pl"].to_numpy(), base["odds"].to_numpy(), base["won"].to_numpy(),
                                        commission=commission).as_dict()
    return res
