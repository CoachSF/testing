"""Join parsed alerts to Betfair markets and compute everything per alert.

Input: a DataFrame of *parsed* alerts with (at least) these columns
    alert_id      unique id (e.g. Telegram message id)
    ts            alert time, pandas Timestamp (UTC)
    sport         Betfair sport name ("Soccer", "Tennis", ...)
    home, away    participant names as written in the alert
    kickoff       pandas Timestamp (UTC) or NaT
    bf_market     Betfair marketType to use (e.g. MATCH_ODDS, OVER_UNDER_25, ASIAN_HANDICAP)
    sel_kind      home | away | draw | over | under | yes | no | team | ...
    sel_team      optional team name for 'team' selections
    sel_hc        optional handicap / line for handicap markets
    ref_odds      optional reference odds quoted in the alert (e.g. Pinnacle after the drop)
Other columns are carried through untouched (useful as strategy features).

Executable order-book columns (ADVANCED/PRO data; "x" prefix = taker on the book; NaN without a book):
    has_book                    the runner has best back / best lay observations
    bb_l{L}, bb_l{L}_size       best back (price a BACKER can take) and its size, L seconds after the alert
    bl_l{L}, bl_l{L}_size       best lay (price a LAYER can take) and its size
    bb_l0_age_min, bl_l0_age_min  staleness of those book observations at L=0
    spread_ticks_l0, tv_l0      ladder ticks bb_l0 -> bl_l0; traded volume at the alert
    bb_close, bl_close          last pre-off book (off - 1 ms)
    xback_pl_l{L}_{ctag}        P/L backing at bb_l{L};  xlay_pl_l{L}_{ctag}  laying at bl_l{L}
    xclose_back_pl_c0, xclose_lay_pl_c0   same at the pre-off close, no commission
    xfx_{side}_h{h}[_reason]    green-up P/L (and exit reason) of a book taker scalp, pre-off alerts only
ctag = commission percent, e.g. c5 (5%), c0, c4p8 (4.8%).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .betfair_files import MarketRecord
from .matching import EventInfo, find_markets, match_event, resolve_runner
from .pricing import book_at, ltp_at, preoff_close, traded_volume_at
from .roi import back_pl, lay_pl, runner_outcome
from .scalping import book_fixed_horizon, event_study, fixed_horizon, tp_sl
from .ticks import tick_diff

MIN = 60_000


@dataclass
class Config:
    commissions: tuple[float, ...] = (0.0, 0.02, 0.05)
    latencies_s: tuple[int, ...] = (0, 60, 120, 300)      # reaction delay after the alert
    max_ltp_age_min: float = 30.0                          # older last-trade => flag as stale
    kickoff_window_min: float = 20.0
    event_offsets_min: tuple[int, ...] = (-120, -60, -30, -15, -5, -2, -1, 0, 1, 2, 5, 10, 15, 30, 60, 120, 240)
    horizons_min: tuple[int, ...] = (1, 2, 5, 10, 15, 30, 60, 120, 240, 100000)  # 100000 = hold to the off
    tp_ticks: tuple[int, ...] = (1, 2, 3, 5)
    sl_ticks: tuple[int | None, ...] = (None, 3, 5)
    tp_max_min: int = 240
    spread_pen: tuple[int, ...] = (0, 1)
    scalp_commission: float = 0.05
    book: bool = True                                      # add the executable order-book columns (x*, bb_*, bl_*)
    extra: dict = field(default_factory=dict)


def _ctag(c: float) -> str:
    """Commission tag: 0.05 -> 'c5', 0 -> 'c0', 0.048 -> 'c4p8' (up to 2 decimals of a percent)."""
    return "c" + f"{round(c * 100, 2):.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _ms(ts) -> int | None:
    if ts is None:
        return None
    try:
        if pd.isna(ts):
            return None
    except (TypeError, ValueError):
        pass
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return int(round(t.timestamp() * 1000))


def _pick_market(markets: dict[str, MarketRecord], ev: EventInfo, mtype: str) -> MarketRecord | None:
    ids = find_markets(ev, mtype)
    if not ids:
        return None
    # if several (rare), prefer the one with most LTP observations
    recs = [markets[i] for i in ids if i in markets]
    return max(recs, key=lambda m: sum(len(r.ltp_t) for r in m.runners.values())) if recs else None


def enrich(alerts: pd.DataFrame, markets: dict[str, MarketRecord], index: dict[str, EventInfo],
           cfg: Config | None = None) -> pd.DataFrame:
    cfg = cfg or Config()
    rows = []
    for a in alerts.itertuples(index=False):
        a = a._asdict()
        out = dict(a)
        t_alert = _ms(a["ts"])
        em = match_event(index, a.get("sport"), a.get("home") or "", a.get("away") or "",
                         kickoff_ms=_ms(a.get("kickoff")), alert_ms=t_alert,
                         window_min=cfg.kickoff_window_min)
        out.update(match_event_id=em.event_id, match_score=em.score, match_home=em.home_score,
                   match_away=em.away_score, match_runner_up=em.runner_up_score, match_swapped=em.swapped,
                   match_dt_min=em.time_diff_min, match_reason=em.reason)
        if em.event_id is None:
            out["status"] = "no_event_match"
            rows.append(out)
            continue
        ev = index[em.event_id]
        out.update(bf_event_name=ev.event_name, bf_country=ev.country_code)
        m = _pick_market(markets, ev, a.get("bf_market"))
        if m is None:
            out["status"] = "no_market"
            rows.append(out)
            continue
        hc = a.get("sel_hc")
        hc = None if hc is None or (isinstance(hc, float) and np.isnan(hc)) else float(hc)
        kind = a.get("sel_kind")
        # if the alert's home/away order was swapped relative to Betfair, swap home/away selections
        if em.swapped and kind in ("home", "away"):
            kind = "away" if kind == "home" else "home"
        r = resolve_runner(m, ev, kind, team=a.get("sel_team"), handicap=hc, sport=a.get("sport"))
        if r is None:
            out["status"] = "no_runner"
            out["bf_market_id"] = m.market_id
            rows.append(out)
            continue
        off = m.off_ms()
        closed = m.final_status == "CLOSED"
        oc = runner_outcome(r, closed)
        st, ip = m.status_at(t_alert)
        out.update(bf_market_id=m.market_id, bf_market_name=m.market_name, bf_runner=r.name,
                   bf_selection_id=r.selection_id, bf_hc=r.handicap, bf_data_level=m.data_level,
                   bf_market_base_rate=m.market_base_rate, bf_off=pd.to_datetime(off, unit="ms", utc=True) if off else pd.NaT,
                   mins_to_off=(off - t_alert) / MIN if off else np.nan,
                   in_play_at_alert=bool(ip) if ip is not None else (off is not None and t_alert >= off),
                   bf_status_at_alert=st, result_status=oc.status, result_known=oc.known,
                   won=oc.win if oc.known else np.nan, void=oc.void if oc.known else np.nan)
        # prices at alert (+latency)
        for L in cfg.latencies_s:
            p = ltp_at(r, t_alert + L * 1000)
            out[f"p_l{L}"] = p.price if p else np.nan
            out[f"p_l{L}_age_min"] = p.age_ms / MIN if p else np.nan
        pc = preoff_close(m, r)
        out["p_close"] = pc.price if pc else np.nan
        p0 = out["p_l0"]
        out["stale_at_alert"] = bool(np.isnan(out["p_l0_age_min"]) or out["p_l0_age_min"] > cfg.max_ltp_age_min)
        # P/L for back and lay at each latency/commission
        if oc.known and not np.isnan(p0):
            for L in cfg.latencies_s:
                pL = out[f"p_l{L}"]
                for c in cfg.commissions:
                    tag = f"l{L}_{_ctag(c)}"
                    out[f"back_pl_{tag}"] = back_pl(pL, oc, c) if not np.isnan(pL) else np.nan
                    out[f"lay_pl_{tag}"] = lay_pl(pL, oc, c) if not np.isnan(pL) else np.nan
            pr = a.get("ref_odds")
            if pr is not None and not (isinstance(pr, float) and np.isnan(pr)):
                out["ref_back_pl"] = back_pl(float(pr), oc, 0.0)  # bookmaker odds: no commission
            if not np.isnan(out["p_close"]):
                out["close_back_pl_c0"] = back_pl(out["p_close"], oc, 0.0)
        if cfg.book:
            bk = {L: book_at(r, t_alert + L * 1000) for L in cfg.latencies_s}
            for L, b in bk.items():
                for k in ("bb", "bl"):
                    out[f"{k}_l{L}"], out[f"{k}_l{L}_size"] = b[k], b[f"{k}_size"]
            b0 = book_at(r, t_alert)
            tv = traded_volume_at(r, t_alert)
            bc = book_at(r, off - 1) if off else {"bb": np.nan, "bl": np.nan}
            both = not (np.isnan(b0["bb"]) or np.isnan(b0["bl"]))
            out.update(has_book=bool(r.bb_t or r.bl_t), bb_l0_age_min=b0["bb_age_ms"] / MIN,
                       bl_l0_age_min=b0["bl_age_ms"] / MIN, tv_l0=np.nan if tv is None else tv,
                       spread_ticks_l0=tick_diff(b0["bb"], b0["bl"]) if both else np.nan,
                       bb_close=bc["bb"], bl_close=bc["bl"])
            if oc.known:
                for L, b in bk.items():
                    for c in cfg.commissions:
                        out[f"xback_pl_l{L}_{_ctag(c)}"] = np.nan if np.isnan(b["bb"]) else back_pl(b["bb"], oc, c)
                        out[f"xlay_pl_l{L}_{_ctag(c)}"] = np.nan if np.isnan(b["bl"]) else lay_pl(b["bl"], oc, c)
                if not np.isnan(bc["bb"]):
                    out["xclose_back_pl_c0"] = back_pl(bc["bb"], oc, 0.0)
                if not np.isnan(bc["bl"]):
                    out["xclose_lay_pl_c0"] = lay_pl(bc["bl"], oc, 0.0)
            if not out["in_play_at_alert"]:
                for h in cfg.horizons_min:
                    for side in ("back_lay", "lay_back"):
                        tr = book_fixed_horizon(r, t_alert, h * MIN, off, side, cfg.scalp_commission)
                        out[f"xfx_{side}_h{h}"] = tr.green_pl
                        out[f"xfx_{side}_h{h}_reason"] = tr.exit_reason
        # event study (relative tick level vs t_alert)
        if not np.isnan(p0) and not out["in_play_at_alert"]:
            es = event_study(r, t_alert, [o * MIN for o in cfg.event_offsets_min], off)
            for o in cfg.event_offsets_min:
                out[f"es_{o}"] = es.get(o * MIN)
            # scalping grid
            for pen in cfg.spread_pen:
                for h in cfg.horizons_min:
                    for side in ("back_lay", "lay_back"):
                        tr = fixed_horizon(r, t_alert, h * MIN, off, side, cfg.scalp_commission, pen, pen)
                        out[f"fx_{side}_h{h}_s{pen}"] = tr.green_pl
                for tp in cfg.tp_ticks:
                    for sl in cfg.sl_ticks:
                        for side in ("back_lay", "lay_back"):
                            tr = tp_sl(r, t_alert, tp, sl, cfg.tp_max_min * MIN, off, side,
                                       cfg.scalp_commission, pen, pen, fill="touch")
                            out[f"tp_{side}_tp{tp}_sl{sl}_s{pen}"] = tr.green_pl
                            out[f"tp_{side}_tp{tp}_sl{sl}_s{pen}_reason"] = tr.exit_reason
        out["status"] = "ok"
        rows.append(out)
    df = pd.DataFrame(rows)
    return df
