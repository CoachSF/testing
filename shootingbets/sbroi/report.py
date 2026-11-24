"""Write the CSV tables and ONE self-contained report.html (inline CSS / SVG / JS, no external references)."""
from __future__ import annotations

import html
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import analysis as A

MEL = ZoneInfo("Australia/Melbourne")
CAT_COLS = ["sport", "league", "bf_country", "bf_market", "sel_kind", "odds_band", "mins_to_off_band", "spread_band",
            "liq_band", "drop_band", "hour_utc", "dow", "is_repeat"]
NUM_COLS = ["drop_pct", "ref_odds", "gap_book", "gap_ltp", "mins_to_off", "bb_l0", "bb_l0_size", "spread_ticks_l0", "repeat_n"]
LEG_LABEL = {("back", "book"): "Back at best back", ("lay", "book"): "Lay at best lay", ("back", "ltp"): "Back at last traded price",
             ("lay", "ltp"): "Lay at last traded price", ("back", "ref"): "Back at Pinnacle quote",
             ("back", "close"): "Back at Betfair close (last traded)"}

# --------------------------------------------------------------------------------------------- formatting
esc = lambda x: html.escape(str(x))  # noqa: E731
_na = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))  # noqa: E731


def pct(v, d=1):
    return "n/a" if _na(v) else f"{v * 100:.{d}f}%"


def num(v, d=2):
    return "n/a" if _na(v) else f"{v:.{d}f}"


def intf(v):
    return "n/a" if _na(v) else f"{int(round(v))}"


def pval(v):
    return "n/a" if _na(v) else f"{v:.3f}"


def ci(lo, hi, f=pct):
    return "n/a" if _na(lo) or _na(hi) else f"[{f(lo)}, {f(hi)}]"


def ts_both(t) -> str:
    if _na(t):
        return "n/a"
    t = pd.Timestamp(t)
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    m = t.tz_convert(MEL)
    return f"{t:%Y-%m-%d %H:%M} UTC / {m:%Y-%m-%d %H:%M} {m.tzname()} Melbourne"


def table(head, rows, cls="data", left=1) -> str:
    """``left`` = number of leading text columns (left-aligned); the rest are numbers (right-aligned)."""
    k = lambda i: ' class="t"' if i < left else ""  # noqa: E731
    th = "".join(f"<th{k(i)}>{esc(h)}</th>" for i, h in enumerate(head))
    body = "".join("<tr>" + "".join(f"<td{k(i)}>{c}</td>" for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if math.isnan(o) or math.isinf(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return o


# --------------------------------------------------------------------------------------------- chart toolkit
W = 720


def _nice_ticks(lo, hi, n=5):
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw - 1e-12)
    t0, t1 = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    return [round(t0 + i * step, 10) for i in range(int(round((t1 - t0) / step)) + 1)], step


def _tick_text(v, step, unit=""):
    d = 0
    while abs(step * 10 ** d - round(step * 10 ** d)) > 1e-9 and d < 6:
        d += 1
    s = f"{v:.{d}f}"
    return ("0" if float(s) == 0 else s) + unit


def _time_ticks(t0, t1, maxn=7):
    span_h = (t1 - t0) / 3.6e6
    step_h = next((s for s in (1 / 12, 1 / 4, 1 / 2, 1, 2, 3, 6, 12, 24, 48, 168, 336, 720) if span_h / s <= maxn - 1), 720)
    step = step_h * 3.6e6
    t, out = math.ceil(t0 / step) * step, []
    while t <= t1 + 1:
        out.append(t)
        t += step
    return out if len(out) >= 2 else [t0, t1]


def _y_axis(lo, hi, x0, x1, y0, y1, unit=""):
    ticks, step = _nice_ticks(lo, hi)
    ymin, ymax = ticks[0], ticks[-1]
    fy = lambda v: y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)  # noqa: E731
    parts = []
    for v in ticks:
        parts.append(f'<line class="{"base" if v == 0 else "grid"}" x1="{x0}" x2="{x1}" y1="{fy(v):.1f}" y2="{fy(v):.1f}"/>')
        parts.append(f'<text class="tick" x="{x0 - 8}" y="{fy(v) + 4:.1f}" text-anchor="end">{esc(_tick_text(v, step, unit))}</text>')
    return fy, "".join(parts)


def _spread_labels(ys, lo, hi, gap=15.0):
    """Move label y positions apart (min ``gap``) and clamp them to [lo, hi]."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    prev = lo - gap
    for i in order:
        out[i] = max(out[i], prev + gap)
        prev = out[i]
    over = out[order[-1]] - hi if order else 0
    if over > 0:
        for i in order:
            out[i] -= over
    return out


def _legend(items) -> str:
    return '<div class="legend">' + "".join(f'<span class="key {c}"><i class="sw"></i>{esc(n)}</span>' for c, n in items) + "</div>"


def _figure(title, svg, legend, data, table_html, note="") -> str:
    js = json.dumps(_clean(data), separators=(",", ":")).replace("<", "\\u003c")
    return (f'<figure class="chart"><figcaption>{esc(title)}</figcaption>{legend}<div class="plot">{svg}<div class="tip"></div></div>'
            f'<script type="application/json" class="chart-data">{js}</script>'
            f'{f"<p class=note>{esc(note)}</p>" if note else ""}<details><summary>Table view</summary>{table_html}</details></figure>')


def cum_chart(series, title):
    """series: [{name, short, cls, t (ms, sorted), y (cumulative)}]."""
    H, x0, x1, y0, y1 = 340, 76, W - 150, 14, 288
    t0, t1 = min(s["t"][0] for s in series), max(s["t"][-1] for s in series)
    if t1 - t0 < 60_000:
        t0, t1 = t0 - 1_800_000, t1 + 1_800_000
    allv = np.concatenate([s["y"] for s in series])
    fy, ya = _y_axis(float(allv.min()), float(allv.max()), x0, x1, y0, y1)
    fx = lambda t: x0 + (t - t0) / (t1 - t0) * (x1 - x0)  # noqa: E731
    g = [ya]
    for t in _time_ticks(t0, t1):
        x = fx(t)
        u = pd.Timestamp(t, unit="ms", tz="UTC")
        m = u.tz_convert(MEL)
        g.append(f'<text class="tick" x="{x:.1f}" y="{y1 + 16}" text-anchor="middle">{u:%m-%d %H:%M}</text>'
                 f'<text class="tick" x="{x:.1f}" y="{y1 + 30}" text-anchor="middle">{m:%m-%d %H:%M}</text>')
    g.append(f'<text class="ax" x="{x0 - 8}" y="{y1 + 16}" text-anchor="end">UTC</text>'
             f'<text class="ax" x="{x0 - 8}" y="{y1 + 30}" text-anchor="end">Melbourne</text>')
    ends = [fy(s["y"][-1]) for s in series]
    lab_y = _spread_labels(ends, y0 + 6, y1 - 6)
    for s, ye, yl in zip(series, ends, lab_y):
        pts = " ".join(f"{'M' if i == 0 else 'L'}{fx(t):.1f},{fy(v):.1f}" for i, (t, v) in enumerate(zip(s["t"], s["y"])))
        xe = fx(s["t"][-1])
        g.append(f'<g class="{s["cls"]}"><path class="line" d="{pts}"/><circle class="ring" cx="{xe:.1f}" cy="{ye:.1f}" r="5"/>'
                 f'<circle class="dot" cx="{xe:.1f}" cy="{ye:.1f}" r="4"/><circle class="dot" cx="{x1 + 14}" cy="{yl:.1f}" r="3"/></g>'
                 f'<text class="lbl" x="{x1 + 22}" y="{yl + 4:.1f}">{esc(s["short"])} {s["y"][-1]:+.1f}</text>')
    n_g = 160
    grid = np.linspace(t0, t1, n_g)
    cols = []
    for s in series:
        idx = np.searchsorted(np.asarray(s["t"]), grid, side="right") - 1
        cols.append([None if i < 0 else float(s["y"][i]) for i in idx])
    data = {"kind": "line", "x": [round(fx(t), 1) for t in grid], "lab": [ts_both(pd.Timestamp(t, unit="ms", tz="UTC")) for t in grid],
            "series": [{"name": s["name"], "cls": s["cls"], "txt": ["no bets yet" if v is None else f"{v:+.2f} units" for v in c]}
                       for s, c in zip(series, cols)]}
    svg = (f'<svg class="c" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(g)}'
           f'<line class="xhair" y1="{y0}" y2="{y1}" x1="0" x2="0" style="display:none"/>'
           f'<rect class="hit" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/></svg>')
    pick = np.unique(np.linspace(0, n_g - 1, 20).astype(int))
    rows = [[esc(data["lab"][i])] + [("n/a" if c[i] is None else f"{c[i]:+.2f}") for c in cols] for i in pick]
    tbl = table(["Time (UTC / Melbourne)"] + [s["name"] for s in series], rows)
    return _figure(title, svg, _legend([(s["cls"], s["name"]) for s in series]), data, tbl,
                   "Units of backer's stake, cumulative in alert-time order. Table view samples 20 points; pl_curve.csv has every bet.")


def es_chart(es, title):
    H, x0, x1, y0, y1 = 330, 56, W - 16, 28, 282
    k = len(es)
    off = es["offset_min"].to_numpy(float)
    m, lo, hi = (es[c].to_numpy(float) for c in ("mean_ticks", "ci_lo", "ci_hi"))
    vals = np.concatenate([m, lo, hi])
    vals = vals[np.isfinite(vals)]
    fy, ya = _y_axis(float(vals.min()), float(vals.max()), x0, x1, y0, y1)
    fx = lambda i: x0 + (i + 0.5) / k * (x1 - x0)  # noqa: E731
    g = [ya]
    for i, o in enumerate(off):
        g.append(f'<text class="tick" x="{fx(i):.1f}" y="{y1 + 16}" text-anchor="middle">{int(o):+d}</text>' if o else
                 f'<text class="tick" x="{fx(i):.1f}" y="{y1 + 16}" text-anchor="middle">0</text>')
    g.append(f'<text class="ax" x="{(x0 + x1) / 2:.0f}" y="{y1 + 36}" text-anchor="middle">minutes relative to the alert (UTC clock; offsets, not timestamps)</text>')
    xa = x0 + (float(np.interp(0, off, np.arange(k))) + 0.5) / k * (x1 - x0)
    g.append(f'<line class="grid" x1="{xa:.1f}" x2="{xa:.1f}" y1="{y0}" y2="{y1}"/><text class="ax" x="{xa:.1f}" y="{y0 - 8}" text-anchor="middle">alert</text>')
    runs, cur = [], []
    for i in range(k):
        if np.isfinite(lo[i]) and np.isfinite(hi[i]):
            cur.append(i)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    band = "".join('<path class="band" d="M' + " L".join(f"{fx(i):.1f},{fy(hi[i]):.1f}" for i in r) + " L" +
                   " L".join(f"{fx(i):.1f},{fy(lo[i]):.1f}" for i in reversed(r)) + ' Z"/>' for r in runs)
    line, first = "", True
    for i in range(k):
        if np.isfinite(m[i]):
            line += f"{'M' if first else 'L'}{fx(i):.1f},{fy(m[i]):.1f}"
            first = False
        else:
            first = True
    dots = "".join(f'<circle class="dot" cx="{fx(i):.1f}" cy="{fy(m[i]):.1f}" r="3"/>' for i in range(k) if np.isfinite(m[i]))
    g.append(f'<g class="s1">{band}<path class="line" d="{line}"/>{dots}</g>')
    data = {"kind": "line", "x": [round(fx(i), 1) for i in range(k)],
            "lab": [("alert time" if o == 0 else f"{int(o):+d} min vs the alert") for o in off],
            "sub": [f"n={int(r.n)}; 95% CI {num(r.ci_lo)} to {num(r.ci_hi)} ticks" for r in es.itertuples()],
            "series": [{"name": "Mean tick move", "cls": "s1", "txt": ["n/a" if _na(v) else f"{v:+.2f} ticks" for v in m]}]}
    svg = (f'<svg class="c" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(g)}'
           f'<line class="xhair" y1="{y0}" y2="{y1}" x1="0" x2="0" style="display:none"/>'
           f'<rect class="hit" x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}"/></svg>')
    tbl = es_table(es)
    return _figure(title, svg, "", data, tbl, "Negative = the price shortened relative to the price at the alert. Band = 95% CI, bootstrap by event.")


def es_table(es):
    rows = [[intf(r.offset_min), intf(r.n), num(r.mean_ticks), num(r.median_ticks), ci(r.ci_lo, r.ci_hi, num), pct(r.pct_shorter, 0), pct(r.pct_longer, 0)]
            for r in es.itertuples()]
    return table(["Offset (min)", "n", "Mean ticks", "Median ticks", "95% CI (by event)", "Shorter", "Longer"], rows)


def bar_chart(fx_df, title, basis_note):
    H, x0, x1, y0, y1 = 330, 56, W - 16, 16, 280
    hs = sorted(fx_df["horizon_min"].unique())
    k, slot = len(hs), (x1 - x0) / len(hs)
    names = {"back_lay": ("Back then lay", "s1"), "lay_back": ("Lay then back", "s2")}
    cell = {(r.side, r.horizon_min): r for r in fx_df.itertuples()}
    lim = 100.0  # drawn axis limit, % of the opening stake: one thin-book alert (e.g. a lay at 100) must not flatten the chart
    cl = lambda v: max(-lim, min(lim, v * 100))  # noqa: E731
    vals = [cl(v) for r in cell.values() for v in (r.mean_green, r.ci_lo, r.ci_hi) if not _na(v)]
    clipped = any(abs(v * 100) > lim for r in cell.values() for v in (r.mean_green, r.ci_lo, r.ci_hi) if not _na(v))
    fy, ya = _y_axis(min(vals, default=0.0), max(vals, default=1.0), x0, x1, y0, y1, "%")
    bw = min(24.0, (slot - 10) / 2 - 1)
    g, hits, bars = [ya], [], []
    for gi, h in enumerate(hs):
        cx = x0 + (gi + 0.5) * slot
        g.append(f'<text class="tick" x="{cx:.1f}" y="{y1 + 16}" text-anchor="middle">{"to off" if h == 100000 else int(h)}</text>')
        for si, side in enumerate(("back_lay", "lay_back")):
            bx = cx - bw - 1 if si == 0 else cx + 1
            r = cell.get((side, h))
            lab, (nm, cls) = ("to the off" if h == 100000 else f"{int(h)} min"), names[side]
            if r is None or _na(r.mean_green):
                txt = "n/a"
            else:
                v = r.mean_green * 100
                yt, yb = fy(cl(r.mean_green)), fy(0)
                rr = min(4.0, abs(yb - yt), bw / 2)
                if abs(yb - yt) >= 0.5:
                    d = (f"M{bx:.1f},{yb:.1f}V{yt + rr:.1f}Q{bx:.1f},{yt:.1f} {bx + rr:.1f},{yt:.1f}H{bx + bw - rr:.1f}Q{bx + bw:.1f},{yt:.1f} {bx + bw:.1f},{yt + rr:.1f}V{yb:.1f}Z"
                         if yt < yb else
                         f"M{bx:.1f},{yb:.1f}V{yt - rr:.1f}Q{bx:.1f},{yt:.1f} {bx + rr:.1f},{yt:.1f}H{bx + bw - rr:.1f}Q{bx + bw:.1f},{yt:.1f} {bx + bw:.1f},{yt - rr:.1f}V{yb:.1f}Z")
                    g.append(f'<g class="{cls}"><path class="bar" d="{d}"/></g>')
                if not (_na(r.ci_lo) or _na(r.ci_hi)):
                    wx = bx + bw / 2
                    g.append(f'<path class="whisk" d="M{wx:.1f},{fy(cl(r.ci_lo)):.1f}V{fy(cl(r.ci_hi)):.1f}'
                             f'M{wx - 3:.1f},{fy(cl(r.ci_lo)):.1f}H{wx + 3:.1f}M{wx - 3:.1f},{fy(cl(r.ci_hi)):.1f}H{wx + 3:.1f}"/>')
                txt = f"{v:+.2f}% (95% CI {ci(r.ci_lo, r.ci_hi, lambda q: f'{q * 100:+.2f}%')}; n={int(r.n)})"
            hx0, hx1 = (cx - slot / 2, cx) if si == 0 else (cx, cx + slot / 2)
            hits.append(f'<rect class="hit" data-i="{len(bars)}" x="{hx0:.1f}" y="{y0}" width="{hx1 - hx0:.1f}" height="{y1 - y0}"/>')
            bars.append({"x": round(bx + bw / 2, 1), "lab": f"Horizon {lab}", "name": nm, "cls": cls, "txt": txt})
    g.append(f'<text class="ax" x="{(x0 + x1) / 2:.0f}" y="{y1 + 36}" text-anchor="middle">horizon after the alert (minutes)</text>')
    svg = f'<svg class="c" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">{"".join(g)}{"".join(hits)}</svg>'
    rows = [[("to_off" if h == 100000 else intf(h))] + [
        (lambda r: "n/a" if r is None else f"{pct(r.mean_green, 2)} {ci(r.ci_lo, r.ci_hi, lambda q: pct(q, 2))}")(cell.get((s, h)))
        for s in ("back_lay", "lay_back")] for h in hs]
    tbl = table(["Horizon (min)", "Back then lay: mean [95% CI]", "Lay then back: mean [95% CI]"], rows)
    if clipped:
        basis_note += " Bars and whiskers beyond +/-100% of stake are drawn clipped at the axis edge; tooltips and the table give the true values."
    return _figure(title, svg, _legend([("s1", "Back then lay"), ("s2", "Lay then back")]), {"kind": "bar", "bars": bars}, tbl, basis_note)


# --------------------------------------------------------------------------------------------- page assets
CSS = """
.viz-root{--surface:#fcfcfb;--page:#f9f9f7;--text:#0b0b0b;--text-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--series-1:#2a78d6;--series-2:#eb6834;--series-3:#1baf7a;--border:rgba(11,11,11,0.10);color-scheme:light}
@media (prefers-color-scheme: dark){:root:where(:not([data-theme="light"])) .viz-root{--surface:#1a1a19;--page:#0d0d0d;--text:#f0efec;--text-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--border:rgba(255,255,255,0.10);color-scheme:dark}}
:root[data-theme="dark"] .viz-root{--surface:#1a1a19;--page:#0d0d0d;--text:#f0efec;--text-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--series-1:#3987e5;--series-2:#d95926;--series-3:#199e70;--border:rgba(255,255,255,0.10);color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--text);font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif}
main{max-width:1000px;margin:0 auto;padding:16px}
h1{font-size:24px;margin:8px 0 4px}h2{font-size:19px;margin:34px 0 10px;padding-top:8px;border-top:1px solid var(--border)}h3{font-size:16px;margin:20px 0 6px}
p{margin:8px 0}.sub,.note{color:var(--text-2);font-size:13px}.muted{color:var(--muted)}
a{color:var(--text)}
.banner{background:var(--surface);border:3px solid var(--series-2);border-radius:8px;padding:14px 16px;margin:8px 0 16px;font-size:20px;font-weight:700}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:12px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.tile .k{font-size:12px;color:var(--text-2)}.tile .v{font-size:24px;font-weight:650;margin:2px 0}.tile .s{font-size:12px;color:var(--text-2)}
.scroll{overflow-x:auto;margin:8px 0}
table.data{border-collapse:collapse;width:100%;font-size:13px;background:var(--surface);border:1px solid var(--border)}
table.data th,table.data td{padding:5px 8px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}
table.data .t{text-align:left}
table.data th{font-weight:600;color:var(--text-2);background:var(--surface)}
table.pos td{font-weight:600}
figure.chart{margin:14px 0;background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px}
figure.chart figcaption{font-weight:600;margin-bottom:6px}
.legend{display:flex;flex-wrap:wrap;gap:4px 16px;margin:2px 0 6px;font-size:13px;color:var(--text)}
.key .sw{display:inline-block;width:10px;height:10px;border-radius:2px;background:var(--c);margin-right:6px}
.plot{position:relative}
svg.c{display:block;width:100%;max-width:720px;height:auto;touch-action:pan-y}
.s1{--c:var(--series-1)}.s2{--c:var(--series-2)}.s3{--c:var(--series-3)}
.grid{stroke:var(--grid);stroke-width:1;fill:none}.base{stroke:var(--axis);stroke-width:1;fill:none}
.tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}.ax{fill:var(--text-2);font-size:11px}.lbl{fill:var(--text);font-size:12px}
.line{fill:none;stroke:var(--c);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.dot{fill:var(--c)}.ring{fill:var(--surface)}.band{fill:var(--c);fill-opacity:.1;stroke:none}.bar{fill:var(--c)}
.whisk{fill:none;stroke:var(--text-2);stroke-width:1}
.xhair{stroke:var(--axis);stroke-width:1}.hit{fill:transparent}
.tip{position:absolute;top:6px;left:0;display:none;pointer-events:none;z-index:2;background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:6px 8px;font-size:12px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.15);transform:translateX(10px)}
.tip.flip{transform:translateX(calc(-100% - 10px))}.tip div{line-height:1.5}.tip .sw{display:inline-block;width:8px;height:8px;border-radius:2px;background:var(--c);margin-right:5px}
details{margin:8px 0}summary{cursor:pointer;color:var(--text-2);font-size:13px}
.verdict{background:var(--surface);border:1px solid var(--border);border-left:4px solid var(--series-1);border-radius:6px;padding:10px 12px;margin:10px 0}
@media (max-width:600px){h1{font-size:20px}.tile .v{font-size:20px}}
"""

JS = """
(function(){
var figs=document.querySelectorAll('figure.chart');
Array.prototype.forEach.call(figs,function(fig){
  var js=fig.querySelector('script.chart-data'),svg=fig.querySelector('svg'),tip=fig.querySelector('.tip');
  if(!js||!svg||!tip)return;
  var D=JSON.parse(js.textContent),xh=svg.querySelector('.xhair'),vb=svg.viewBox.baseVal;
  function row(cls,text){var d=document.createElement('div');if(cls){d.className=cls;var i=document.createElement('i');i.className='sw';d.appendChild(i);}d.appendChild(document.createTextNode(text));return d;}
  function show(rows,vx){var r=svg.getBoundingClientRect(),left=vx*r.width/vb.width;tip.textContent='';rows.forEach(function(x){tip.appendChild(row(x[0],x[1]));});
    tip.style.display='block';tip.style.left=left+'px';tip.classList.toggle('flip',left>r.width*0.6);}
  function hide(){tip.style.display='none';if(xh)xh.style.display='none';}
  if(D.kind==='line'){
    var hit=svg.querySelector('.hit');
    hit.addEventListener('pointermove',function(e){
      var r=svg.getBoundingClientRect(),vx=(e.clientX-r.left)*vb.width/r.width,k=0,b=1e9;
      for(var i=0;i<D.x.length;i++){var d=Math.abs(D.x[i]-vx);if(d<b){b=d;k=i;}}
      xh.setAttribute('x1',D.x[k]);xh.setAttribute('x2',D.x[k]);xh.style.display='';
      var rows=[[null,D.lab[k]]];D.series.forEach(function(s){rows.push([s.cls,s.name+': '+s.txt[k]]);});
      if(D.sub&&D.sub[k])rows.push([null,D.sub[k]]);
      show(rows,D.x[k]);});
    hit.addEventListener('pointerleave',hide);
  }else{
    Array.prototype.forEach.call(svg.querySelectorAll('rect.hit'),function(h){
      function go(){var b=D.bars[+h.getAttribute('data-i')];show([[null,b.lab],[b.cls,b.name+': '+b.txt]],b.x);}
      h.addEventListener('pointerenter',go);h.addEventListener('pointermove',go);h.addEventListener('pointerleave',hide);});
  }
});
})();
"""

METHOD = """
<ul>
<li>Prices are the best back / best lay actually on offer in the Betfair PRO order-book data at the alert second, and after 60, 120 and 300 seconds of reaction delay. The last traded price (LTP) is shown for comparison only: it is not a price you can necessarily take.</li>
<li>Commission is charged on net winnings only (the commission shown in each row); the owner's real rate is the 4.8% case. P/L is per 1 unit of the backer's stake for backs and lays alike; for lays the return on liability (ROL = P/L divided by sum of odds minus 1) is shown as well. The Pinnacle quote row carries no commission.</li>
<li>Void results are excluded from every ROI, strike rate and count; every excluded alert is counted by reason in headline.csv (column excluded). Alerts placed in-play are excluded from the pre-off legs.</li>
<li>Nothing observed after the alert time selects or prices an entry. Alerts on the same match are not independent, so every ROI and CLV interval is given i.i.d. and clustered by event (bootstrap over whole events).</li>
<li>The null p-value is the chance of an ROI at least this good if every selection won with probability 1/odds at the price taken (a fair price), under the same commission.</li>
<li>Scalping assumes taking the book on both legs (back at the best back then lay at the best lay, or the reverse) and says nothing about queue position or fills for resting orders; the take-profit / stop-loss grid is simulated on last-traded prices.</li>
<li>Small samples make realised ROI very noisy. Closing-line value (CLV: price taken against the last pre-off traded price) is the lower-noise measure of whether the price taken was good.</li>
<li>Segments: every rule tested is counted; Benjamini-Hochberg (q = 0.10) is applied across the whole table. Segment cut-points (quantile thresholds, quartile bins) are computed on the full leg, a mild look-ahead on the cut-points only. A rule counts as positive only if it survives BH, its clustered ROI interval and its CLV interval are above zero, and it earned a positive ROI in the walk-forward folds where it was picked.</li>
</ul>
"""


def _page(title, label, body) -> str:
    banner = f'<div class="banner" role="alert">{esc(label)}</div>' if label else ""
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{CSS}</style></head><body class="viz-root"><main>{banner}{body}</main>'
            f'<script>{JS}</script></body></html>')


# --------------------------------------------------------------------------------------------- the report
def _leg(df, side, price, lat, comm):
    if A.leg_missing(df, side, price, lat, comm):
        e = pd.DataFrame(columns=A.LEG_COLS)
        e.attrs.update(excluded={}, side=side, price=price)
        return e
    return A.leg_frame(df, side, price, lat, comm)


def _tile(k, v, s=""):
    return f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{v}</div><div class="s">{s}</div></div>'


def _excl(js):
    d = {k: v for k, v in json.loads(js or "{}").items() if v}
    return "; ".join(f"{k} {v}" for k, v in d.items()) or "-"


def _hl_rows(h):
    order = {"book": 0, "ltp": 1, "ref": 2, "close": 3}
    h = h.assign(_o=h["price"].map(order), _s=h["side"].map({"back": 0, "lay": 1})).sort_values(["_o", "_s", "latency_s", "commission"], kind="stable")
    return [[esc(LEG_LABEL[(r.side, r.price)]), intf(r.latency_s), pct(r.commission), intf(r.n), intf(r.n_events), pct(r.roi), ci(r.roi_ci_lo, r.roi_ci_hi),
             ci(r.roi_cl_lo, r.roi_cl_hi), pval(r.p_null), pct(r.strike_rate), num(r.avg_odds), num(r.max_drawdown), pct(r.rol),
             esc(r.note or _excl(r.excluded))] for r in h.itertuples()]


HL_HEAD = ["Leg", "Delay (s)", "Commission", "n", "Events", "ROI", "ROI 95% CI (i.i.d.)", "ROI 95% CI (by event)", "Null p",
           "Strike", "Avg odds", "Max drawdown (units)", "ROL (lays)", "Excluded / note"]


def build_report(enriched: pd.DataFrame, out_dir, commission: float = A.REAL_COMMISSION, label: str | None = None,
                 n_alerts_input: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if "status" not in enriched:
        raise ValueError("enriched frame has no 'status' column")
    df = A.add_features(enriched)
    n_in = int(n_alerts_input) if n_alerts_input is not None else len(df)
    n_ok = int(df["status"].eq("ok").sum())
    comms = tuple(sorted({0.0, 0.02, 0.048, 0.05, float(commission)}))
    cpct = f"{commission * 100:g}%"

    summary, review, spot = A.match_report(df)
    hl = A.headline(df, comms)
    es = A.event_study_table(df)
    fixed, tpsl = A.scalping_tables(df)
    has_book = any(c in df and pd.to_numeric(df[c], errors="coerce").notna().any() for c in ("bb_l0", "bl_l0"))
    basis = "book" if has_book else "ltp"
    cats, nums = [c for c in CAT_COLS if c in df], [c for c in NUM_COLS if c in df]
    legs, seg, wf, ver = {}, {}, {}, {}
    for side in ("back", "lay"):
        legs[side] = _leg(df, side, basis, 0, commission)
        seg[side] = A.scan_segments(legs[side], side, commission, cats, nums)
        wf[side] = A.walk_forward_leg(legs[side], cats, nums, commission, side=side)
        ver[side] = A.verdict(seg[side], wf[side])
    ref_leg = _leg(df, "back", "ref", 0, 0.0)
    series_def = [("back", legs["back"], "s1", f"Back at {'best back' if basis == 'book' else 'last traded price'} ({cpct})", "Back"),
                  ("lay", legs["lay"], "s2", f"Lay at {'best lay' if basis == 'book' else 'last traded price'} ({cpct})", "Lay"),
                  ("ref", ref_leg, "s3", "Back at Pinnacle quote", "Pinnacle")]
    curves = [(key, name, short, cls, A.pl_curve(leg)) for key, leg, cls, name, short in series_def if len(leg)]

    # ---- CSVs / JSON
    def put(name, frame):
        frame.to_csv(out / name, index=False)

    put("enriched.csv", df)
    put("match_summary.csv", summary)
    put("match_review.csv", review)
    put("match_spotcheck.csv", spot)
    put("headline.csv", hl)
    put("event_study.csv", es)
    for side in ("back", "lay"):
        put(f"segments_{side}_book.csv", seg[side].assign(price_basis=basis))
    (out / "walk_forward.json").write_text(json.dumps(_clean({"price_basis": basis, "back": wf["back"], "lay": wf["lay"]}), indent=1), encoding="utf-8")
    vall = pd.concat([ver[s].assign(side=s) for s in ("back", "lay")], ignore_index=True)
    vall = vall[["side", *A.VERDICT_COLS]] if len(vall) else pd.DataFrame(columns=["side", *A.VERDICT_COLS])
    put("verdict.csv", vall)
    put("scalping_fixed.csv", fixed)
    put("scalping_tpsl.csv", tpsl)
    cur_all = (pd.concat([c[4].assign(series=c[1]) for c in curves], ignore_index=True)[["series", "ts", "pl", "cum_pl"]]
               if curves else pd.DataFrame(columns=["series", "ts", "pl", "cum_pl"]))
    put("pl_curve.csv", cur_all)

    # ---- numbers
    hrow = hl[(hl.side == "back") & (hl.price == basis) & (hl.latency_s == 0) & np.isclose(hl.commission, commission)].iloc[0]
    positives = [(s, r) for s in ("back", "lay") for r in ver[s].itertuples() if r.positive]
    ts_all = pd.to_datetime(df["ts"], utc=True) if "ts" in df and len(df) else pd.Series([], dtype="datetime64[us, UTC]")
    t_first, t_last = (ts_all.min(), ts_all.max()) if len(ts_all) else (pd.NaT, pd.NaT)
    n_tests = {s: int(seg[s].attrs.get("n_tests", 0)) for s in seg}
    key = {"label": label, "n_alerts_input": n_in, "n_ok": n_ok, "match_rate": n_ok / n_in if n_in else None,
           "status_counts": {str(k): int(v) for k, v in df["status"].value_counts(dropna=False).items()},
           "commission": commission, "segment_price_basis": basis, "headline_leg": f"{LEG_LABEL[('back', basis)]}, delay 0 s, commission {cpct}",
           "headline_n": int(hrow.n), "headline_n_events": int(hrow.n_events), "headline_roi": hrow.roi,
           "headline_roi_ci_iid": [hrow.roi_ci_lo, hrow.roi_ci_hi], "headline_roi_ci_clustered": [hrow.roi_cl_lo, hrow.roi_cl_hi],
           "headline_p_null": hrow.p_null, "headline_clv_mean": hrow.clv_mean, "headline_clv_ci": [hrow.clv_ci_lo, hrow.clv_ci_hi],
           "headline_clv_n": int(hrow.clv_n), "first_alert_utc": None if pd.isna(t_first) else t_first.isoformat(),
           "last_alert_utc": None if pd.isna(t_last) else t_last.isoformat(), "n_review_rows": int(len(review)),
           "n_rules_tested": n_tests, "n_bh_pass": {s: int(seg[s]["bh_pass"].sum()) for s in seg},
           "n_rules_positive": len(positives), "any_positive_rule": bool(positives),
           "positive_rules": [f"{s}: {r.rule}" for s, r in positives], "features_skipped": df.attrs.get("features_skipped", [])}
    (out / "key_numbers.json").write_text(json.dumps(_clean(key), indent=1), encoding="utf-8")

    # ---- html
    nothing = n_ok == 0
    nothing_p = '<p class="note">Nothing to analyse: no alert was matched to a Betfair runner with a result.</p>'
    b = []
    # 1 summary
    b.append(f"<h1>ShootingBets alerts vs Betfair Exchange</h1><p class='sub'>Generated {esc(ts_both(pd.Timestamp.now(tz='UTC')))}</p>")
    b.append("<h2>1. Summary</h2>")
    tiles = [_tile("Alerts in", intf(n_in)), _tile("Matched ok", intf(n_ok), pct(key["match_rate"]) + " of alerts"),
             _tile("Bets in the headline leg", intf(hrow.n), esc(f"{hrow.n_events:.0f} events" if hrow.n else "no bets after exclusions")),
             _tile("Headline ROI", pct(hrow.roi), f"clustered 95% CI {ci(hrow.roi_cl_lo, hrow.roi_cl_hi)}"),
             _tile("Mean CLV vs close", pct(hrow.clv_mean), f"95% CI {ci(hrow.clv_ci_lo, hrow.clv_ci_hi)}; n={intf(hrow.clv_n)}"),
             _tile("Alert date span", "" if pd.isna(t_first) else f"{t_first:%Y-%m-%d} to {t_last:%Y-%m-%d}",
                   esc(ts_both(t_first)) + "<br>to " + esc(ts_both(t_last)))]
    b.append(f'<div class="tiles">{"".join(tiles)}</div>')
    b.append(f"<p class='sub'>Headline leg: {esc(key['headline_leg'])}.</p>")
    if nothing:
        b.append(f"<p><strong>Nothing to analyse:</strong> none of the {n_in} alerts was matched to a Betfair runner with a known result "
                 f"(status counts: {esc(json.dumps(key['status_counts']))}). See the matching tables below.</p>")
    if not has_book and not nothing:
        b.append("<p><strong>No order-book prices in this data:</strong> the strategy tables, walk-forward and verdict fall back to the last traded price (LTP) legs.</p>")
    # 2 matching
    b.append("<h2>2. Matching</h2>")
    m_rows = [[esc(r.sport), esc(r.bf_market), esc(r.status), intf(r.n), pct(r.share)] for r in summary.itertuples()]
    b.append(table(["Sport", "Market", "Status", "n", "Share"], m_rows[:80], left=3))
    b.append(f"<p>{len(review)} alert(s) need review (not matched, ambiguous, score below 85, kickoff gap above 5 minutes, or home/away swapped). "
             f"Files: <code>match_summary.csv</code>, <code>match_review.csv</code>, <code>match_spotcheck.csv</code> (up to 20 random matched alerts next to the Betfair event and runner).</p>")
    # 3 headline
    b.append("<h2>3. Headline ROI</h2>")
    sel = hl[np.isclose(hl.commission, commission) | hl.price.isin(["ref", "close"])]
    b.append(f"<p class='sub'>Commission {cpct} on net winnings. ROI is the mean P/L per bet (per 1 unit of backer's stake). The null p is one-sided: the chance of an ROI this good at fair prices.</p>")
    b.append(table(HL_HEAD, _hl_rows(sel)))
    b.append(f"<details><summary>All commission levels ({len(hl)} rows)</summary>{table(HL_HEAD, _hl_rows(hl))}</details>")
    # 4 cumulative
    b.append("<h2>4. Cumulative P/L</h2>")
    if curves:
        b.append(cum_chart([{"name": n, "short": sh, "cls": c, "t": A._to_ms(cv["ts"]), "y": cv["cum_pl"].to_numpy(float)}
                            for _, n, sh, c, cv in curves], "Cumulative P/L by alert time (units of backer's stake)"))
    else:
        b.append(nothing_p)
    # 5 CLV
    b.append("<h2>5. CLV (closing-line value)</h2>")
    b.append("<p class='sub'>CLV = price taken / last pre-off traded price - 1 for backs (the reverse ratio for lays); above zero means the price beat the close.</p>")
    cl = sel[sel.n > 0] if len(sel) else sel
    order = {"book": 0, "ltp": 1, "ref": 2, "close": 3}
    cl = cl.assign(_o=cl["price"].map(order), _s=cl["side"].map({"back": 0, "lay": 1})).sort_values(["_o", "_s", "latency_s"], kind="stable")
    c_rows = [[esc(LEG_LABEL[(r.side, r.price)]), intf(r.latency_s), intf(r.clv_n), pct(r.clv_mean), ci(r.clv_ci_lo, r.clv_ci_hi), pct(r.clv_pct_pos, 0)]
              for r in cl.itertuples()]
    b.append(table(["Leg", "Delay (s)", "n with a close", "Mean CLV", "95% CI (by event)", "Beat the close"], c_rows) if c_rows else nothing_p)
    # 6 event study
    b.append("<h2>6. Event study</h2>")
    if len(es) and es["n"].sum():
        b.append(es_chart(es, "Last traded price versus the price at the alert, in ticks (pre-off alerts)"))
    else:
        b.append(nothing_p)
    # 7 strategies
    b.append("<h2>7. Strategies</h2>")
    b.append(f"<p>Legs tested: {esc(LEG_LABEL[('back', basis)].lower())} and {esc(LEG_LABEL[('lay', basis)].lower())}, delay 0 s, commission {cpct}"
             + ("" if has_book else " (LTP fallback: no order-book prices)") + ".</p>")
    w_rows = []
    for s in ("back", "lay"):
        w = wf[s]
        if "skipped" in w:
            wtxt, oos, base = f"not run: {w['skipped']} (n = {w['n']})", "n/a", "n/a"
        else:
            o, bs = w.get("oos"), w.get("baseline_all_oos")
            wtxt = f"{sum(1 for p in w['picks'] if p.get('rule'))} of {len(w['picks'])} folds picked a rule; {w['n_rules']} candidate rules"
            oos = "n/a" if not o else f"{pct(o['roi'])} {ci(o['roi_ci_lo'], o['roi_ci_hi'])}, n={o['n']}"
            base = "n/a" if not bs else f"{pct(bs['roi'])}, n={bs['n']}"
        w_rows.append([esc(s), intf(len(legs[s])), intf(n_tests[s]), intf(key["n_bh_pass"][s]), esc(wtxt), esc(oos), esc(base)])
    b.append(table(["Side", "Bets", "Rules tested", "BH survivors (q=0.10)", "Walk-forward", "Out-of-sample ROI [95% CI]", "All bets, same folds"], w_rows))
    skipped = {s: seg[s].attrs.get("skipped", []) for s in seg}
    if any(skipped.values()):
        b.append(f"<p class='note'>Columns skipped (absent, empty or fewer than 5 distinct values): {esc(', '.join(sorted({c for v in skipped.values() for c in v})))}.</p>")
    if positives:
        b.append(f"<div class='verdict'><strong>{len(positives)} rule(s) met all four criteria</strong> (BH survivor, clustered ROI interval above zero, CLV interval above zero, positive out-of-sample ROI): "
                 f"{esc('; '.join(key['positive_rules']))}. Treat them as candidates for a forward test, not as proven edges.</div>")
    else:
        b.append("<div class='verdict'><strong>No rule met all four criteria</strong> (Benjamini-Hochberg survivor at q = 0.10, clustered ROI interval above zero, "
                 "CLV interval above zero, and a positive out-of-sample walk-forward ROI), so none of the segments tested can be called a positive strategy.</div>")
    if len(vall):
        v_rows = [[esc(r.side), esc(r.rule), intf(r.n), pct(r.roi), pct(r.roi_cl_lo), pval(r.p_null), "yes" if r.bh_pass else "no", pct(r.clv_mean),
                   pct(r.clv_ci_lo), "yes" if r.oos_selected else "no", pct(r.oos_roi), "<strong>YES</strong>" if r.positive else "no"] for r in vall.head(30).itertuples()]
        b.append(f"<p>Verdict table (rules with a BH pass, or ROI above zero with n of at least 30); showing {len(v_rows)} of {len(vall)} - full table in <code>verdict.csv</code>.</p>")
        b.append(table(["Side", "Rule", "n", "ROI", "ROI lower (by event)", "Null p", "BH pass", "CLV mean", "CLV lower", "Picked OOS", "OOS ROI", "Positive"], v_rows, left=2))
    else:
        b.append("<p>No rule had a BH pass or a positive ROI with enough bets, so the verdict table is empty.</p>")
    # 8 scalping
    b.append("<h2>8. Scalping</h2>")
    book_fx = fixed[fixed.basis == "book"]
    ltp_fx = fixed[(fixed.basis == "ltp") & (fixed.pen == 0)]
    bar_src, bnote = (book_fx, "Taker on the book at both legs (back at the best back, lay at the best lay, or the reverse). Whiskers = 95% CI by event.") if len(book_fx) else \
        (ltp_fx, "No order-book prices: last traded prices with no spread penalty (optimistic). Whiskers = 95% CI by event.")
    if len(bar_src) and bar_src["n"].sum():
        b.append(bar_chart(bar_src, "Mean green-up per 1 unit of opening stake, by horizon", bnote))
    else:
        b.append(nothing_p)

    def fx_rows(f):
        return [[esc(r.basis), esc(r.side), esc(r.horizon), "n/a" if _na(r.pen) else intf(r.pen), intf(r.n), intf(r.n_missing), pct(r.mean_green, 2),
                 ci(r.ci_lo, r.ci_hi, lambda q: pct(q, 2)), pct(r.median, 2), pct(r.pct_profitable, 0), pct(r.pct_loss, 0)] for r in f.itertuples()]
    fx_head = ["Basis", "Direction", "Horizon (min)", "Spread penalty (ticks)", "n", "Missing", "Mean green", "95% CI (by event)", "Median", "Profitable", "Loss"]
    b.append("<h3>Fixed horizon</h3>" + (table(fx_head, fx_rows(fixed[fixed.basis == 'book']), left=2) if len(book_fx) else "<p class='note'>No order-book fixed-horizon trades.</p>"))
    b.append(f"<details><summary>Last-traded-price basis ({int((fixed.basis == 'ltp').sum())} rows, spread penalty 0 and 1 tick)</summary>{table(fx_head, fx_rows(fixed[fixed.basis == 'ltp']), left=2)}</details>")
    t_rows = [[esc(r.side), intf(r.tp), esc(r.sl), intf(r.pen), intf(r.n), intf(r.n_missing), pct(r.mean_green, 2), ci(r.ci_lo, r.ci_hi, lambda q: pct(q, 2)),
               pct(r.pct_profitable, 0), pct(r.reason_tp, 0), pct(r.reason_sl, 0), pct(r.reason_time, 0), pct(r.reason_off, 0), pct(r.reason_no_exit, 0)] for r in tpsl.itertuples()]
    b.append("<h3>Take-profit / stop-loss (ticks, last-traded-price simulation)</h3>" +
             (table(["Direction", "TP", "SL", "Spread pen.", "n", "Missing", "Mean green", "95% CI (by event)", "Profitable", "TP", "SL", "Time", "Off", "No exit"], t_rows) if t_rows else nothing_p))
    # 9 method
    b.append("<h2>9. Method and caveats</h2>" + METHOD)
    title = f"{label} - ShootingBets ROI report" if label else "ShootingBets ROI report"
    (out / "report.html").write_text(_page(title, label, "".join(b)), encoding="utf-8")
    return _clean(key)
