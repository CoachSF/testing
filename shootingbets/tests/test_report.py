"""build_report / run_analysis / make_smoke_alerts on synthetic PRO fixtures (NOT real data)."""
import html
import importlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, T0
from sbroi.pipeline import Config, _ctag, enrich
from sbroi.pro_index import MarketStore, build_event_index_from_manifest, load_manifest
from sbroi import analysis as A
from sbroi.report import LEG_LABEL, TOO_FEW, _book, _few, build_report
from sbroi.roi import Outcome, back_pl, lay_pl
from test_pro_book import OFF, alerts, pro_dir, split_pro_dir  # noqa: F401  (pro_dir is a fixture)

sys.path.insert(0, str(ROOT / "scripts"))
COMMS = (0.0, 0.02, 0.048, 0.05)
LATS = (0, 60, 120, 300)
REF_LABEL = "Back at the alert's quoted Min. Odds (bookmaker, no commission)"
FILES = ["enriched.csv", "match_summary.csv", "match_review.csv", "match_spotcheck.csv", "headline.csv", "event_study.csv",
         "segments_back_book.csv", "segments_lay_book.csv", "walk_forward.json", "verdict.csv", "scalping_fixed.csv",
         "scalping_tpsl.csv", "pl_curve.csv", "funnel.csv", "min_odds_check.csv", "report.html", "key_numbers.json"]
HEADINGS = ["1. Could these alerts be bet on Betfair?", "2. Against the alert's Min. Odds", "3. Summary", "4. Matching",
            "5. Headline ROI", "6. Cumulative P/L", "7. CLV (closing-line value)", "8. Event study", "9. Strategies",
            "10. Scalping", "11. Method and caveats"]
MINODDS_H = html.escape("Against the alert's Min. Odds")


def svg_text_violations(h):
    """(svg #, text, x, y, why) for every <text> whose anchor or estimated extent (0.6 em per character) leaves its viewBox."""
    import html as _html
    bad, size = [], {"tick": 11, "ax": 11, "lbl": 12}
    for n, m in enumerate(re.finditer(r'<svg[^>]*viewBox="0 0 (\d+) (\d+)"[^>]*>(.*?)</svg>', h, re.S)):
        w, ht, body = int(m[1]), int(m[2]), m[3]
        for t in re.finditer(r'<text class="(\w+)" x="([-\d.]+)" y="([-\d.]+)"(?: text-anchor="(\w+)")?>(.*?)</text>', body):
            cls, x, y, anchor, txt = t[1], float(t[2]), float(t[3]), t[4] or "start", _html.unescape(t[5])
            wd = 0.6 * size[cls] * len(txt)
            x0, x1 = {"start": (x, x + wd), "end": (x - wd, x), "middle": (x - wd / 2, x + wd / 2)}[anchor]
            if x0 < 0 or x1 > w or y - 0.8 * size[cls] < 0 or y + 0.25 * size[cls] > ht:
                bad.append((n, txt, x, y, f"extent {x0:.0f}..{x1:.0f} of {w}, y {y} of {ht}"))
    return bad


def real_enriched(pro_dir, pairs=(("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham"))):
    df = load_manifest(pro_dir)
    cfg = Config(commissions=COMMS, scalp_commission=0.048)
    return enrich(alerts(*pairs), MarketStore(pro_dir, df), build_event_index_from_manifest(df), cfg)


def inflate(base, n=80, seed=5):
    """Replicate the real enrich rows to n alerts, spread them in time / events / prices / outcomes, and recompute every
    P/L column from the new outcome with roi.back_pl / lay_pl so the frame stays internally consistent."""
    rng = np.random.default_rng(seed)
    d = pd.concat([base.iloc[[i % len(base)]] for i in range(n)], ignore_index=True)
    d["alert_id"] = np.arange(1, n + 1)
    d["ts"] = pd.to_datetime(T0, unit="ms", utc=True) + pd.to_timedelta(np.arange(n) * 3, unit="h")
    d["match_event_id"] = [f"ev{i % 25}" for i in range(n)]
    d["league"] = rng.choice(["L1", "L2", "L3"], n)
    d["drop_pct"] = rng.uniform(3, 20, n)
    d["mins_to_off"] = rng.uniform(20, 600, n)
    d["won"] = (rng.random(n) < 0.4).astype(float)
    mult = rng.uniform(0.85, 1.25, n)
    for L in LATS:
        d[f"p_l{L}"] = np.round(2.5 * mult, 2)
        d[f"bb_l{L}"], d[f"bl_l{L}"] = np.round(2.48 * mult, 2), np.round(2.52 * mult, 2)
    d["p_close"] = np.round(2.5 * rng.uniform(0.9, 1.1, n), 2)
    d["ref_odds"] = np.round(2.2 * mult, 2)
    d["bb_l0_size"] = rng.uniform(5, 400, n)
    for i in d.index:
        oc = Outcome(1.0, 0.0, 0.0, "WINNER", True) if d.at[i, "won"] == 1 else Outcome(0.0, 1.0, 0.0, "LOSER", True)
        for L in LATS:
            for c in COMMS:
                t = _ctag(c)
                d.at[i, f"back_pl_l{L}_{t}"] = back_pl(d.at[i, f"p_l{L}"], oc, c)
                d.at[i, f"lay_pl_l{L}_{t}"] = lay_pl(d.at[i, f"p_l{L}"], oc, c)
                d.at[i, f"xback_pl_l{L}_{t}"] = back_pl(d.at[i, f"bb_l{L}"], oc, c)
                d.at[i, f"xlay_pl_l{L}_{t}"] = lay_pl(d.at[i, f"bl_l{L}"], oc, c)
        d.at[i, "ref_back_pl"] = back_pl(d.at[i, "ref_odds"], oc, 0.0)
        d.at[i, "close_back_pl_c0"] = back_pl(d.at[i, "p_close"], oc, 0.0)
    return d


@pytest.fixture
def synth(pro_dir):  # noqa: F811
    base = real_enriched(pro_dir)
    assert list(base["status"]) == ["ok", "ok"]
    return inflate(base)


# ----------------------------------------------------------------------------- build_report
def test_build_report_writes_every_file_and_a_self_contained_html(synth, tmp_path):
    out = tmp_path / "rep"
    key = build_report(synth, out, label="SMOKE TEST - NOT REAL", n_alerts_input=80)
    for f in FILES:
        assert (out / f).is_file() and (out / f).stat().st_size > 0, f
    assert key["n_alerts_input"] == 80 and key["n_ok"] == 80 and key["segment_price_basis"] == "book"
    assert json.loads((out / "key_numbers.json").read_text())["n_ok"] == 80
    wf = json.loads((out / "walk_forward.json").read_text())
    assert set(wf) == {"price_basis", "back", "lay"} and wf["back"]["n_rules"] > 0
    hl = pd.read_csv(out / "headline.csv")
    assert len(hl) == 2 * 2 * 4 * 4 + 2 and hl["n"].max() == 80          # nothing excluded: all 80 ok pre-off, none void
    assert key["headline_n"] == 80 and key["n_rules_tested"]["back"] > 5 and key["n_rules_tested"]["lay"] > 5
    assert len(pd.read_csv(out / "segments_back_book.csv")) == key["n_rules_tested"]["back"]
    assert set(pd.read_csv(out / "pl_curve.csv")["series"]) == {
        "Back at best back (4.8%)", "Lay at best lay (4.8%)", REF_LABEL}
    assert len(pd.read_csv(out / "event_study.csv")) == 17 and len(pd.read_csv(out / "match_spotcheck.csv")) == 20

    h = (out / "report.html").read_text(encoding="utf-8")
    for s in HEADINGS:
        assert f"<h2>{html.escape(s)}</h2>" in h, s
    order = [h.index(f"<h2>{html.escape(s)}</h2>") for s in HEADINGS]
    assert order == sorted(order)
    assert "At bookmaker odds" not in h and not (out / "vendor_join.csv").exists()      # no vendor columns: no such section
    assert "vendor's public results page" not in h                                       # ... and no method note about it
    assert "alerts kick off after the last published" not in h                           # nothing is pro_not_published here
    assert "two-sided book" in key["headline_leg"] and "Headline leg: Back at the best back of a two-sided book" in h
    assert "no more than 20% above the best back" in h and "no_tradeable_book" in h
    head_roi = h.split("<h2>5. Headline ROI</h2>")[1].split("</table>")[0]
    assert TOO_FEW not in head_roi                                                       # 80 bets: nothing to label
    assert h.count("<svg") == 3 and h.count("<figure") == 3 and h.count("<summary>Table view</summary>") == 3
    assert "<title>SMOKE TEST - NOT REAL - ShootingBets ROI report</title>" in h
    assert '<div class="banner" role="alert">SMOKE TEST - NOT REAL</div>' in h and h.index('class="banner"') < h.index("<h1>")
    assert 'body class="viz-root"' in h and "background:var(--page)" in h
    for token in ("#fcfcfb", "#f9f9f7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#2a78d6", "#eb6834", "#1baf7a",
                  "#1a1a19", "#0d0d0d", "#f0efec", "#2c2c2a", "#383835", "#3987e5", "#d95926", "#199e70"):
        assert token in h, token
    assert "@media (prefers-color-scheme: dark){:root:where(:not([data-theme=\"light\"])) .viz-root{--surface:#1a1a19" in h
    assert ':root[data-theme="dark"] .viz-root{--surface:#1a1a19' in h
    assert "UTC" in h and "Melbourne" in h and ("AEDT" in h or "AEST" in h)
    assert "http://" not in h and "https://" not in h and "<link" not in h and "@import" not in h and "url(" not in h
    assert not re.search(r"<(script|img|iframe)[^>]+src=", h)
    for name in ("Back at best back (4.8%)", "Lay at best lay (4.8%)", REF_LABEL, "Back then lay", "Lay then back"):
        assert html.escape(name) in h
    assert LEG_LABEL[("back", "ref")] == REF_LABEL and "Pinnacle" not in h
    assert "The row for the alert's Min. Odds carries no commission." in h and ">Min. Odds " in h
    datas = [json.loads(m) for m in re.findall(r'<script type="application/json" class="chart-data">(.*?)</script>', h, re.S)]
    assert [d["kind"] for d in datas] == ["line", "line", "bar"]
    assert [s["cls"] for s in datas[0]["series"]] == ["s1", "s2", "s3"]            # fixed colour order
    assert all(0 < b["x"] < 720 for b in datas[2]["bars"]) and len(datas[2]["bars"]) == 20
    assert re.search(r"(No rule met all four criteria|rule\(s\) met all four criteria)", h)
    assert "n/a" in h and not re.search(r"\bnan\b", h, re.I)                          # missing numbers read "n/a", never "nan"
    assert "tabular-nums" in h and 'system-ui, -apple-system, "Segoe UI", sans-serif' in h
    assert len(re.findall(r"<text ", h)) > 60 and svg_text_violations(h) == []         # every label sits inside its chart
    assert "drawn clipped" not in h                                                    # no bar needed clipping here


def test_scalping_chart_clips_an_extreme_bar_but_keeps_the_true_value(synth, tmp_path):
    d = synth.copy()
    d.loc[3, "xfx_lay_back_h5"] = -88.0                                              # one thin-book alert: -8800% of stake
    out = tmp_path / "clip"
    build_report(d, out)
    h = (out / "report.html").read_text(encoding="utf-8")
    assert "drawn clipped" in h and svg_text_violations(h) == []
    bars = json.loads(re.findall(r'<script type="application/json" class="chart-data">(.*?)</script>', h, re.S)[2])["bars"]
    b = next(x for x in bars if x["lab"] == "Horizon 5 min" and x["name"] == "Lay then back")
    f = pd.read_csv(out / "scalping_fixed.csv")
    true_mean = f[(f.basis == "book") & (f.side == "lay_back") & (f.horizon_min == 5)]["mean_green"].iloc[0]
    assert true_mean < -1 and b["txt"].startswith(f"{true_mean * 100:+.2f}%")        # tooltip carries the unclipped mean


def test_report_without_book_prices_falls_back_to_ltp_and_says_so(synth, tmp_path):
    thin = synth.drop(columns=[c for c in synth.columns if c.startswith(("bb_", "bl_", "xback", "xlay", "xclose", "xfx"))
                               or c in ("spread_ticks_l0", "tv_l0", "has_book")])
    out = tmp_path / "ltp"
    key = build_report(thin, out)
    assert key["segment_price_basis"] == "ltp" and key["headline_n"] == 80
    h = (out / "report.html").read_text(encoding="utf-8")
    assert "fall back to the last traded price" in h and "Back at last traded price (4.8%)" in h
    assert "Back at best back (4.8%)" not in h and '<div class="banner"' not in h      # chart series renamed; no label -> no banner
    assert (pd.read_csv(out / "segments_back_book.csv")["price_basis"] == "ltp").all()
    hl = pd.read_csv(out / "headline.csv")
    assert (hl[hl.price == "book"]["note"] == "columns missing").all() and (hl[hl.price == "ltp"]["n"] == 80).all()


def test_report_with_nothing_matched_still_builds_and_says_so(tmp_path):
    n = pd.DataFrame({"alert_id": [1, 2, 3], "ts": [pd.Timestamp("2026-10-01", tz="UTC")] * 3, "sport": "Soccer",
                      "home": "A", "away": "B", "bf_market": "MATCH_ODDS", "sel_kind": "home", "status": "no_event_match",
                      "match_score": [40.0, 55.0, np.nan], "match_reason": "low_score", "match_event_id": None})
    out = tmp_path / "empty"
    key = build_report(n, out, label="X")
    assert key["n_ok"] == 0 and key["n_alerts_input"] == 3 and key["any_positive_rule"] is False
    h = (out / "report.html").read_text(encoding="utf-8")
    assert "Nothing to analyse" in h and "<svg" not in h and "http://" not in h and "https://" not in h
    for s in HEADINGS:
        assert f"<h2>{html.escape(s)}</h2>" in h
    assert "for 0 of 0 priced alerts" in h and "No resting-order columns in this data" in h        # the Min. Odds blocks, n = 0
    for f in FILES:
        assert (out / f).is_file(), f
    assert len(pd.read_csv(out / "match_review.csv")) == 3 and (pd.read_csv(out / "headline.csv")["n"] == 0).all()


def _section(h, start, end):
    return h.split(start)[1].split(end)[0]


def test_funnel_counts_the_sentence_and_the_table_of_ok_alerts(synth, tmp_path):
    d = synth.copy()
    d["sport"] = ["Soccer"] * 40 + ["Basketball"] * 40
    for lo, hi, st in ((0, 9, "pro_not_published"), (10, 14, "no_event_match"), (15, 17, "unmapped_market"),
                       (18, 19, "no_market"), (20, 21, "no_runner"), (40, 44, "pro_not_published")):
        d.loc[lo:hi, "status"] = st
    d.loc[22, ["bb_l0", "book_ok_l0"]] = [np.nan, False]     # ok, but no best back at the alert
    d.loc[23, "result_known"] = False        # ok and priced, result not known
    d.loc[24, "void"] = 1.0                  # ok and priced, void
    d.loc[25:27, "book_ok_l0"] = False       # priced and settled, but the book is not tradeable
    out = tmp_path / "f"
    key = build_report(d, out)
    f = pd.read_csv(out / "funnel.csv")
    assert list(f.columns) == A.FUNNEL_COLS == ["sport", "alerts", "pro_not_published", "in_coverage", "event_on_betfair",
                                               "market_found", "runner_found", "priced_at_alert", "tradeable_at_alert",
                                               "priced_and_settled", "tradeable_and_settled"]
    assert f.values.tolist() == [["Basketball", 40, 5, 35, 35, 35, 35, 35, 35, 35, 35],
                                 ["Soccer", 40, 10, 30, 25, 20, 18, 17, 14, 15, 12], ["All", 80, 15, 65, 60, 55, 53, 52, 49, 50, 47]]
    assert key["funnel"] == dict(zip(A.FUNNEL_COLS, ["All", 80, 15, 65, 60, 55, 53, 52, 49, 50, 47]))
    h = (out / "report.html").read_text(encoding="utf-8")
    top = _section(h, "<h2>1. Could these alerts be bet on Betfair?</h2>", f"<h2>2. {MINODDS_H}</h2>")
    assert all(f">{c}</th>" in top for c in A.FUNNEL_COLS)
    assert ("Of 80 alerts, 15 kicked off after the published Betfair data ended and were not matched; of the 65 in coverage, "
            "60 had their event on Betfair, 55 had the mapped market, 53 the runner, 52 a best back at the alert time, "
            "49 a tradeable two-sided book (best lay within 20% of the best back), and 47 were tradeable and settled "
            "(50 priced and settled).") in top
    assert ("15 alerts kick off after the last published Betfair PRO month; they can be priced once Betfair publishes it.") in top
    ok = _section(top, "<h3>Alerts matched to a Betfair runner</h3>", "</table>")
    assert ok.count("<tr>") - 1 == 53 and "Arsenal FC v Chelsea FC" in ok and "Arsenal v Chelsea" in ok and "MATCH_ODDS" in ok
    assert all(f">{c}</th>" in ok for c in ("Alert time", "Betfair event", "Selection", "Best back (size)", "Best lay", "Book", "Min. Odds", "Result"))
    assert "UTC /" in ok and "Melbourne" in ok and "one-sided" in ok and re.search(r"two-sided \d+\.\d%", ok)
    assert (_book(2.0, 2.2), _book(np.nan, 2.2), _book(2.0, np.nan), _book(np.nan, np.nan)) == ("two-sided 10.0%", "one-sided",
                                                                                                "one-sided", "none")


def test_a_roi_on_fewer_than_30_bets_is_labelled_in_the_headline_the_tiles_and_the_verdict(synth, tmp_path, monkeypatch):
    assert _few(29, "5.0%").endswith(f'<span class="warn">{TOO_FEW}</span>') and TOO_FEW == "too few bets to conclude"
    assert _few(30, "x") == _few(0, "x") == _few(None, "x") == _few(float("nan"), "x") == "x" and _few(1, "x") != "x"
    out = tmp_path / "tiny"
    key = build_report(synth.head(5), out)
    assert key["headline_n"] == 5
    h = (out / "report.html").read_text(encoding="utf-8")
    hl = pd.read_csv(out / "headline.csv")
    sel = hl[np.isclose(hl.commission, 0.048) | hl.price.isin(["ref", "close"])]
    want = int(((sel.n > 0) & (sel.n < 30)).sum())
    assert want >= 4 and _section(h, "<h2>5. Headline ROI</h2>", "</table>").count(TOO_FEW) == want
    assert re.search(r'<div class="k">Headline ROI</div><div class="v">-?[\d.]+% <span class="warn">too few bets', h)
    for f in FILES:
        assert (out / f).is_file()
    # the verdict table: n and out-of-sample n decide, row by row
    cols = dict(roi_cl_lo=0.1, p_null=0.01, bh_pass=True, clv_mean=0.01, clv_ci_lo=0.0, oos_selected=True, oos_roi=0.2, positive=False)
    fake = pd.DataFrame([dict(rule="a == 1", n=12, roi=0.5, oos_n=5, **cols), dict(rule="b == 2", n=40, roi=0.3, oos_n=45, **cols)],
                        columns=A.VERDICT_COLS)
    monkeypatch.setattr(A, "verdict", lambda s, w, min_n=30: fake)
    out2 = tmp_path / "verdict"
    build_report(synth, out2)
    v = _section((out2 / "report.html").read_text(encoding="utf-8"), "<p>Verdict table", "<h2>10. Scalping</h2>")
    rows = [r for r in v.split("<tr>")[2:] if "</td>" in r]
    assert len(rows) == 4                                              # two rules, back and lay
    lab = f' <span class="warn">{TOO_FEW}</span>'
    for r in rows:
        if "a == 1" in r:
            assert f"<td>50.0%{lab}</td>" in r and f"<td>20.0%{lab}</td>" in r        # ROI on 12, out-of-sample ROI on 5
        else:
            assert "<td>30.0%</td>" in r and "<td>20.0%</td>" in r and TOO_FEW not in r


def test_vendor_section_appears_only_with_vendor_columns(synth, tmp_path):
    d = synth.copy()
    d["min_odds"] = d["ref_odds"]
    n = np.arange(80)
    d["own_pl_min"] = np.where(n < 50, np.where(d["won"] == 1, d["min_odds"] - 1, -1.0), np.nan)
    d["v_b365_odds"] = d["min_odds"] + 0.1
    d["v_pl_b365"] = np.where(n < 12, np.where(d["won"] == 1, d["v_b365_odds"] - 1, -1.0), np.nan)
    d["v_found"], d["check_status"], d["selection_raw"] = n < 50, "checked", "Tritium +1.5"
    out = tmp_path / "v"
    build_report(d, out)
    vt = A.vendor_table(d)
    p1, p2 = d["own_pl_min"].dropna(), d["v_pl_b365"].dropna()
    assert list(vt["n"]) == [50, 12] and vt.loc[0, "roi"] == pytest.approx(p1.mean()) and vt.loc[1, "roi"] == pytest.approx(p2.mean())
    assert vt.loc[0, "strike_rate"] == pytest.approx((p1 > 0).mean()) and vt.loc[0, "avg_odds"] == pytest.approx(d["min_odds"][:50].mean())
    assert vt.loc[1, "avg_odds"] == pytest.approx(d["v_b365_odds"][:12].mean())
    assert (vt["roi_ci_lo"] <= vt["roi"]).all() and (vt["roi"] <= vt["roi_ci_hi"]).all() and vt["p_null"].between(0, 1).all()
    h = (out / "report.html").read_text(encoding="utf-8")
    order = [h.index(f"<h2>{html.escape(s)}</h2>") for s in (
        "1. Could these alerts be bet on Betfair?", "2. At bookmaker odds (vendor-scored alerts)", "3. Against the alert's Min. Odds",
        "4. Summary", "5. Matching", "6. Headline ROI", "12. Method and caveats")]
    assert order == sorted(order)
    sec = _section(h, "<h2>2. At bookmaker odds (vendor-scored alerts)</h2>", f"<h2>3. {MINODDS_H}</h2>")
    assert html.escape("Back at the alert's Min. Odds on the alert's own line") in sec
    assert html.escape("Back at Bet365 odds on Bet365's line, as the vendor scores it") in sec
    assert sec.count(TOO_FEW) == 1 and "public results page" in sec and '(the "Checked" alerts)' in sec
    assert "can differ from the alert's own line" in sec
    assert '"At bookmaker odds" uses results taken from the vendor' in h
    j = pd.read_csv(out / "vendor_join.csv")
    assert list(j.columns) == ["alert_id", "v_found", "v_b365_odds", "own_pl_min", "v_pl_b365", "check_status", "selection_raw",
                               "min_odds"] and len(j) == 80
    # nothing joined at all: the section still renders, with n = 0 and no numbers
    e = d.assign(own_pl_min=np.nan, v_pl_b365=np.nan)
    build_report(e, tmp_path / "v0")
    s0 = _section((tmp_path / "v0" / "report.html").read_text(encoding="utf-8"), "<h2>2. At bookmaker odds", f"<h2>3. {MINODDS_H}</h2>")
    assert "<td>0</td><td>n/a</td>" in s0 and TOO_FEW not in s0


def test_min_odds_section_book_legs_and_the_headline_wording(synth, tmp_path):
    d, n = synth.copy(), np.arange(80)
    d["ref_odds"] = np.where(n % 4 == 0, d["bb_l0"] + 0.3, d["bb_l0"] - 0.1)       # the best back is below Min. Odds for every 4th
    d["bb_l300"] = d["bb_l300"] - np.where(n < 10, 0.6, 0.0)                        # ... and, at 300 s, for 7 more of the first ten
    d["book_ok_l0"] = n >= 10                                                      # ten alerts without a tradeable book
    d["n_trades_after"] = np.where(n % 3 == 0, 0, 5)
    d["limit_fill"] = n % 3 == 1
    d["max_ltp_after"] = np.where(n % 3 == 0, np.nan, d["ref_odds"] + np.where(n % 3 == 1, 0.1, -0.1))
    d["limit_pl_c4p8"] = np.where(d["limit_fill"], np.where(d["won"] == 1, (d["ref_odds"] - 1) * 0.952, -1.0), np.nan)
    out = tmp_path / "mo"
    key = build_report(d, out)
    h = (out / "report.html").read_text(encoding="utf-8")
    sec = _section(h, f"<h2>2. {MINODDS_H}</h2>", "<h2>3. Summary</h2>")
    # a. k of n priced alerts, worked out here with plain numpy
    ks = [int((d[f"bb_l{L}"] >= d["ref_odds"]).sum()) for L in (0, 60, 120, 300)]
    assert ks == [60, 60, 60, 53] and [r["k"] for r in key["min_odds"]["best_back_at_or_above_min"]] == ks
    for L, k in zip((0, 60, 120, 300), ks):
        assert f'<td class="t">{L}</td><td>80</td><td>{k}</td><td>{k / 80 * 100:.1f}%</td>' in sec
    assert "At 0 s the best back was at or above Min. Odds for 60 of 80 priced alerts." in sec
    # b. tradeable books only (70 of 80): price / Min. Odds - 1 for the best back and the mid price
    m = n >= 10
    g, mid = (d["bb_l0"] / d["ref_odds"] - 1)[m], ((d["bb_l0"] + d["bl_l0"]) / 2 / d["ref_odds"] - 1)[m]
    lay = int((d["bl_l0"] >= d["ref_odds"])[m].sum())
    gk = key["min_odds"]["tradeable_gap"]
    assert gk["n"] == 70 and gk["lay_ge_min"] == lay and gk["best_back"]["median"] == pytest.approx(g.median())
    assert gk["best_back"]["min"] == pytest.approx(g.min()) and gk["mid"]["max"] == pytest.approx(mid.max())
    assert f"The best lay was at or above Min. Odds for {lay} of 70 tradeable books." in sec
    assert f"<td>{g.median() * 100:.1f}%</td>" in sec and f"<td>{mid.min() * 100:.1f}%</td>" in sec
    # c. the resting order: 27 of 80 filled, so its ROI carries the label; the other blocks (n = 80, 70) do not
    f = d[d["limit_fill"]]
    r = key["min_odds"]["resting_order"]
    assert (r["placed"], r["traded"], r["filled"], r["wins"]) == (80, 53, 27, int((f["won"] == 1).sum()))
    assert r["roi"] == pytest.approx(f["limit_pl_c4p8"].mean()) and r["strike_rate"] == pytest.approx((f["won"] == 1).mean())
    assert r["roi_ci_lo"] <= r["roi"] <= r["roi_ci_hi"]
    cells = f"{r['wins']}</td><td>{r['strike_rate'] * 100:.1f}%</td><td>{r['roi'] * 100:.1f}% "
    assert '<td class="t">80</td><td>53</td><td>27</td><td>' + cells + f'<span class="warn">{TOO_FEW}</span></td>' in sec
    assert sec.count(TOO_FEW) == 1
    assert "queue position" in sec and "drifts out against the alert" in sec and "ROI on filled (4.8%)" in sec
    c = pd.read_csv(out / "min_odds_check.csv")
    assert len(c) == 80 and list(c.columns) == ["alert_id", "ts", "sport", "fixture", "selection_raw", "min_odds", "bb_l0", "bl_l0",
                                                "spread_pct_l0", "book_ok_l0", "bb/min-1", "max_ltp_after", "n_trades_after",
                                                "limit_fill", "won", "void", "limit_pl_c4p8"]
    assert c["bb/min-1"].tolist() == pytest.approx((d["bb_l0"] / d["ref_odds"] - 1).tolist()) and c["limit_fill"].sum() == 27
    # book legs: the ten alerts without a tradeable book leave the back-at-best-back leg at 0 s, not the others
    hl = pd.read_csv(out / "headline.csv")
    row = lambda price, lat: hl[(hl.side == "back") & (hl.price == price) & (hl.latency_s == lat) & np.isclose(hl.commission, 0.048)].iloc[0]  # noqa: E731
    assert row("book", 0)["n"] == 70 and json.loads(row("book", 0)["excluded"])["no_tradeable_book"] == 10
    assert row("book", 60)["n"] == 80 and row("ltp", 0)["n"] == 80 and key["headline_n"] == 70
    assert key["funnel"]["tradeable_at_alert"] == key["funnel"]["tradeable_and_settled"] == 70 and key["funnel"]["priced_at_alert"] == 80


def test_run_analysis_coverage_end(pro_dir, tmp_path, capsys):  # noqa: F811
    ra = importlib.import_module("run_analysis")
    csv = tmp_path / "alerts.csv"
    al = alerts(("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham"))
    al["ts"], al["kickoff"] = (al[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ") for c in ("ts", "kickoff"))
    al.to_csv(csv, index=False)
    off = pd.to_datetime(OFF, unit="ms", utc=True)
    base = ["--alerts", str(csv), "--pro-dir", str(pro_dir)]
    counts = lambda name: json.loads((tmp_path / name / "key_numbers.json").read_text())["status_counts"]  # noqa: E731
    assert ra.main(base + ["--out", str(tmp_path / "a"), "--coverage-end", off.strftime("%Y-%m-%dT%H:%M:%SZ")]) == 0
    assert counts("a") == {"pro_not_published": 2}                                     # kickoff at the end: not published
    assert ra.main(base + ["--out", str(tmp_path / "b"), "--coverage-end", (off + pd.Timedelta(days=1)).strftime("%Y-%m-%d")]) == 0
    assert counts("b") == {"ok": 2}                                                    # a date alone is midnight UTC
    assert ra.main(base + ["--out", str(tmp_path / "c")]) == 0 and counts("c") == {"ok": 2}   # default: no coverage end
    capsys.readouterr()
    assert ra.main(base + ["--out", str(tmp_path / "d"), "--coverage-end", "not a date"]) == 2
    assert "bad input" in capsys.readouterr().out and not (tmp_path / "d").exists()


def test_report_needs_a_status_column(tmp_path):
    with pytest.raises(ValueError):
        build_report(pd.DataFrame({"alert_id": [1]}), tmp_path / "x")


# ----------------------------------------------------------------------------- scripts
def _manifest_bytes(pro_dir, size=50_000):
    for p in pro_dir.glob("manifest_*.jsonl"):
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.startswith("{")]
        p.write_text("\n".join(json.dumps({**r, "bytes": 5 if "missing" in r["saved"] else size}) for r in rows) + "\n", encoding="utf-8")


def test_run_analysis_turns_an_enrich_exception_into_an_error_row_and_exits_zero(pro_dir, tmp_path, monkeypatch, capsys):  # noqa: F811
    ra = importlib.import_module("run_analysis")
    pairs = [("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham")] * 5
    csv = tmp_path / "alerts.csv"
    al = alerts(*pairs)
    al["ts"] = al["ts"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    al["kickoff"] = al["kickoff"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    al.to_csv(csv, index=False)
    real = ra.enrich

    def flaky(one, *a, **k):
        if int(one.iloc[0]["alert_id"]) == 4:
            raise RuntimeError("boom")
        return real(one, *a, **k)

    monkeypatch.setattr(ra, "enrich", flaky)
    out = tmp_path / "out"
    rc = ra.main(["--alerts", str(csv), "--pro-dir", str(pro_dir), "--out", str(out), "--label", "SMOKE"])
    text = capsys.readouterr().out
    assert rc == 0 and "progress: 10/10 alerts" in text and "status counts:" in text and "error:RuntimeError" in text
    assert str(out.resolve()) in text and text.isascii()
    e = pd.read_csv(out / "enriched.csv").set_index("alert_id")
    assert e.loc[4, "status"] == "error:RuntimeError" and e.loc[4, "error"] == "boom"
    assert (e.drop(index=4)["status"] == "ok").all() and len(e) == 10
    assert e["match_event_id"].dropna().astype(str).nunique() == 2
    assert (out / "report.html").is_file() and json.loads((out / "key_numbers.json").read_text())["status_counts"] == {"ok": 9, "error:RuntimeError": 1}


def test_run_analysis_reads_several_pro_dirs(pro_dir, tmp_path, tmp_path_factory, capsys):  # noqa: F811
    ra = importlib.import_module("run_analysis")
    assert ra.parse_pro_dirs(["a", "b=Tennis", "C:/x/y=Ice Hockey", "d=", "e=f=Golf"]) == (
        ["a", "b", "C:/x/y", "d", "e=f"], ["Soccer", "Tennis", "Ice Hockey", "Soccer", "Golf"])
    other = tmp_path_factory.mktemp("other")
    split_pro_dir(pro_dir, other)                      # Everton v Fulham now lives in another folder
    csv = tmp_path / "alerts.csv"
    al = alerts(("Arsenal FC", "Chelsea FC"), ("Everton", "Fulham"))
    al["ts"], al["kickoff"] = (al[c].dt.strftime("%Y-%m-%dT%H:%M:%SZ") for c in ("ts", "kickoff"))
    al.to_csv(csv, index=False)
    run = lambda *dirs, out: ra.main(["--alerts", str(csv), "--pro-dir", *map(str, dirs), "--out", str(tmp_path / out)])  # noqa: E731
    # DIR=Sport is the sport of rows without event_type_id: as "Tennis" the moved Everton v Fulham event cannot match
    assert run(pro_dir, f"{other}=Tennis", out="o1") == 0
    assert list(pd.read_csv(tmp_path / "o1" / "enriched.csv")["status"]) == ["ok", "no_event_match"]
    assert run(pro_dir, f"{other}=Soccer", out="o2") == 0
    e = pd.read_csv(tmp_path / "o2" / "enriched.csv")
    assert list(e["status"]) == ["ok", "ok"] and e["bf_market_id"].notna().all() and list(e["bb_l0"]) == [2.48, 2.48]
    assert json.loads((tmp_path / "o2" / "key_numbers.json").read_text())["status_counts"] == {"ok": 2}
    capsys.readouterr()
    assert run(tmp_path / "nothing", tmp_path / "none2", out="o3") == 2           # no manifest rows in any folder
    assert "no manifest_*.jsonl rows in" in capsys.readouterr().out and not (tmp_path / "o3").exists()


def test_run_analysis_bad_input_exits_2(pro_dir, tmp_path, capsys):  # noqa: F811
    ra = importlib.import_module("run_analysis")
    good = tmp_path / "a.csv"
    alerts(("Arsenal FC", "Chelsea FC")).assign(ts="2026-10-01T00:00:00Z").to_csv(good, index=False)
    args = lambda **k: ["--alerts", str(k.get("a", good)), "--pro-dir", str(k.get("p", pro_dir)), "--out", str(tmp_path / "o")]  # noqa: E731
    assert ra.main(args(a=tmp_path / "missing.csv")) == 2
    nocols = tmp_path / "b.csv"
    pd.DataFrame({"alert_id": [1], "ts": ["2026-10-01T00:00:00Z"]}).to_csv(nocols, index=False)
    assert ra.main(args(a=nocols)) == 2
    badts = tmp_path / "c.csv"
    alerts(("Arsenal FC", "Chelsea FC")).assign(ts="not a time").to_csv(badts, index=False)
    assert ra.main(args(a=badts)) == 2
    (tmp_path / "emptydir").mkdir()
    assert ra.main(args(p=tmp_path / "emptydir")) == 2
    assert "bad input" in capsys.readouterr().out and not (tmp_path / "o").exists()


def test_smoke_alerts_end_to_end_on_the_fixture_store(pro_dir, tmp_path, capsys):  # noqa: F811
    _manifest_bytes(pro_dir)
    ms = importlib.import_module("make_smoke_alerts")
    ra = importlib.import_module("run_analysis")
    csv = tmp_path / "smoke.csv"
    assert ms.main(["--n", "5", "--pro-dir", str(pro_dir), "--out", str(csv)]) == 0
    a = pd.read_csv(csv)
    assert len(a) == 2 and set(a["text"]) == {"SMOKE"} and list(a["bf_market"]) == ["MATCH_ODDS", "OVER_UNDER_25"]
    assert a["sel_kind"][0] in ("home", "away", "draw") and a["sel_kind"][1] in ("over", "under")
    gap = (pd.to_datetime(a["kickoff"], utc=True) - pd.to_datetime(a["ts"], utc=True)).dt.total_seconds() / 60
    assert gap.between(20, 600.5).all() and a[["sel_team", "sel_hc", "ref_odds"]].isna().all().all()
    assert list(pd.read_csv(csv).columns) == ms.COLUMNS
    assert ms.make(pro_dir, 5, 7).equals(ms.make(pro_dir, 5, 7))                         # seeded
    capsys.readouterr()
    assert ra.main(["--alerts", str(csv), "--pro-dir", str(pro_dir), "--out", str(tmp_path / "o"), "--label", "SMOKE"]) == 0
    e = pd.read_csv(tmp_path / "o" / "enriched.csv")
    assert list(e["status"]) == ["ok", "ok"] and e["bf_market_id"].notna().all()
    _manifest_bytes(pro_dir, 5)                                                          # too small -> nothing to pick
    assert ms.main(["--n", "5", "--pro-dir", str(pro_dir), "--out", str(csv)]) == 2
