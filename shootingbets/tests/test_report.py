"""build_report / run_analysis / make_smoke_alerts on synthetic PRO fixtures (NOT real data)."""
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
from sbroi.report import build_report
from sbroi.roi import Outcome, back_pl, lay_pl
from test_pro_book import alerts, pro_dir  # noqa: F401  (pro_dir is a fixture)

sys.path.insert(0, str(ROOT / "scripts"))
COMMS = (0.0, 0.02, 0.048, 0.05)
LATS = (0, 60, 120, 300)
FILES = ["enriched.csv", "match_summary.csv", "match_review.csv", "match_spotcheck.csv", "headline.csv", "event_study.csv",
         "segments_back_book.csv", "segments_lay_book.csv", "walk_forward.json", "verdict.csv", "scalping_fixed.csv",
         "scalping_tpsl.csv", "pl_curve.csv", "report.html", "key_numbers.json"]
HEADINGS = ["1. Summary", "2. Matching", "3. Headline ROI", "4. Cumulative P/L", "5. CLV (closing-line value)", "6. Event study",
            "7. Strategies", "8. Scalping", "9. Method and caveats"]


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
        "Back at best back (4.8%)", "Lay at best lay (4.8%)", "Back at Pinnacle quote"}
    assert len(pd.read_csv(out / "event_study.csv")) == 17 and len(pd.read_csv(out / "match_spotcheck.csv")) == 20

    h = (out / "report.html").read_text(encoding="utf-8")
    for s in HEADINGS:
        assert f"<h2>{s}</h2>" in h, s
    order = [h.index(f"<h2>{s}</h2>") for s in HEADINGS]
    assert order == sorted(order)
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
    for name in ("Back at best back (4.8%)", "Lay at best lay (4.8%)", "Back at Pinnacle quote", "Back then lay", "Lay then back"):
        assert name in h
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
        assert f"<h2>{s}</h2>" in h
    for f in FILES:
        assert (out / f).is_file(), f
    assert len(pd.read_csv(out / "match_review.csv")) == 3 and (pd.read_csv(out / "headline.csv")["n"] == 0).all()


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
