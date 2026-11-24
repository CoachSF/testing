import bz2
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

from conftest import ROOT, basic_market_lines, to_bytes
from sbroi.telegram_export import chats_summary, load_export


def test_telegram_single_chat(tmp_path):
    doc = {"name": "ShootingBets", "type": "bot_chat", "id": 123, "messages": [
        {"id": 2, "type": "message", "date": "2026-10-01T22:00:05", "date_unixtime": "1790805605",
         "from": "ShootingBets", "from_id": "user999",
         "text": ["⚽ ", {"type": "bold", "text": "Arsenal vs Chelsea"}, "\n",
                  {"type": "text_link", "text": "bet", "href": "https://example.org/e/1"}]},
        {"id": 1, "type": "service", "date": "2026-10-01T21:00:00", "date_unixtime": "1790802000", "action": "x"},
        {"id": 3, "type": "message", "date": "2026-10-01T22:10:00", "date_unixtime": "1790806200",
         "edited": "2026-10-01T23:00:00", "edited_unixtime": "1790809200", "from": "ShootingBets",
         "text": "plain text alert"},
    ]}
    p = tmp_path / "result.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    msgs = load_export(str(p))
    assert [m.msg_id for m in msgs] == [2, 3]
    assert msgs[0].text == "⚽ Arsenal vs Chelsea\nbet" and msgs[0].links == ["https://example.org/e/1"]
    from datetime import datetime, timezone
    # must come from date_unixtime (UTC), not the exporter-local "date" string
    assert msgs[0].ts_utc == datetime.fromtimestamp(1790805605, tz=timezone.utc)
    assert msgs[0].ts_utc.isoformat() != "2026-10-01T22:00:05+00:00"
    assert msgs[1].edited_utc is not None
    assert chats_summary(str(p))[0]["n_messages"] == 2


def test_telegram_full_account_export(tmp_path):
    doc = {"chats": {"list": [
        {"name": "Other", "type": "personal_chat", "id": 1, "messages": [
            {"id": 1, "type": "message", "date_unixtime": "1790800000", "text": "hi"}]},
        {"name": "ShootingBets", "type": "bot_chat", "id": 2, "messages": [
            {"id": 9, "type": "message", "date_unixtime": "1790800001", "text": "alert"}]},
    ]}}
    p = tmp_path / "result.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    assert [m.text for m in load_export(str(p), chat_name_contains="shooting")] == ["alert"]


def _load_fetch_script():
    spec = importlib.util.spec_from_file_location("fetch_betfair_basic", ROOT / "scripts" / "fetch_betfair_basic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fetch_script_helpers():
    fb = _load_fetch_script()
    assert fb.split_event("Boston Celtics @ Miami Heat") == ("Miami Heat", "Boston Celtics")
    assert fb.sim("Arsenal FC", "Arsenal") == 1.0
    assert fb.sim("Arsenal", "Everton") < 0.6
    md = fb.first_definition(bz2.compress(to_bytes(basic_market_lines())))
    assert md["eventName"] == "Arsenal v Chelsea"
    f = fb.day_filter("Soccer", "Basic Plan", fb.dt.date(2026, 10, 1), ["MATCH_ODDS"])
    assert f["fromDay"] == f["toDay"] == 1 and f["fileTypeCollection"] == ["M"]


def test_fetch_script_end_to_end_mocked(tmp_path, monkeypatch):
    """Full run against a fake Betfair API: discovery, lenient matching, extra markets, tar output."""
    fb = _load_fetch_script()
    mo = bz2.compress(to_bytes(basic_market_lines("1.200000001")))
    ou = bz2.compress(to_bytes(basic_market_lines("1.200000009")))
    other = bz2.compress(to_bytes(basic_market_lines("1.200000077")))
    files = {
        "/data/BASIC/2026/Oct/1/34000001/1.200000001.bz2": mo,
        "/data/BASIC/2026/Oct/1/34000001/1.200000009.bz2": ou,
        "/data/BASIC/2026/Oct/1/35555555/1.200000077.bz2": other,
    }
    calls = []

    def fake_post(self, method, payload):
        calls.append((method, payload))
        if method == "GetMyData":
            return [{"sport": "Soccer", "plan": "Basic Plan", "forDate": "2026-10-01T00:00:00"}]
        if method == "DownloadListOfFiles":
            if payload["marketTypesCollection"] == ["MATCH_ODDS"]:
                return ["/data/BASIC/2026/Oct/1/34000001/1.200000001.bz2"]
            return ["/data/BASIC/2026/Oct/1/34000001/1.200000009.bz2", "/data/BASIC/2026/Oct/1/35555555/1.200000077.bz2"]
        raise AssertionError(method)

    monkeypatch.setattr(fb.Client, "post", fake_post)
    monkeypatch.setattr(fb.Client, "download", lambda self, p: files[p])
    monkeypatch.setenv("BETFAIR_SSOID", "dummy-for-test")
    from conftest import T0, MIN
    from datetime import datetime, timezone
    ko = datetime.fromtimestamp((T0 + 120 * MIN) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day = ko[:10]
    spec = {"plan": "Basic Plan", "requests": [{"sport": "Soccer", "from": day, "to": day,
            "market_types": ["MATCH_ODDS", "OVER_UNDER_25"],
            "events": [{"home": "Arsenal FC", "away": "Chelsea", "kickoff_utc": ko}]}]}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / "out.tar"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["x", "--spec", str(sp), "--out", str(out), "--cache", str(tmp_path / "c")])
    fb.main()
    with tarfile.open(out) as tf:
        names = sorted(tf.getnames())
    assert "manifest.json" in names
    assert "data/BASIC/2026/Oct/1/34000001/1.200000001.bz2" in names
    assert "data/BASIC/2026/Oct/1/34000001/1.200000009.bz2" in names     # extra market for matched event
    assert not any("35555555" in n for n in names)                        # unmatched event excluded
    # and our own reader can consume the produced archive
    from sbroi.betfair_files import load_markets
    assert set(load_markets(out)) == {"1.200000001", "1.200000009"}
