"""Tests for scripts/fetch_telegram_web.py - no network, no Chrome: a FakeTab returns canned JSON strings."""
import importlib.util
import json
import re
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from conftest import ROOT
from sbroi.telegram_export import load_export

_spec = importlib.util.spec_from_file_location("fetch_telegram_web", ROOT / "scripts" / "fetch_telegram_web.py")
web = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(web)

TS = 1790825415  # 2026-10-01T03:30:15Z
CHAT = {"id": 777, "username": "ShootingBetsBot", "first_name": "ShootingBets", "bot": True}


def u16(s):
    return len(s.encode("utf-16-le")) // 2


def raw(i, text="hello", **kw):
    return {"_": "message", "id": i, "date": TS + i, "message": text,
            "peer_id": {"_": "peerUser", "user_id": 777}, "from_id": {"_": "peerUser", "user_id": 777}, **kw}


class FakeTab:
    """Answers by substring of the JS expression; ``pages`` maps offset_id -> list of raw messages."""

    def __init__(self, logged_in=True, pages=None, errors=None):
        self.logged_in, self.pages, self.errors, self.calls, self.closed = logged_in, pages or {}, errors or {}, [], False

    def evaluate(self, expr):
        self.calls.append(expr)
        for key, err in self.errors.items():
            if key in expr:
                return json.dumps({"__error__": err})
        if "users.getUsers" in expr:
            return json.dumps([{"_": "user", "id": 1}] if self.logged_in else [])
        if "contacts.resolveUsername" in expr:
            return json.dumps(CHAT)
        if "messages.getHistory" in expr:
            return json.dumps(self.pages.get(int(re.search(r"offset_id: (\d+)", expr).group(1)), []))
        if "auth.logOut" in expr:
            self.logged_in = False
            return json.dumps({"_": "auth.loggedOut"})
        if "location.reload" in expr:
            return None
        if "typeof (window.rootScope" in expr:
            return True
        raise AssertionError("unexpected expression: " + expr[:80])

    def close(self):
        self.closed = True

    def offsets(self):
        return [int(re.search(r"offset_id: (\d+)", c).group(1)) for c in self.calls if "messages.getHistory" in c]


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(web.time, "sleep", sleeps.append)
    return sleeps


# ----------------------------------------------------------------------------- raw -> export mapping
def test_mapping_emoji_links_buttons(tmp_path):
    text = "\U0001F525 Arsenal v Chelsea - details here: https://example.com/odds/1 ok"
    off = lambda sub: u16(text[:text.index(sub)])  # noqa: E731
    m = raw(1, text, entities=[
        {"_": "messageEntityTextUrl", "offset": off("details"), "length": 7, "url": "https://example.com/event/9"},
        {"_": "messageEntityUrl", "offset": off("https://example.com/odds/1"), "length": len("https://example.com/odds/1")}],
        reply_markup={"_": "replyInlineMarkup", "rows": [{"_": "keyboardButtonRow", "buttons": [
            {"_": "keyboardButtonUrl", "text": "Open", "url": "https://btn.example/x"},
            {"_": "keyboardButtonCallback", "text": "Seen", "data": {"__bytes__": list(b"seen:1")}},
            {"_": "keyboardButton", "text": "Plain"}]}]},
        reply_to={"_": "messageReplyHeader", "reply_to_msg_id": 3}, media={"_": "messageMediaWebPage"})
    d = web.export_raw(m, "ShootingBets")
    assert "".join(p["text"] for p in d["text_entities"]) == text
    assert [p["type"] for p in d["text_entities"]] == ["plain", "text_link", "plain", "link", "plain"]
    assert d["text_entities"][1]["href"] == "https://example.com/event/9"
    assert d["inline_bot_buttons"] == [[{"type": "url", "text": "Open", "data": "https://btn.example/x"},
                                        {"type": "callback", "text": "Seen", "data": "seen:1"},
                                        {"type": "other", "text": "Plain", "data": None}]]
    assert d["reply_to_message_id"] == 3 and d["media_type"] == "MessageMediaWebPage"
    assert (d["from"], d["from_id"], d["type"], d["date_unixtime"]) == ("ShootingBets", "user777", "message", str(TS + 1))
    doc = tmp_path / "e.json"
    doc.write_text(json.dumps({"name": "x", "messages": [d]}, ensure_ascii=False), encoding="utf-8")
    got = load_export(str(doc))[0]
    assert got.text == text and "https://example.com/event/9" in got.links and "https://example.com/odds/1" in got.links
    assert {"text": "Open", "url": "https://btn.example/x"} in got.buttons


def test_mapping_edit_service_and_sender():
    d = web.export_raw(raw(2, edit_date=TS + 600), "ShootingBets")
    assert d["edited_unixtime"] == str(TS + 600) and "edited" in d
    assert "edited" not in web.export_raw(raw(2), "ShootingBets")
    svc = web.export_raw({"_": "messageService", "id": 3, "date": TS, "peer_id": {"_": "peerUser", "user_id": 777},
                          "action": {"_": "messageActionChatCreate"}}, "ShootingBets")
    assert svc["type"] == "service" and svc["text"] == "" and svc["from_id"] == "user777"  # peer_id fallback
    out = web.export_raw(raw(4, out=True), "ShootingBets")
    assert out["from"] == "me" and web.export_raw(raw(4), "ShootingBets")["from"] == "ShootingBets"


# ----------------------------------------------------------------------------- pagination
def test_fetch_all_pagination_oldest_first_and_dedupe(no_sleep):
    tab = FakeTab(pages={0: [raw(9), raw(8), raw(7)], 7: [raw(7), raw(6), {"_": "messageEmpty", "id": 5}],
                         5: [raw(4), raw(3), raw(2)], 2: [raw(1)]})
    msgs = web.fetch_all(tab, page_size=3, pause=0.25)
    assert [m["id"] for m in msgs] == [1, 2, 3, 4, 6, 7, 8, 9]  # oldest first, id 7 once, messageEmpty 5 dropped
    assert tab.offsets() == [0, 7, 5, 2, 1]                     # next offset = smallest id received (5 = empty too)
    assert no_sleep == [0.25] * 4                               # a pause between pages, none after the last
    assert [m["id"] for m in web.fetch_all(tab, limit=3, page_size=3)] == [1, 2, 3]  # --limit keeps the OLDEST


def test_fetch_all_stops_when_a_page_repeats():
    same = [raw(3), raw(2), raw(1)]
    tab = FakeTab(pages={0: same, 1: same, 2: same, 3: same})  # the server ignores the offset
    assert [m["id"] for m in web.fetch_all(tab, page_size=3, pause=0)] == [1, 2, 3]
    assert tab.offsets() == [0, 1]
    assert web.fetch_all(FakeTab(), pause=0) == []


# ----------------------------------------------------------------------------- JS building / protocol
def test_js_calls_keep_access_hash_in_the_page():
    tab = FakeTab()
    assert web.resolve_chat(tab, "@ShootingBetsBot") == CHAT
    js = tab.calls[0]
    assert "'contacts.resolveUsername'" in js and '"ShootingBetsBot"' in js and "@" not in js
    assert "window.__sb_peer = {_: 'inputPeerUser'" in js and "access_hash: u.access_hash" in js
    t2 =FakeTab(pages={5: [raw(4)]})
    assert web.history_page(t2, 5, 100) == [raw(4)]
    assert "peer: window.__sb_peer" in t2.calls[0] and "offset_id: 5" in t2.calls[0] and "limit: 100" in t2.calls[0]
    assert "__bytes__" in web._JS and "bigint" in web._JS and "slice(0, 300)" in web._JS


class FakeWS:
    def __init__(self, replies):
        self.sent, self.replies = [], list(replies)

    def send(self, s):
        self.sent.append(json.loads(s))

    def recv(self):
        return json.dumps(self.replies.pop(0))


def test_tab_evaluate_protocol():
    t = web.Tab.__new__(web.Tab)
    t._id = 0
    t.ws = FakeWS([{"method": "Page.frameNavigated", "params": {}}, {"id": 99, "result": {}},
                   {"id": 1, "result": {"result": {"type": "string", "value": "ok"}}},
                   {"id": 2, "result": {"exceptionDetails": {"text": "Uncaught",
                                                             "exception": {"description": "TypeError: boom"}}}},
                   {"id": 3, "error": {"message": "Cannot find context"}}])
    assert t.evaluate("1+1") == "ok"  # events and other ids are skipped
    p = t.ws.sent[0]
    assert p["method"] == "Runtime.evaluate" and p["params"]["awaitPromise"] and p["params"]["returnByValue"]
    with pytest.raises(RuntimeError, match="TypeError: boom"):
        t.evaluate("x")
    with pytest.raises(RuntimeError, match="Cannot find context"):
        t.evaluate("y")


def test_tab_discovery_failures():
    with pytest.raises(RuntimeError, match="no Telegram Web tab"):  # nothing listens on the port
        web.Tab(1)

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps([{"type": "service_worker", "url": "https://web.telegram.org/sw.js",
                                "webSocketDebuggerUrl": "ws://x"},
                               {"type": "page", "url": "https://example.com/", "webSocketDebuggerUrl": "ws://y"}]).encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RuntimeError, match="no Telegram Web tab"):  # no *page* on web.telegram.org
            web.Tab(srv.server_address[1])
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------------- main()
def test_not_logged_in_returns_2_and_writes_nothing(tmp_path, monkeypatch, capsys):
    tab = FakeTab(logged_in=False)
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    out = tmp_path / "x" / "e.json"
    assert web.main(["--out", str(out)]) == 2
    assert web.NOT_LOGGED_IN in capsys.readouterr().out and not out.exists() and not out.parent.exists()
    assert len(tab.calls) == 1 and "users.getUsers" in tab.calls[0] and tab.closed  # nothing but the login check


def test_api_error_is_raised_and_main_reports_it(tmp_path, monkeypatch, capsys):
    with pytest.raises(web.ApiError, match="FLOOD_WAIT_5"):
        web.api_call(FakeTab(errors={"foo.bar": '{"type":"FLOOD_WAIT_5","code":420}'}), "foo.bar", {"a": 1})
    tab = FakeTab(errors={"contacts.resolveUsername": '{"type":"USERNAME_NOT_OCCUPIED"}'})
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    out = tmp_path / "e.json"
    assert web.main(["--out", str(out)]) == 1
    assert "USERNAME_NOT_OCCUPIED" in capsys.readouterr().out and not out.exists() and tab.closed


def test_main_writes_loadable_export_without_printing_texts(tmp_path, monkeypatch, capsys):
    pages = {0: [raw(5, "SECRETTEXT five", out=True), {"_": "messageEmpty", "id": 4}, raw(3, "SECRETTEXT three")],
             3: [{"_": "messageService", "id": 2, "date": TS + 2, "peer_id": {"_": "peerUser", "user_id": 777},
                  "action": {"_": "messageActionChatCreate"}}, raw(1, "SECRETTEXT one", edit_date=TS + 60)]}
    tab = FakeTab(pages=pages)
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    out = tmp_path / "d" / "e.json"
    assert web.main(["--out", str(out), "--page-size", "3", "--pause", "0"]) == 0
    cap = capsys.readouterr()
    assert "messages: 4  service: 1  edited: 1  with_buttons: 0" in cap.out and "loader check OK n=3" in cap.out
    assert "SECRETTEXT" not in cap.out + cap.err and tab.closed
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert (doc["name"], doc["type"], doc["id"], doc["username"], doc["source"]) == \
        ("ShootingBets", "bot_chat", 777, "ShootingBetsBot", "telegram_web_cdp")
    assert [m["id"] for m in doc["messages"]] == [1, 2, 3, 5] and doc["messages"][3]["from"] == "me"
    got = load_export(str(out))
    assert [m.msg_id for m in got] == [1, 3, 5] and got[0].edited_utc == datetime.fromtimestamp(TS + 60, tz=timezone.utc)
    assert tab.offsets() == [0, 3, 1] and [m.msg_id for m in (load_export(str(out), include_service=True))] == [1, 2, 3, 5]
    # --limit keeps the oldest N
    assert web.main(["--out", str(out), "--page-size", "3", "--pause", "0", "--limit", "2"]) == 0
    assert [m["id"] for m in json.loads(out.read_text(encoding="utf-8"))["messages"]] == [1, 2]


def test_logout_calls_auth_logout_and_confirms(monkeypatch, capsys):
    tab = FakeTab()
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    assert web.main(["--logout"]) == 0
    joined = "\n".join(tab.calls)
    assert "auth.logOut" in joined and "location.reload()" in joined and "messages.getHistory" not in joined
    assert joined.index("auth.logOut") < joined.index("location.reload()")
    assert tab.logged_in is False and "logged out: True" in capsys.readouterr().out
    # already logged out: idempotent, no logOut call
    tab2 = FakeTab(logged_in=False)
    monkeypatch.setattr(web, "Tab", lambda port: tab2)
    assert web.main(["--logout"]) == 0 and not any("auth.logOut" in c for c in tab2.calls)
    # logOut rejected -> exit 1
    tab3 = FakeTab(errors={"auth.logOut": '{"type":"NOPE"}'})
    monkeypatch.setattr(web, "Tab", lambda port: tab3)
    assert web.main(["--logout"]) == 1


def test_no_tab_returns_4(monkeypatch, capsys):
    def boom(port):
        raise RuntimeError("no Telegram Web tab")
    monkeypatch.setattr(web, "Tab", boom)
    assert web.main([]) == 4 and "no Telegram Web tab" in capsys.readouterr().out
