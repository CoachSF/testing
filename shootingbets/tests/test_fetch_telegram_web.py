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
AUTH_ERR = {"__error__": '{"code":401,"type":"AUTH_KEY_UNREGISTERED"}'}  # what the page returns for an unauthorised call


def u16(s):
    return len(s.encode("utf-16-le")) // 2


def raw(i, text="hello", **kw):
    return {"_": "message", "id": i, "date": TS + i, "message": text,
            "peer_id": {"_": "peerUser", "user_id": 777}, "from_id": {"_": "peerUser", "user_id": 777}, **kw}


class FakeTab:
    """Answers by substring of the JS expression; ``pages`` maps offset_id -> list of raw messages. DevTools calls and
    reopen() are recorded in ``calls`` too (prefixed DEVTOOLS / REOPEN) so the logout order can be asserted.
    ``relogin``: the reopened page is logged in again (storage not wiped); ``dc_fails``: help.getNearestDc is rejected."""

    def __init__(self, logged_in=True, pages=None, errors=None, relogin=False, dc_fails=False):
        self.logged_in, self.pages, self.errors, self.calls, self.closed = logged_in, pages or {}, errors or {}, [], False
        self.relogin, self.dc_fails = relogin, dc_fails

    def evaluate(self, expr):
        self.calls.append(expr)
        for key, err in self.errors.items():
            if key in expr:
                return json.dumps({"__error__": err})
        if "users.getUsers" in expr:
            return json.dumps([{"_": "user", "id": 1}]) if self.logged_in else json.dumps(AUTH_ERR)
        if "contacts.resolveUsername" in expr:
            return json.dumps(CHAT)
        if "messages.getHistory" in expr:
            return json.dumps(self.pages.get(int(re.search(r"offset_id: (\d+)", expr).group(1)), []))
        if "auth.logOut" in expr:
            self.logged_in = False
            return json.dumps({"_": "auth.loggedOut"})
        if "help.getNearestDc" in expr:
            return json.dumps({"__error__": '{"type":"NETWORK_DOWN"}'} if self.dc_fails else {"_": "nearestDc"})
        if "typeof (window.rootScope" in expr:
            return True
        raise AssertionError("unexpected expression: " + expr[:80])

    def call(self, method, params=None):
        self.calls.append(f"DEVTOOLS {method} {json.dumps(params)}")
        return {}

    def reopen(self, url):
        self.calls.append(f"REOPEN {url}")
        self.logged_in = self.relogin

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
    assert web.export_raw(raw(4, pFlags={"out": True}), "ShootingBets")["from"] == "me"     # the real TL form
    assert web.export_raw(raw(4, pFlags={}), "ShootingBets")["from"] == "ShootingBets"


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
        self.sent, self.replies, self.closed = [], list(replies), False

    def close(self):
        self.closed = True

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
    t.ws.replies.append({"id": 4, "result": {}})
    assert t.call("Storage.clearDataForOrigin", {"origin": "o", "storageTypes": "all"}) == {}
    assert t.ws.sent[-1] == {"id": 4, "method": "Storage.clearDataForOrigin",
                             "params": {"origin": "o", "storageTypes": "all"}}


def test_tab_reopen_closes_then_opens_a_new_tab_over_http(monkeypatch):
    seen = []

    class H(BaseHTTPRequestHandler):
        def _reply(self, body):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            seen.append(("GET", self.path))
            self._reply(json.dumps([{"type": "page", "id": "T1", "url": "https://web.telegram.org/k/",
                                     "webSocketDebuggerUrl": "ws://x/1"}]).encode() if self.path == "/json/list"
                        else b"Target is closing")

        def do_PUT(self):
            seen.append(("PUT", self.path))
            self._reply(json.dumps({"id": "T2", "type": "page", "webSocketDebuggerUrl": "ws://x/2"}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    made = []
    monkeypatch.setattr(web.websocket, "create_connection", lambda url, **kw: made.append(url) or FakeWS([]))
    for k in ("HTTP_PROXY", "http_proxy"):  # a dead proxy in the environment must not matter: DevTools is local
        monkeypatch.setenv(k, "http://127.0.0.1:9")
    w = importlib.util.module_from_spec(_spec)   # a fresh copy, imported AFTER the proxy variable was set
    _spec.loader.exec_module(w)
    try:
        t = w.Tab(srv.server_address[1])
        old_ws = t.ws
        assert (t.port, t.target_id) == (srv.server_address[1], "T1")
        t.reopen(w.START_URL)
    finally:
        srv.shutdown()
    assert seen == [("GET", "/json/list"), ("GET", "/json/close/T1"), ("PUT", "/json/new?https://web.telegram.org/k/")]
    assert made == ["ws://x/1", "ws://x/2"] and t.target_id == "T2" and old_ws.closed and t.ws is not old_ws


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
    pages = {0: [raw(5, "SECRETTEXT five", pFlags={"out": True}), {"_": "messageEmpty", "id": 4}, raw(3, "SECRETTEXT three")],
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
    # the raw TL messages are saved beside --out, atomically (no .tmp left), without the bot flag / access hash
    rawf = out.parent / "shootingbets_raw.json"
    saved = json.loads(rawf.read_text(encoding="utf-8"))
    assert saved["chat"] == {"id": 777, "username": "ShootingBetsBot", "first_name": "ShootingBets"}
    assert saved["fetched_utc"] == doc["fetched_utc"] and [m["id"] for m in saved["messages"]] == [1, 2, 3, 5]
    assert saved["messages"][3]["pFlags"] == {"out": True} and not list(out.parent.glob("*.tmp"))
    assert f"raw written: {rawf}" in cap.out
    # --from-raw: same export without any browser
    monkeypatch.setattr(web, "Tab", lambda port: pytest.fail("no browser expected"))
    out2 = tmp_path / "again" / "e.json"
    assert web.main(["--from-raw", str(rawf), "--out", str(out2)]) == 0
    assert json.loads(out2.read_text(encoding="utf-8")) == doc and "SECRETTEXT" not in capsys.readouterr().out
    assert web.main(["--from-raw", str(rawf), "--out", str(out2), "--limit", "2"]) == 0
    assert [m["id"] for m in json.loads(out2.read_text(encoding="utf-8"))["messages"]] == [1, 2]
    (tmp_path / "nochat.json").write_text('{"messages": []}', encoding="utf-8")
    capsys.readouterr()
    for bad in (tmp_path / "missing.json", tmp_path / "nochat.json"):
        assert web.main(["--from-raw", str(bad), "--out", str(out2)]) == 1
    assert "failed:" in capsys.readouterr().out
    # --limit keeps the oldest N, and does NOT overwrite the raw file unless --raw-out is explicit
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    assert web.main(["--out", str(out), "--page-size", "3", "--pause", "0", "--limit", "2"]) == 0
    assert [m["id"] for m in json.loads(out.read_text(encoding="utf-8"))["messages"]] == [1, 2]
    assert json.loads(rawf.read_text(encoding="utf-8")) == saved
    explicit = tmp_path / "r" / "x.json"
    assert web.main(["--out", str(out), "--raw-out", str(explicit), "--page-size", "3", "--pause", "0", "--limit", "2"]) == 0
    assert [m["id"] for m in json.loads(explicit.read_text(encoding="utf-8"))["messages"]] == [1, 2]


def test_is_logged_in_only_false_on_an_authorisation_error():
    assert web.is_logged_in(FakeTab()) is True
    for err in ('{"code":401,"type":"AUTH_KEY_UNREGISTERED"}', '{"type":"SESSION_REVOKED"}', '{"type":"USER_DEACTIVATED"}',
                '{"type":"X","code":401}'):
        assert web.is_logged_in(FakeTab(errors={"users.getUsers": err})) is False
    for err in ('{"type":"JS_TIMEOUT"}', '{"type":"FLOOD_WAIT_5","code":420}'):  # a hung / failing page is NOT 'logged out'
        with pytest.raises(web.ApiError, match="JS_TIMEOUT|FLOOD_WAIT_5"):
            web.is_logged_in(FakeTab(errors={"users.getUsers": err}))


def test_hung_page_is_a_failure_not_a_logged_out_verdict(monkeypatch, capsys):
    tab = FakeTab(errors={"users.getUsers": '{"type":"JS_TIMEOUT"}'})
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    assert web.main([]) == 1
    out = capsys.readouterr().out
    assert "JS_TIMEOUT" in out and "failed:" in out and web.NOT_LOGGED_IN not in out
    assert web.main(["--logout"]) == 1 and not any("auth.logOut" in c or c.startswith(("DEVTOOLS", "REOPEN")) for c in tab.calls)
    assert "failed:" in capsys.readouterr().out


def test_logout_sequence_and_proof(monkeypatch, capsys):
    tab = FakeTab()
    monkeypatch.setattr(web, "Tab", lambda port: tab)
    assert web.main(["--logout"]) == 0
    joined = "\n".join(tab.calls)
    steps = ["auth.logOut", "DEVTOOLS Storage.clearDataForOrigin", "REOPEN https://web.telegram.org/k/", "help.getNearestDc",
             "users.getUsers"]
    pos = [joined.index(s) for s in steps[:4]] + [joined.rindex(steps[4])]
    assert pos == sorted(pos) and "messages.getHistory" not in joined and "location.reload" not in joined
    assert json.loads(next(c for c in tab.calls if c.startswith("DEVTOOLS")).split(" ", 2)[2]) == \
        {"origin": "https://web.telegram.org", "storageTypes": "all"}
    assert tab.logged_in is False and tab.closed and "logged out: True" in capsys.readouterr().out
    # already logged out: idempotent, nothing but the login check
    tab2 = FakeTab(logged_in=False)
    monkeypatch.setattr(web, "Tab", lambda port: tab2)
    assert web.main(["--logout"]) == 0 and len(tab2.calls) == 1 and "already logged out" in capsys.readouterr().out
    # logOut rejected -> exit 1, storage untouched
    tab3 = FakeTab(errors={"auth.logOut": '{"type":"NOPE"}'})
    monkeypatch.setattr(web, "Tab", lambda port: tab3)
    assert web.main(["--logout"]) == 1 and not any(c.startswith(("DEVTOOLS", "REOPEN")) for c in tab3.calls)
    capsys.readouterr()
    # the reopened page still logs in (login keys survived) -> NOT 'logged out', exit 1, and it says what it saw
    tab4 = FakeTab(relogin=True)
    monkeypatch.setattr(web, "Tab", lambda port: tab4)
    assert web.main(["--logout"]) == 1
    out = capsys.readouterr().out
    assert "logged out: True" not in out and "logged out: False" in out and "still logged in: True" in out
    # the client does not answer after the reopen -> exit 1
    tab5 = FakeTab(dc_fails=True)
    monkeypatch.setattr(web, "Tab", lambda port: tab5)
    assert web.main(["--logout"]) == 1
    out = capsys.readouterr().out
    assert "logged out: False" in out and "getNearestDc ok: False" in out and "NETWORK_DOWN" in out


def test_no_tab_returns_4(monkeypatch, capsys):
    def boom(port):
        raise RuntimeError("no Telegram Web tab")
    monkeypatch.setattr(web, "Tab", boom)
    assert web.main([]) == 4 and "no Telegram Web tab" in capsys.readouterr().out
