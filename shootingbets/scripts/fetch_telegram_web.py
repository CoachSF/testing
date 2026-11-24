#!/usr/bin/env python3
"""Pull one Telegram chat's history through an already logged-in Telegram Web tab (Chrome DevTools port).

No api_id / api_hash is needed: the script talks to the page https://web.telegram.org/k/ over the Chrome
DevTools protocol and calls the page's own MTProto client (``window.rootScope.managers.apiManager``).
It reads ONLY the named chat (contacts.resolveUsername + messages.getHistory) and writes it in the same
Telegram-Desktop-like JSON as scripts/fetch_telegram.py (read by sbroi.telegram_export.load_export).
Message texts, phone numbers and tokens are never printed; the access_hash stays inside the page.

Setup: start Chrome with --remote-debugging-port=9222, open https://web.telegram.org/k/ and log in by QR
(phone: Settings > Devices > Link Desktop Device). The web session is a FULL login to the account: when
done, log it out with ``--logout`` (or on the phone: Settings > Devices > terminate the session).

Run (from the ``shootingbets`` folder)
    .venv\\Scripts\\python.exe scripts\\fetch_telegram_web.py                      # whole chat
    .venv\\Scripts\\python.exe scripts\\fetch_telegram_web.py --limit 50           # oldest 50 only (test)
    .venv\\Scripts\\python.exe scripts\\fetch_telegram_web.py --logout             # log the web session out
    .venv\\Scripts\\python.exe scripts\\fetch_telegram_web.py --from-raw RAW.json  # rebuild the export, no browser

The raw TL messages are also saved (``--raw-out``, default ``shootingbets_raw.json`` beside ``--out``), so the export
can be rebuilt later without a login. With ``--limit`` and no explicit ``--raw-out`` the raw file is NOT written.

Exit codes: 0 ok | 1 API / page error (or --logout did not log out) | 2 Telegram Web not logged in |
3 written file failed the loader round-trip | 4 no Telegram Web tab on the DevTools port.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_telegram as ft  # noqa: E402  (export_message, write_export, default output path)

ORIGIN, START_URL = "https://web.telegram.org", "https://web.telegram.org/k/"
AUTH_ERRORS = ("AUTH_KEY_UNREGISTERED", "SESSION_REVOKED", "USER_DEACTIVATED", '"code":401')
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # DevTools is local: never via a proxy
NOT_LOGGED_IN = ("Telegram Web is not logged in: scan the QR code in the 'Telegram Web' tab "
                 "(phone: Settings > Devices > Link Desktop Device), then re-run.")
# Page-side wrapper: runs BODY (an async function body using inv(method, params)) and returns a JSON string;
# a rejected call comes back as {"__error__": ...}. Bytes -> {"__bytes__": [...]}, bigint -> string.
_JS = """(async () => {
  const rep = (k, v) => typeof v === 'bigint' ? v.toString() : v instanceof Uint8Array ? {__bytes__: Array.from(v)}
    : v instanceof ArrayBuffer ? {__bytes__: Array.from(new Uint8Array(v))} : v;
  const inv = (m, p) => Promise.race([window.rootScope.managers.apiManager.invokeApi(m, p),
    new Promise((_, rej) => setTimeout(() => rej({type: 'JS_TIMEOUT'}), __MS__))]);
  try { return JSON.stringify(await (async () => { __BODY__ })(), rep); }
  catch (e) {
    let s; try { s = JSON.stringify(e, e && typeof e === 'object' ? Object.getOwnPropertyNames(e) : undefined); }
    catch (_) { s = String(e); }
    return JSON.stringify({__error__: String(s).slice(0, 300)});
  }
})()"""


class ApiError(RuntimeError):
    """A Telegram call was rejected inside the page."""


def _http(port: int, path: str, method: str = "GET", timeout: float = 10) -> bytes:
    with _OPENER.open(urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method), timeout=timeout) as r:
        return r.read()


class Tab:
    def __init__(self, port: int = 9222):
        self.port = port
        try:
            tabs = json.loads(_http(port, "/json/list"))
        except (OSError, ValueError):
            raise RuntimeError("no Telegram Web tab") from None
        t = next((t for t in tabs if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
                  and str(t.get("url", "")).startswith(ORIGIN)), None)
        if not t:
            raise RuntimeError("no Telegram Web tab")
        self.connect(t)

    def connect(self, target: dict):
        """Attach to a /json target dict (``id`` + ``webSocketDebuggerUrl``)."""
        self.target_id = target.get("id")
        self.ws, self._id = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=60, suppress_origin=True), 0

    def close(self):
        self.ws.close()

    def reopen(self, url: str):
        """Close this tab (GET /json/close/<id>) and open ``url`` in a new one (PUT /json/new?<url>); attach to it."""
        try:
            self.ws.close()
        except (websocket.WebSocketException, OSError):
            pass
        _http(self.port, f"/json/close/{self.target_id}")
        self.connect(json.loads(_http(self.port, f"/json/new?{url}", method="PUT")))

    def call(self, method: str, params: dict | None = None) -> dict:
        """One DevTools protocol call (any domain); returns its ``result`` dict."""
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params or {}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                break
        if "error" in msg:
            raise RuntimeError(f"DevTools error: {msg['error'].get('message')}")
        return msg["result"]

    def evaluate(self, expr: str):
        res = self.call("Runtime.evaluate", {"expression": expr, "awaitPromise": True, "returnByValue": True})
        if "exceptionDetails" in res:
            d = res["exceptionDetails"]
            raise RuntimeError((d.get("exception") or {}).get("description") or d.get("text") or "page exception")
        return res["result"].get("value")


def _run(tab, body: str, timeout_ms: int = 45000):
    """Run a JS body in the page; return its parsed JSON result or raise ApiError."""
    value = tab.evaluate(_JS.replace("__MS__", str(timeout_ms)).replace("__BODY__", body))
    out = json.loads(value) if value else None
    if isinstance(out, dict) and "__error__" in out:
        raise ApiError(out["__error__"])
    return out


def api_call(tab, method: str, params=None, timeout_ms: int = 45000):
    return _run(tab, f"return await inv({json.dumps(method)}, {json.dumps(params or {})});", timeout_ms)


def is_logged_in(tab) -> bool:
    """True when users.getUsers(self) works; False ONLY on an authorisation error. A hung page (JS_TIMEOUT) or any
    other failure propagates: it must not be mistaken for 'logged out'."""
    try:
        api_call(tab, "users.getUsers", {"id": [{"_": "inputUserSelf"}]}, timeout_ms=10000)
    except ApiError as exc:
        if any(k in str(exc) for k in AUTH_ERRORS):
            return False
        raise
    return True


def resolve_chat(tab, username: str) -> dict:
    """Resolve a user/bot by username; its input peer (with access_hash) is kept in window.__sb_peer."""
    return _run(tab, f"""
      const r = await inv('contacts.resolveUsername', {{username: {json.dumps(username.lstrip('@'))}}});
      const uid = String((r.peer || {{}}).user_id);
      const u = (r.users || []).find(x => String(x.id) === uid);
      if (!u) throw {{type: 'NOT_A_USER'}};
      window.__sb_peer = {{_: 'inputPeerUser', user_id: u.id, access_hash: u.access_hash}};
      return {{id: u.id, username: u.username, first_name: u.first_name, bot: !!u.bot}};""")


def history_page(tab, offset_id: int, limit: int) -> list:
    """One messages.getHistory page for the resolved chat, newest first."""
    return _run(tab, f"""return (await inv('messages.getHistory', {{peer: window.__sb_peer,
      offset_id: {int(offset_id)}, offset_date: 0, add_offset: 0, limit: {int(limit)}, max_id: 0, min_id: 0,
      hash: 0}})).messages;""") or []


def fetch_all(tab, limit=None, page_size: int = 100, pause: float = 0.5) -> list:
    """Whole history, oldest first, de-duplicated, messageEmpty dropped; ``limit`` keeps the OLDEST N."""
    seen, msgs, offset = set(), {}, 0
    while True:
        page = history_page(tab, offset, page_size)
        fresh = {m["id"] for m in page} - seen
        if not fresh:  # empty page, or a page that repeats what we already have
            break
        seen |= fresh
        msgs.update({m["id"]: m for m in page if m.get("_") != "messageEmpty"})
        offset = min(m["id"] for m in page)
        time.sleep(pause)
    out = [msgs[i] for i in sorted(msgs)]
    return out if limit is None else out[:limit]


def _obj(ctor: str, **kw):
    """Instance of a class named like the Telethon type (first letter upper-cased): export_message keys off it."""
    return type(ctor[:1].upper() + ctor[1:], (SimpleNamespace,), {})(**kw)


def _button(b: dict):
    data = b.get("data")
    return SimpleNamespace(text=b.get("text"), url=b.get("url"),
                           data=bytes(data["__bytes__"]) if isinstance(data, dict) else None)


def adapt(m: dict):
    """Raw TL message dict -> duck-typed object understood by fetch_telegram.export_message."""
    def when(ts):
        return datetime.fromtimestamp(ts, tz=timezone.utc) if ts else None
    rows = [SimpleNamespace(buttons=[_button(b) for b in row.get("buttons", [])])
            for row in (m.get("reply_markup") or {}).get("rows", [])]
    media, reply = m.get("media"), m.get("reply_to")
    return SimpleNamespace(
        id=m["id"], date=when(m.get("date")), edit_date=when(m.get("edit_date")), message=m.get("message"),
        entities=[_obj(e["_"], offset=e["offset"], length=e["length"], url=e.get("url"))
                  for e in m.get("entities") or []],
        reply_markup=SimpleNamespace(rows=rows) if rows else None,
        action=True if m.get("_") == "messageService" else None,
        sender_id=(m.get("from_id") or {}).get("user_id") or (m.get("peer_id") or {}).get("user_id"),
        reply_to=SimpleNamespace(reply_to_msg_id=reply.get("reply_to_msg_id")) if reply else None,
        media=_obj(media["_"]) if media else None)


def export_raw(m: dict, chat_name: str) -> dict:
    return ft.export_message(adapt(m), "me" if m.get("out") or (m.get("pFlags") or {}).get("out") else chat_name)


def build_doc(chat: dict, msgs: list, fetched_utc: str) -> dict:
    return {"name": "ShootingBets", "type": "bot_chat", "id": chat["id"], "username": chat["username"],
            "source": "telegram_web_cdp", "fetched_utc": fetched_utc,
            "messages": [export_raw(m, chat["first_name"]) for m in msgs]}


def save_raw(path, chat: dict, msgs: list, fetched_utc: str) -> None:
    """Atomically write the raw TL messages (the one thing that cannot be re-fetched after logout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"chat": {k: chat.get(k) for k in ("id", "username", "first_name")},
                               "fetched_utc": fetched_utc, "messages": msgs}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)


def _wait_ready(tab, tries: int = 15) -> bool:
    for _ in range(tries):
        time.sleep(2)
        try:
            if tab.evaluate("typeof (window.rootScope && window.rootScope.managers && "
                            "window.rootScope.managers.apiManager.invokeApi) === 'function'"):
                return True
        except (RuntimeError, websocket.WebSocketException, OSError):
            pass
    return False


def logout(tab) -> int:
    """Server-side auth.logOut, then wipe the origin's storage and reopen the page, then PROVE it: the client answers
    (help.getNearestDc works without a login) and users.getUsers(self) now fails with an authorisation error."""
    if not is_logged_in(tab):
        print("already logged out")
        return 0
    api_call(tab, "auth.logOut")
    tab.call("Storage.clearDataForOrigin", {"origin": ORIGIN, "storageTypes": "all"})
    tab.reopen(START_URL)
    ready, dc_ok, logged_in = _wait_ready(tab), False, None
    try:
        api_call(tab, "help.getNearestDc", timeout_ms=15000)
        dc_ok = True
        logged_in = is_logged_in(tab)
    except RuntimeError as exc:
        print(f"verification error: {type(exc).__name__}: {exc}")
    ok = dc_ok and logged_in is False
    print(f"logged out: {ok}" + ("" if ok else f" (page ready: {ready}, getNearestDc ok: {dc_ok}, still logged in: {logged_in})"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--chat", default="ShootingBetsBot")
    ap.add_argument("--out", default=str(ft.ROOT / "data" / "telegram" / "shootingbets_export.json"))
    ap.add_argument("--raw-out", default=None,
                    help="raw TL messages JSON (default: shootingbets_raw.json beside --out; not written with --limit "
                         "unless given explicitly)")
    ap.add_argument("--from-raw", default=None, metavar="PATH", help="rebuild the export from a --raw-out file (no browser)")
    ap.add_argument("--limit", type=int, default=None, help="only the oldest N messages (testing)")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between history pages")
    ap.add_argument("--logout", action="store_true", help="log the web session out (no fetch) and exit")
    args = ap.parse_args(argv)
    if args.from_raw:
        try:
            raw = json.loads(Path(args.from_raw).read_text(encoding="utf-8"))
            msgs = sorted(raw["messages"], key=lambda m: m["id"])[:args.limit]
            doc = build_doc(raw["chat"], msgs, raw.get("fetched_utc") or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"failed: {type(exc).__name__}: {exc}")
            return 1
        return ft.write_export(doc, args.out)
    try:
        tab = Tab(args.port)
    except RuntimeError as exc:
        print(f"{exc}: open https://web.telegram.org/k/ in the Chrome started with "
              f"--remote-debugging-port={args.port}")
        return 4
    try:
        if args.logout:
            return logout(tab)
        if not is_logged_in(tab):
            print(NOT_LOGGED_IN)
            return 2
        chat = resolve_chat(tab, args.chat)
        msgs = fetch_all(tab, args.limit, args.page_size, args.pause)
        fetched = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if args.raw_out or not args.limit:
            raw_out = args.raw_out or str(Path(args.out).with_name("shootingbets_raw.json"))
            save_raw(raw_out, chat, msgs, fetched)
            print(f"raw written: {raw_out}")
        return ft.write_export(build_doc(chat, msgs, fetched), args.out)
    except (RuntimeError, websocket.WebSocketException, OSError) as exc:
        print(f"failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        tab.close()


if __name__ == "__main__":
    raise SystemExit(main())
