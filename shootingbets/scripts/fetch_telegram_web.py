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

Exit codes: 0 ok | 1 API / page error (or --logout did not log out) | 2 Telegram Web not logged in |
3 written file failed the loader round-trip | 4 no Telegram Web tab on the DevTools port.
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import websocket

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_telegram as ft  # noqa: E402  (export_message, write_export, default output path)

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


class Tab:
    def __init__(self, port: int = 9222):
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                    f"http://127.0.0.1:{port}/json/list", timeout=10) as r:
                tabs = json.load(r)
        except (OSError, ValueError):
            raise RuntimeError("no Telegram Web tab") from None
        url = next((t.get("webSocketDebuggerUrl") for t in tabs if t.get("type") == "page"
                    and str(t.get("url", "")).startswith("https://web.telegram.org")), None)
        if not url:
            raise RuntimeError("no Telegram Web tab")
        self.ws, self._id = websocket.create_connection(url, timeout=60, suppress_origin=True), 0

    def close(self):
        self.ws.close()

    def evaluate(self, expr: str):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": "Runtime.evaluate", "params": {
            "expression": expr, "awaitPromise": True, "returnByValue": True}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                break
        if "error" in msg:
            raise RuntimeError(f"DevTools error: {msg['error'].get('message')}")
        res = msg["result"]
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
    try:
        return bool(api_call(tab, "users.getUsers", {"id": [{"_": "inputUserSelf"}]}, timeout_ms=10000))
    except RuntimeError:  # ApiError, or the page has no Telegram client yet
        return False


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
    return ft.export_message(adapt(m), "me" if m.get("out") else chat_name)


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
    if not is_logged_in(tab):
        print("already logged out")
        return 0
    api_call(tab, "auth.logOut")
    try:
        tab.evaluate("location.reload()")
    except RuntimeError:
        pass
    ok = _wait_ready(tab) and not is_logged_in(tab)
    print(f"logged out: {ok}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--chat", default="ShootingBetsBot")
    ap.add_argument("--out", default=str(ft.ROOT / "data" / "telegram" / "shootingbets_export.json"))
    ap.add_argument("--limit", type=int, default=None, help="only the oldest N messages (testing)")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--pause", type=float, default=0.5, help="seconds between history pages")
    ap.add_argument("--logout", action="store_true", help="log the web session out (no fetch) and exit")
    args = ap.parse_args(argv)
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
        doc = {"name": "ShootingBets", "type": "bot_chat", "id": chat["id"], "username": chat["username"],
               "source": "telegram_web_cdp", "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "messages": [export_raw(m, chat["first_name"]) for m in msgs]}
        return ft.write_export(doc, args.out)
    except (RuntimeError, websocket.WebSocketException, OSError) as exc:
        print(f"failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        tab.close()


if __name__ == "__main__":
    raise SystemExit(main())
