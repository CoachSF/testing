#!/usr/bin/env python3
"""Pull the full history of one Telegram chat (default: the @ShootingBetsBot chat) with Telethon.

It logs in as YOUR user account (not a bot token - bots cannot read this chat) and writes the
history in the Telegram Desktop JSON shape that ``sbroi.telegram_export.load_export`` reads.

One-off setup
  1. Log in at https://my.telegram.org -> "API development tools" -> create an app, and copy
     ``api_id`` and ``api_hash``.
  2. Copy ``shootingbets/.env.example`` to ``shootingbets/.env`` and fill in TG_API_ID and
     TG_API_HASH. ``.env`` is gitignored - never commit it.

Run (from the ``shootingbets`` folder)
    .venv\\Scripts\\python.exe scripts\\fetch_telegram.py
    .venv\\Scripts\\python.exe scripts\\fetch_telegram.py --list-dialogs        # find the chat if the username differs
    .venv\\Scripts\\python.exe scripts\\fetch_telegram.py --chat "<name or @username>" --limit 50   # quick test

The first run asks, in the terminal, for your phone number, the login code Telegram sends you and
(if set) your 2FA password. Nothing typed there is logged or echoed by this script.

The ``.session`` file written next to the output (``data/telegram/shootingbets.session``) is a LOGIN
CREDENTIAL for your Telegram account: keep it private. It lives under the gitignored ``data/``
folder and ``*.session`` is also gitignored - never copy it into git or share it.

Fallback without the API: Telegram Desktop -> open the chat -> ... menu -> "Export chat history" ->
format "Machine-readable JSON" -> use the resulting ``result.json`` instead.

Exit codes: 2 = credentials missing, 3 = written file failed the loader round-trip check.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_LINK_TYPES = ("MessageEntityTextUrl", "MessageEntityUrl")
_TYPE_NAMES = {
    "MessageEntityTextUrl": "text_link", "MessageEntityUrl": "link", "MessageEntityBold": "bold",
    "MessageEntityItalic": "italic", "MessageEntityCode": "code", "MessageEntityPre": "pre",
    "MessageEntityMention": "mention", "MessageEntityHashtag": "hashtag",
}


def load_credentials(env_path) -> tuple[int, str]:
    vals = dotenv_values(env_path)  # {} when the file does not exist
    api_id, api_hash = (vals.get("TG_API_ID") or "").strip(), (vals.get("TG_API_HASH") or "").strip()
    if api_id.isdigit() and api_hash:
        return int(api_id), api_hash
    print(f"Telegram credentials missing or invalid in {env_path}.\n"
          "Create that file (copy .env.example) with TG_API_ID=<number> and TG_API_HASH=<string>, "
          "both from https://my.telegram.org -> API development tools.")
    raise SystemExit(2)


def _utc(d: datetime) -> datetime:
    return (d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d).astimezone(timezone.utc)


def _entity_type(e) -> str:
    name = type(e).__name__
    return _TYPE_NAMES.get(name) or name.removeprefix("MessageEntity").lower()


def _pieces(raw: str, entities) -> list[dict]:
    """Ordered pieces covering ``raw`` exactly once; entity offsets/lengths are UTF-16 units."""
    u = raw.encode("utf-16-le")
    total = len(u) // 2

    def sl(a: int, b: int) -> str:
        return u[2 * a:2 * b].decode("utf-16-le")

    def is_link(e) -> bool:
        return type(e).__name__ in _LINK_TYPES

    ents = [e for e in (entities or []) if getattr(e, "length", 0) > 0 and 0 <= e.offset < total]
    links = [e for e in ents if is_link(e)]
    # a link wins over any other entity overlapping it, so its href/url is never lost
    ents = [e for e in ents if is_link(e) or not any(
        e.offset < k.offset + k.length and k.offset < e.offset + e.length for k in links)]
    ents.sort(key=lambda e: (e.offset, not is_link(e)))
    out, cur = [], 0
    for e in ents:
        if e.offset < cur:  # overlap / nesting
            continue
        end = min(e.offset + e.length, total)
        if e.offset > cur:
            out.append({"type": "plain", "text": sl(cur, e.offset)})
        p = {"type": _entity_type(e), "text": sl(e.offset, end)}
        if p["type"] == "text_link":
            p["href"] = e.url
        out.append(p)
        cur = end
    if cur < total:
        out.append({"type": "plain", "text": sl(cur, total)})
    return out


def export_message(msg, sender_name: str | None) -> dict:
    raw = getattr(msg, "message", None) or ""
    pieces = _pieces(raw, getattr(msg, "entities", None))
    d = _utc(msg.date)
    sender_id = getattr(msg, "sender_id", None)
    out: dict = {
        "id": msg.id,
        "type": "service" if getattr(msg, "action", None) is not None else "message",
        "date": d.strftime("%Y-%m-%dT%H:%M:%S"),
        "date_unixtime": str(int(d.timestamp())),
    }
    edit = getattr(msg, "edit_date", None)
    if edit:
        e = _utc(edit)
        out["edited"] = e.strftime("%Y-%m-%dT%H:%M:%S")
        out["edited_unixtime"] = str(int(e.timestamp()))
    out["from"] = sender_name
    out["from_id"] = None if sender_id is None else f"user{sender_id}"
    if any(p["type"] != "plain" for p in pieces):
        out["text"] = [p["text"] if p["type"] == "plain" else p for p in pieces]
    else:
        out["text"] = raw
    out["text_entities"] = pieces
    out["text_plain"] = raw
    rows = getattr(getattr(msg, "reply_markup", None), "rows", None)
    if rows:
        out["inline_bot_buttons"] = [[_button(b) for b in getattr(r, "buttons", [])] for r in rows]
    reply = getattr(getattr(msg, "reply_to", None), "reply_to_msg_id", None)
    if reply is not None:
        out["reply_to_message_id"] = reply
    media = getattr(msg, "media", None)
    if media is not None:
        out["media_type"] = type(media).__name__
    return out


def _button(b) -> dict:
    t = getattr(b, "type", None) or b  # newer Telethon layers nest url/data under button.type
    url, data = getattr(t, "url", None), getattr(t, "data", None)
    if url:
        return {"type": "url", "text": getattr(b, "text", None), "data": url}
    if data is not None:
        return {"type": "callback", "text": getattr(b, "text", None),
                "data": bytes(data).decode("utf-8", errors="replace")}
    return {"type": "other", "text": getattr(b, "text", None), "data": None}


def _ascii(s) -> str:
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def _span(ts: int) -> str:
    d = datetime.fromtimestamp(ts, tz=timezone.utc)
    return f"{d:%Y-%m-%d %H:%M:%S} UTC / {d.astimezone(ZoneInfo('Australia/Melbourne')):%Y-%m-%d %H:%M:%S} Melbourne"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chat", default="ShootingBetsBot")
    ap.add_argument("--out", default=str(ROOT / "data" / "telegram" / "shootingbets_export.json"))
    ap.add_argument("--session", default=str(ROOT / "data" / "telegram" / "shootingbets"),
                    help="session file path without .session")
    ap.add_argument("--env", default=str(ROOT / ".env"))
    ap.add_argument("--limit", type=int, default=None, help="only the oldest N messages (testing)")
    ap.add_argument("--list-dialogs", nargs="?", const="shoot", default=None, metavar="SUBSTR",
                    help="list dialogs whose name/username contains SUBSTR (default 'shoot') and exit")
    args = ap.parse_args(argv)

    api_id, api_hash = load_credentials(args.env)
    from telethon.sync import TelegramClient  # imported here so tests need no network
    from telethon.utils import get_display_name

    session = Path(args.session)
    session.parent.mkdir(parents=True, exist_ok=True)
    with TelegramClient(str(session), api_id, api_hash) as client:
        if args.list_dialogs is not None:
            needle = args.list_dialogs.lower()
            for dlg in client.iter_dialogs():
                uname = getattr(dlg.entity, "username", None)
                if needle in (dlg.name or "").lower() or needle in (uname or "").lower():
                    kind = type(dlg.entity).__name__ + (" bot" if getattr(dlg.entity, "bot", False) else "")
                    print(f"{_ascii(dlg.name)} | {uname} | {dlg.id} | {kind}")
            return 0
        entity = client.get_entity(args.chat)
        messages = [export_message(m, get_display_name(m.sender) if m.sender is not None else None)
                    for m in client.iter_messages(entity, reverse=True, limit=args.limit)]
        doc = {"name": "ShootingBets", "type": "bot_chat", "id": entity.id,
               "username": getattr(entity, "username", None),
               "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "messages": messages}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, out)

    ts = [int(m["date_unixtime"]) for m in messages]
    print(f"written: {out}\n"
          f"messages: {len(messages)}  service: {sum(m['type'] == 'service' for m in messages)}  "
          f"edited: {sum('edited_unixtime' in m for m in messages)}  "
          f"with_buttons: {sum('inline_bot_buttons' in m for m in messages)}")
    if ts:
        print(f"first: {_span(min(ts))}\nlast:  {_span(max(ts))}")
    from sbroi.telegram_export import load_export
    n, want = len(load_export(str(out))), sum(m["type"] == "message" for m in messages)
    if n != want:
        print(f"loader check FAILED: loader returned {n}, expected {want}")
        return 3
    print(f"loader check OK n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
