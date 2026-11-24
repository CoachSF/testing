"""Load a Telegram Desktop chat export (Machine-readable JSON, ``result.json``).

Notes on the format:
  * ``date`` is the *exporting computer's local time* without an offset;
    ``date_unixtime`` is UTC epoch seconds - we always use the latter.
  * ``text`` is either a plain string or a list mixing strings and entity dicts
    ({"type": "bold"|"link"|"text_link"|..., "text": "...", "href": "..."}).
  * Edited messages carry ``edited``/``edited_unixtime``; the export holds the
    *final* text only.
  * A full-account export nests chats under {"chats": {"list": [...]}}.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass
class TgMessage:
    chat_name: str
    chat_id: Any
    msg_id: int
    ts_utc: datetime
    edited_utc: datetime | None
    sender: str | None
    sender_id: str | None
    text: str
    links: list[str] = field(default_factory=list)
    buttons: list[dict] = field(default_factory=list)
    reply_to: int | None = None
    forwarded_from: str | None = None
    raw_type: str = "message"


def _flatten_text(t: Any) -> tuple[str, list[str]]:
    links: list[str] = []
    if t is None:
        return "", links
    if isinstance(t, str):
        return t, links
    parts: list[str] = []
    for piece in t:
        if isinstance(piece, str):
            parts.append(piece)
        elif isinstance(piece, dict):
            txt = piece.get("text", "")
            parts.append(txt if isinstance(txt, str) else _flatten_text(txt)[0])
            href = piece.get("href")
            if href:
                links.append(href)
            if piece.get("type") in ("link", "url") and isinstance(txt, str):
                links.append(txt)
    return "".join(parts), links


def _ts(unix_s: Any, fallback_local: str | None) -> datetime:
    if unix_s not in (None, ""):
        return datetime.fromtimestamp(int(unix_s), tz=timezone.utc)
    # Older exports lack *_unixtime; the local string has no offset -> treat as naive.
    if fallback_local:
        return datetime.fromisoformat(fallback_local)
    raise ValueError("message without a timestamp")


def _buttons(m: dict) -> list[dict]:
    out = []
    markup = m.get("inline_bot_buttons") or (m.get("reply_markup") or {}).get("inline_keyboard")
    if not markup:
        return out
    for row in markup:
        for b in row if isinstance(row, list) else [row]:
            if isinstance(b, dict):
                out.append({"text": b.get("text"), "url": b.get("url") or b.get("data")})
    return out


def _iter_chats(doc: dict) -> Iterable[dict]:
    if "messages" in doc:
        yield doc
        return
    chats = (doc.get("chats") or {}).get("list") or []
    for c in chats:
        if isinstance(c, dict) and "messages" in c:
            yield c


def load_export(path: str, chat_name_contains: str | None = None,
                include_service: bool = False) -> list[TgMessage]:
    with open(path, "rb") as fh:
        doc = json.loads(fh.read().decode("utf-8-sig"))
    out: list[TgMessage] = []
    for chat in _iter_chats(doc):
        cname = chat.get("name") or ""
        if chat_name_contains and chat_name_contains.lower() not in cname.lower():
            continue
        for m in chat.get("messages", []):
            mtype = m.get("type", "message")
            if mtype != "message" and not include_service:
                continue
            text, links = _flatten_text(m.get("text"))
            if not text and m.get("text_entities"):
                text, links = _flatten_text(m.get("text_entities"))
            edited = None
            if m.get("edited_unixtime") or m.get("edited"):
                try:
                    edited = _ts(m.get("edited_unixtime"), m.get("edited"))
                except ValueError:
                    edited = None
            fwd = m.get("forwarded_from")
            out.append(TgMessage(
                chat_name=cname, chat_id=chat.get("id"), msg_id=int(m.get("id")),
                ts_utc=_ts(m.get("date_unixtime"), m.get("date")), edited_utc=edited,
                sender=m.get("from"), sender_id=m.get("from_id"), text=text,
                links=links, buttons=_buttons(m),
                reply_to=m.get("reply_to_message_id"),
                forwarded_from=fwd if isinstance(fwd, str) else None,
                raw_type=mtype,
            ))
    out.sort(key=lambda x: (x.ts_utc.timestamp() if x.ts_utc.tzinfo else x.ts_utc.timestamp(), x.msg_id))
    return out


def chats_summary(path: str) -> list[dict]:
    """List chats in an export with message counts and date span (to pick the right one)."""
    with open(path, "rb") as fh:
        doc = json.loads(fh.read().decode("utf-8-sig"))
    res = []
    for chat in _iter_chats(doc):
        msgs = [m for m in chat.get("messages", []) if m.get("type") == "message"]
        ts = [int(m["date_unixtime"]) for m in msgs if m.get("date_unixtime")]
        res.append({
            "name": chat.get("name"), "type": chat.get("type"), "id": chat.get("id"),
            "n_messages": len(msgs),
            "first_utc": datetime.fromtimestamp(min(ts), tz=timezone.utc).isoformat() if ts else None,
            "last_utc": datetime.fromtimestamp(max(ts), tz=timezone.utc).isoformat() if ts else None,
        })
    return res
