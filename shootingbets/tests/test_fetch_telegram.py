"""Tests for scripts/fetch_telegram.py (pure functions only - no Telegram / network)."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from conftest import ROOT
from sbroi.telegram_export import load_export

_spec = importlib.util.spec_from_file_location("fetch_telegram", ROOT / "scripts" / "fetch_telegram.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)

DATE = datetime(2026, 10, 1, 3, 30, 15, tzinfo=timezone.utc)


# Fakes named exactly like the Telethon classes: the class name drives the type mapping.
class MessageEntityTextUrl:
    def __init__(self, offset, length, url):
        self.offset, self.length, self.url = offset, length, url


class MessageEntityUrl:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length


class MessageEntityBold:
    def __init__(self, offset, length):
        self.offset, self.length = offset, length


def u16(s):  # length in UTF-16 code units, as Telegram counts
    return len(s.encode("utf-16-le")) // 2


def span(raw, sub, cls, *extra):
    return cls(u16(raw[:raw.index(sub)]), u16(sub), *extra)


def msg(**kw):
    base = dict(id=1, date=DATE, edit_date=None, message="hello", entities=None, reply_markup=None,
                action=None, sender_id=42, reply_to=None, media=None)
    base.update(kw)
    return SimpleNamespace(**base)


def write_doc(tmp_path, messages):
    p = tmp_path / "export.json"
    p.write_text(json.dumps({"name": "ShootingBets", "type": "bot_chat", "id": 7, "messages": messages},
                            ensure_ascii=False), encoding="utf-8")
    return str(p)


def joined(d):
    return "".join(p["text"] for p in d["text_entities"])


def test_plain_message():
    d = ft.export_message(msg(), "Shooting Bets")
    assert d["text"] == "hello" and isinstance(d["text"], str)
    assert d["type"] == "message" and d["from"] == "Shooting Bets" and d["from_id"] == "user42"
    assert d["date"] == "2026-10-01T03:30:15"
    assert d["date_unixtime"] == str(int(DATE.timestamp()))
    assert "edited" not in d and "edited_unixtime" not in d
    assert "inline_bot_buttons" not in d and "media_type" not in d
    assert joined(d) == "hello"


def test_emoji_before_entities_and_roundtrip(tmp_path):
    raw = "\U0001F525 Arsenal v Chelsea - details here: https://example.com/odds/1 ok"
    assert u16(raw) > len(raw)  # the emoji really is 2 UTF-16 units
    ents = [span(raw, "details", MessageEntityTextUrl, "https://example.com/event/9"),
            span(raw, "https://example.com/odds/1", MessageEntityUrl)]
    btn = SimpleNamespace(text="Open", url="https://btn.example/x")
    cb = SimpleNamespace(text="Seen", data="seen:1".encode())
    markup = SimpleNamespace(rows=[SimpleNamespace(buttons=[btn, cb])])
    d = ft.export_message(msg(message=raw, entities=ents, reply_markup=markup), None)
    assert joined(d) == raw
    assert [p["type"] for p in d["text_entities"]] == ["plain", "text_link", "plain", "link", "plain"]
    assert d["text_entities"][1] == {"type": "text_link", "text": "details", "href": "https://example.com/event/9"}
    assert d["from"] is None
    assert d["inline_bot_buttons"] == [[
        {"type": "url", "text": "Open", "data": "https://btn.example/x"},
        {"type": "callback", "text": "Seen", "data": "seen:1"}]]
    m = load_export(write_doc(tmp_path, [d]))[0]
    assert m.text == raw
    assert "https://example.com/event/9" in m.links and "https://example.com/odds/1" in m.links
    assert {"text": "Open", "url": "https://btn.example/x"} in m.buttons


def test_buttons_new_telethon_layer_nests_url_under_type():
    class InlineButtonTypeUrl:
        url = "https://btn.example/new"

    class InlineButtonTypeCallback:
        data = b"cb:\xff"

    class InlineButtonTypeBuy:
        pass

    rows = [SimpleNamespace(buttons=[SimpleNamespace(text=t, type=ty()) for t, ty in (
        ("Open", InlineButtonTypeUrl), ("Cb", InlineButtonTypeCallback), ("Buy", InlineButtonTypeBuy))])]
    d = ft.export_message(msg(reply_markup=SimpleNamespace(rows=rows)), None)
    assert d["inline_bot_buttons"] == [[
        {"type": "url", "text": "Open", "data": "https://btn.example/new"},
        {"type": "callback", "text": "Cb", "data": "cb:\ufffd"},
        {"type": "other", "text": "Buy", "data": None}]]


def test_edited_message(tmp_path):
    edit = DATE + timedelta(minutes=5)
    d = ft.export_message(msg(edit_date=edit), None)
    assert d["edited"] == "2026-10-01T03:35:15" and d["edited_unixtime"] == str(int(edit.timestamp()))
    assert load_export(write_doc(tmp_path, [d]))[0].edited_utc == edit


def test_service_message_skipped_by_loader(tmp_path):
    svc = ft.export_message(msg(id=2, message=None, action=object()), None)
    assert svc["type"] == "service" and svc["text"] == "" and svc["text_plain"] == ""
    p = write_doc(tmp_path, [ft.export_message(msg(), None), svc])
    assert [m.msg_id for m in load_export(p)] == [1]
    assert [m.msg_id for m in load_export(p, include_service=True)] == [1, 2]


def test_load_credentials(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        ft.load_credentials(tmp_path / "nope.env")
    assert e.value.code == 2
    env = tmp_path / ".env"
    env.write_text("TG_API_ID=987654321\nTG_API_HASH=\n")
    with pytest.raises(SystemExit) as e:
        ft.load_credentials(env)
    assert e.value.code == 2
    assert "987654321" not in capsys.readouterr().out  # never echo credential values
    env.write_text("TG_API_ID=123456\nTG_API_HASH=deadbeefcafe\n")
    api_id, api_hash = ft.load_credentials(env)
    assert (api_id, api_hash) == (123456, "deadbeefcafe") and isinstance(api_id, int)


@pytest.mark.parametrize("bold_at", [(0, 29), (12, 8), (15, 4), (13, 3), (0, 4)])
def test_overlapping_entities_keep_href(bold_at):
    raw = "Bold with 🔥 a link inside it"  # the link sits at UTF-16 units 15..19
    link = span(raw, "link", MessageEntityTextUrl, "https://example.com/keep")
    assert (link.offset, link.length) == (15, 4)
    bold = MessageEntityBold(*bold_at)  # first four overlap the link; (0, 4) does not
    d = ft.export_message(msg(message=raw, entities=[bold, link]), None)
    assert joined(d) == raw
    assert [p.get("href") for p in d["text_entities"] if p["type"] == "text_link"] == ["https://example.com/keep"]
    assert ("bold" in [p["type"] for p in d["text_entities"]]) == (bold_at == (0, 4))
    assert all(isinstance(p, str) or "type" in p for p in d["text"])


def test_main_flow_with_fake_client(tmp_path, monkeypatch, capsys):
    """main() end to end against a fake client: oldest-first request, atomic write, summary, loader check."""
    from telethon import types as tt

    seen = {}
    mk = lambda i, minutes, **kw: msg(id=i, date=DATE + timedelta(minutes=minutes), **kw)  # noqa: E731
    sender = tt.User(id=42, first_name="Shooting", last_name="Bets")
    msgs = [mk(1, 0, sender=sender), mk(2, 5, message=None, action=object(), sender=None),
            mk(3, 9, edit_date=DATE + timedelta(minutes=30), sender=sender)]

    class FakeClient:
        def __init__(self, session, api_id, api_hash):
            seen.update(session=session, api_id=api_id)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get_entity(self, chat):
            seen["chat"] = chat
            return SimpleNamespace(id=777, username="ShootingBetsBot")

        def iter_messages(self, entity, reverse, limit):
            seen.update(reverse=reverse, limit=limit)
            return iter(msgs[:limit])

    monkeypatch.setitem(__import__("sys").modules, "telethon.sync", SimpleNamespace(TelegramClient=FakeClient))
    env = tmp_path / ".env"
    env.write_text("TG_API_ID=123456\nTG_API_HASH=sekrethash\n")
    out = tmp_path / "d" / "export.json"
    argv = ["--env", str(env), "--out", str(out), "--session", str(tmp_path / "s" / "sess")]
    assert ft.main(argv) == 0
    text = capsys.readouterr().out
    assert "loader check OK n=2" in text and "messages: 3  service: 1  edited: 1  with_buttons: 0" in text
    assert "2026-10-01 03:30:15 UTC / 2026-10-01 13:30:15 Melbourne" in text  # AEST +10: DST starts Sun 4 Oct 2026
    assert "2026-10-01 03:39:15 UTC / 2026-10-01 13:39:15 Melbourne" in text
    assert "sekrethash" not in text and "123456" not in text
    assert seen["reverse"] is True and seen["chat"] == "ShootingBetsBot" and seen["api_id"] == 123456
    assert (tmp_path / "s").is_dir() and not list(tmp_path.rglob("*.tmp"))
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["id"] == 777 and doc["username"] == "ShootingBetsBot" and doc["type"] == "bot_chat"
    assert [m["id"] for m in doc["messages"]] == [1, 2, 3] and doc["messages"][0]["from"] == "Shooting Bets"
    assert ft.main(argv + ["--limit", "1"]) == 0 and len(json.loads(out.read_text(encoding="utf-8"))["messages"]) == 1
