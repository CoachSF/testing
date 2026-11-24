"""Tests for scripts/fetch_betfair_pro_dap.py - synthetic tars only (conftest T0 = 2026-09-21T13:33:20Z)."""
import bz2
import importlib.util
import io
import json
import tarfile
import threading
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import MIN, ROOT, T0, md, to_bytes
from sbroi.betfair_files import load_markets

_spec = importlib.util.spec_from_file_location("fetch_betfair_pro_dap", ROOT / "scripts" / "fetch_betfair_pro_dap.py")
dap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dap)

DAY = 24 * 60 * MIN
MONTH = "2026-09"
SECRET = "https://example.invalid/pro/2026_09_ProSoccer.tar?X-Amz-Signature=SECRET-TOKEN-123"


def member(mid, market_type="MATCH_ODDS", open_ms=T0 + 120 * MIN, event="34000001", with_def=True, plain=False):
    lines = []
    if with_def:
        d = md(market_type=market_type, open_ms=open_ms)
        d["eventId"] = event
        lines.append({"op": "mcm", "clk": "1", "pt": T0 - 60 * MIN, "mc": [{"id": mid, "marketDefinition": d}]})
    lines.append({"op": "mcm", "clk": "2", "pt": T0 - 30 * MIN, "mc": [{"id": mid, "rc": [{"id": 1001, "ltp": 2.5}]}]})
    raw = to_bytes(lines)
    return raw if plain else bz2.compress(raw)


# (tar member name, bytes, kept?)  - window used below: --from 2026-09-21 --to 2026-09-21, pad 3 => paths 09-18..09-24
MEMBERS = [
    ("PRO/2026/Sep/21/34000001/1.300000001.bz2", member("1.300000001", event="34000001"), True),
    ("PRO/2026/Sep/21/34000002/1.300000002.bz2", member("1.300000002", "CORRECT_SCORE", event="34000002"), False),
    # path date inside the pad window, but marketTime is 09-24 -> outside --from/--to
    ("PRO/2026/Sep/23/34000003/1.300000003.bz2", member("1.300000003", open_ms=T0 + 3 * DAY, event="34000003"), False),
    # in-window market, but the path date is far outside -> must be skipped by the prefilter, unread
    ("PRO/2026/Jan/5/34000004/1.300000004.bz2", member("1.300000004", event="34000004"), False),
    ("misc/1.300000005.bz2", member("1.300000005", event="34000005"), True),  # unparseable path -> peek
    ("PRO/2026/Sep/21/34000006/1.300000006.bz2", member("1.300000006", with_def=False, event="34000006"), False),
    ("PRO/2026/Sep/22/34000007/1.300000007", member("1.300000007", event="34000007", plain=True), True),  # not bz2
]
KEPT = {f"1.30000000{n}" for n in (1, 5, 7)}
KEPT_FILES = {"2026-09-21/34000001/1.300000001.bz2", "2026-09-21/34000005/1.300000005.bz2",
              "2026-09-21/34000007/1.300000007"}


@pytest.fixture
def tar_path(tmp_path):
    p = tmp_path / "2026_09_ProSoccer.tar"
    with tarfile.open(p, "w") as tf:
        d = tarfile.TarInfo("PRO/2026/Sep/21/")  # a directory entry: must be ignored
        d.type = tarfile.DIRTYPE
        tf.addfile(d)
        for name, raw, _ in MEMBERS:
            ti = tarfile.TarInfo(name)
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
    return p


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(dap.time, "sleep", sleeps.append)
    return sleeps


def args(tar, out, *extra):  # prefilter is opt-in: these tests enable it explicitly
    return ["--source", str(tar), "--month", MONTH, "--out-dir", str(out), "--from", "2026-09-21",
            "--to", "2026-09-21", "--min-free-gb", "0", "--path-pad-days", "3", *extra]


def data_files(out):
    return {p.relative_to(out).as_posix() for p in out.rglob("*")
            if p.is_file() and not p.name.startswith(("manifest_", "_done_"))}


def manifest_rows(out):
    return [json.loads(ln) for ln in (out / f"manifest_{MONTH}.jsonl").read_text(encoding="utf-8").splitlines()]


def cfg(out, pad_days=3, **kw):
    return dap.Cfg(out, date(2026, 9, 21), date(2026, 9, 21), frozenset({"MATCH_ODDS"}), pad_days=pad_days,
                   min_free=0, **kw)


def test_extract_end_to_end(tar_path, tmp_path, capsys):
    out = tmp_path / "out"
    assert dap.main(args(tar_path, out)) == 0
    assert data_files(out) == KEPT_FILES
    by_name = {n: raw for n, raw, _ in MEMBERS}
    for rel in KEPT_FILES:  # byte-identical to the tar member, still bz2
        src = next(raw for n, raw in by_name.items() if n.rsplit("/", 1)[-1] == rel.rsplit("/", 1)[-1])
        assert (out / rel).read_bytes() == src
    rows = manifest_rows(out)
    assert {r["market_id"] for r in rows} == KEPT and {r["saved"] for r in rows} == KEPT_FILES
    assert {r["member"] for r in rows} == {n for n, _, k in MEMBERS if k}
    assert all(r["market_type"] == "MATCH_ODDS" and r["country_code"] == "GB" and r["bytes"] > 0 for r in rows)
    done = json.loads((out / f"_done_{MONTH}.json").read_text())
    assert (done["members_seen"], done["in_path_window"], done["kept"], done["already"], done["no_def"]) == (7, 6, 3, 0, 1)
    assert done["kept_bytes"] == sum(len(raw) for _, raw, k in MEMBERS if k)
    assert done["bytes_read"] == tar_path.stat().st_size
    assert done["window"] == {"from": "2026-09-21", "to": "2026-09-21"} and "finished_utc" in done
    assert done["path_pad_days"] == 3
    assert set(load_markets(str(out))) == KEPT  # the existing loader reads the saved tree
    assert not list(out.rglob("*.part"))
    assert "https://" not in capsys.readouterr().out


def test_prefilter_is_off_by_default(tmp_path, capsys):
    """Real PRO tars: folder date != marketTime (PRO/2026/Nov/3/... holds a market that starts on 09-21)."""
    tar = tmp_path / "far.tar"
    far = "PRO/2026/Nov/3/36202316/1.264004005.bz2"
    raw = member("1.264004005", event="36202316")  # marketTime 2026-09-21T15:33Z, inside the window
    with tarfile.open(tar, "w") as tf:
        ti = tarfile.TarInfo(far)
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))
    base = ["--source", str(tar), "--month", MONTH, "--from", "2026-09-21", "--to", "2026-09-21", "--min-free-gb", "0"]
    out = tmp_path / "out"
    assert dap.main([*base, "--out-dir", str(out)]) == 0  # no --path-pad-days
    assert data_files(out) == {"2026-09-21/36202316/1.264004005.bz2"}
    assert (out / "2026-09-21/36202316/1.264004005.bz2").read_bytes() == raw
    marker = out / f"_done_{MONTH}.json"
    done = json.loads(marker.read_text())
    assert (done["members_seen"], done["in_path_window"], done["kept"]) == (1, 1, 1)
    assert done["path_pad_days"] is None and '"path_pad_days": null' in marker.read_text()
    # opting in drops it - the very risk the default avoids
    assert dap.main([*base, "--out-dir", str(tmp_path / "opt"), "--path-pad-days", "3"]) == 0
    assert data_files(tmp_path / "opt") == set()
    # the done-marker compares window + market types only, not the pad setting
    capsys.readouterr()
    assert dap.main([*base, "--out-dir", str(out), "--path-pad-days", "3"]) == 0
    assert "already done" in capsys.readouterr().out


def test_rerun_is_idempotent(tar_path, tmp_path, capsys):
    out = tmp_path / "out"
    assert dap.main(args(tar_path, out)) == 0
    stamp = {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()}
    capsys.readouterr()
    assert dap.main(args(tar_path, out)) == 0  # done-marker short-circuits
    assert "already done" in capsys.readouterr().out
    assert {p: p.stat().st_mtime_ns for p in out.rglob("*") if p.is_file()} == stamp

    (out / f"_done_{MONTH}.json").unlink()  # no marker: files are re-seen, not rewritten
    assert dap.main(args(tar_path, out)) == 0
    done = json.loads((out / f"_done_{MONTH}.json").read_text())
    assert (done["kept"], done["already"]) == (0, 3)
    assert len(manifest_rows(out)) == 3
    assert all(p.stat().st_mtime_ns == stamp[p] for p in stamp if p.name not in (f"_done_{MONTH}.json",)
               and not p.name.startswith("manifest_"))

    # a different window must NOT short-circuit
    assert dap.main(["--source", str(tar_path), "--month", MONTH, "--out-dir", str(out), "--from", "2026-09-21",
                     "--to", "2026-09-22", "--min-free-gb", "0"]) == 0
    assert json.loads((out / f"_done_{MONTH}.json").read_text())["window"]["to"] == "2026-09-22"


def test_resume_adds_manifest_row_missing_for_existing_file(tar_path, tmp_path):
    out = tmp_path / "out"
    assert dap.main(args(tar_path, out)) == 0
    man = out / f"manifest_{MONTH}.jsonl"
    lines = man.read_text(encoding="utf-8").splitlines()
    man.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # crash between file save and row
    (out / f"_done_{MONTH}.json").unlink()
    assert dap.main(args(tar_path, out)) == 0
    assert len(manifest_rows(out)) == 3


def test_truncated_tar(tar_path, tmp_path, no_sleep):
    out = tmp_path / "out"
    size = tar_path.stat().st_size
    cut = tmp_path / "cut.tar"
    with tarfile.open(tar_path) as tf:
        mid = tf.getmembers()[4].offset_data + 10  # inside the 4th file member's data
    cut.write_bytes(tar_path.read_bytes()[:mid])
    # (a) as --source: tar breaks mid-member
    assert dap.main(args(cut, out)) == 3
    assert not (out / f"_done_{MONTH}.json").exists()
    # (b) cut file, ORIGINAL size expected: stream ends early -> 8 reconnects with 5,10,20,40,60.. backoff -> 3
    no_sleep.clear()
    with pytest.raises(dap.Stop) as e:
        dap.extract_tar(dap.local_opener(cut), size, MONTH, cfg(out))
    assert e.value.code == 3 and "TRUNCATED" in str(e.value)
    assert no_sleep == [5, 10, 20, 40, 60, 60, 60, 60]
    # (c) cut exactly at the end-of-tar marker: tar looks complete but byte count is short -> 3
    short = tmp_path / "short.tar"
    short.write_bytes(tar_path.read_bytes()[:size - 2048])
    with pytest.raises(dap.Stop) as e:
        dap.extract_tar(dap.local_opener(short), size, MONTH, cfg(out))
    assert e.value.code == 3
    assert not (out / f"_done_{MONTH}.json").exists()
    # (d) stream longer than the size the urls row promised -> byte-count mismatch -> 3
    with pytest.raises(dap.Stop) as e:
        dap.extract_tar(dap.local_opener(tar_path), size - 4096, MONTH, cfg(out))
    assert e.value.code == 3 and "expected" in str(e.value)


class Flaky:
    """First stream: delivers ``n`` bytes then fails (mode 'raise') or reports EOF early (mode 'short')."""

    def __init__(self, data, n, mode):
        self.f, self.left, self.mode = io.BytesIO(data), n, mode

    def read(self, k):
        if self.left <= 0:
            if self.mode == "raise":
                raise ConnectionError("boom https://example.invalid/?sig=SECRET")
            return b""
        b = self.f.read(min(k, self.left))
        self.left -= len(b)
        return b


@pytest.mark.parametrize("mode", ["raise", "short"])
def test_resumable_reader_reconnects_at_offset(mode, no_sleep):
    data = bytes(range(256)) * 40
    calls = []

    def opener(offset):
        calls.append(offset)
        return Flaky(data[offset:], 3000, mode) if len(calls) == 1 else io.BytesIO(data[offset:])

    r = dap.ResumableReader(opener, len(data))
    got = b""
    while chunk := r.read(1000):
        got += chunk
    assert got == data and r.pos == len(data)
    assert calls == [0, 3000] and no_sleep == [5]


def test_reader_gives_up_after_max_retries_without_leaking_secret(no_sleep):
    def opener(offset):
        raise ConnectionError("failed for " + SECRET)

    r = dap.ResumableReader(opener, 100)
    with pytest.raises(dap.Stop) as e:
        r.read(10)
    assert e.value.code == 3 and "SECRET" not in str(e.value) and e.value.__suppress_context__
    assert len(no_sleep) == 8


def test_url_expired_is_not_retried(tmp_path, no_sleep):
    calls = []

    def opener(offset):
        calls.append(offset)
        raise dap.UrlExpired("HTTP 403")

    with pytest.raises(dap.UrlExpired):
        dap.extract_tar(opener, 1000, MONTH, cfg(tmp_path / "out"))
    assert calls == [0] and no_sleep == []


def test_main_url_expired_names_months_and_hides_url(tmp_path, monkeypatch, capsys):
    urls = tmp_path / "urls.jsonl"
    urls.write_text("\n".join(json.dumps({"path": f"Soccer/{m}.tar", "month": m, "size": 5, "url": SECRET,
                                          "expires_utc": "2026-11-25T00:00:00Z"}) for m in ("2026-09", "2026-10")))

    def raising(url):
        def opener(offset):
            raise dap.UrlExpired("HTTP 403")
        return opener

    monkeypatch.setattr(dap, "http_opener", raising)
    rc = dap.main(["--urls", str(urls), "--out-dir", str(tmp_path / "o"), "--from", "2026-09-21",
                   "--to", "2026-09-21", "--min-free-gb", "0"])
    cap = capsys.readouterr()
    assert rc == 4 and "2026-09" in cap.out and "2026-10" in cap.out
    assert "SECRET" not in cap.out + cap.err and "example.invalid" not in cap.out + cap.err


def test_guards(tar_path, tmp_path):
    out = tmp_path / "out"
    assert dap.main(args(tar_path, out, "--max-gb", "0.000000001")) == 6  # cap: first kept file exceeds it
    assert not (out / f"_done_{MONTH}.json").exists()
    assert dap.main(args(tar_path, tmp_path / "o2", "--min-free-gb", "1e12")) == 5
    assert dap.main(args(tar_path, tmp_path / "o3", "--market-types", "ALL")) == 0
    assert {r["market_type"] for r in manifest_rows(tmp_path / "o3")} == {"MATCH_ODDS", "CORRECT_SCORE"}


class _Server(BaseHTTPRequestHandler):
    body, state = b"", {"drops": 1, "ranges": []}

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/expired":
            return self.send_error(403)
        if self.path != "/tar":
            return self.send_error(404)
        rng = self.headers.get("Range")
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        self.state["ranges"].append(start)
        part = self.body[start:]
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(len(part)))
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{len(self.body) - 1}/{len(self.body)}")
        self.end_headers()
        if not rng and self.state["drops"]:  # first full request: cut the connection mid-body
            self.state["drops"] -= 1
            self.wfile.write(part[: len(part) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(part)


def test_http_opener_range_resume_and_403(tar_path, tmp_path, monkeypatch, no_sleep):
    pytest.importorskip("requests")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    handler = type("H", (_Server,), {"body": tar_path.read_bytes(), "state": {"drops": 1, "ranges": []}})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        out = tmp_path / "out"
        stats = dap.extract_tar(dap.http_opener(base + "/tar"), len(handler.body), MONTH, cfg(out))
        assert stats["kept"] == 3 and stats["bytes_read"] == len(handler.body) and data_files(out) == KEPT_FILES
        assert no_sleep == [5]  # one mid-body drop -> exactly one reconnect
        assert handler.state["ranges"][0] == 0 and len(handler.state["ranges"]) == 2 and handler.state["ranges"][1] > 0
        with pytest.raises(dap.UrlExpired):
            dap.http_opener(base + "/expired")(0)
        with pytest.raises(RuntimeError):
            dap.http_opener(base + "/nope")(0)
    finally:
        srv.shutdown()
