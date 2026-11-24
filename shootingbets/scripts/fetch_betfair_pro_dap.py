#!/usr/bin/env python3
"""Stream a huge Betfair PRO monthly tar over HTTP and keep only the market files we need.

The tars (13+ GB each) are NEVER stored: the HTTP body is read as a stream, each member (one
bz2 JSON-lines file per market) is inspected in memory and only matching members are saved,
byte-for-byte unchanged, to  <out-dir>/<marketTime UTC date>/<eventId>/<member basename>.

    python scripts/fetch_betfair_pro_dap.py --urls urls.jsonl --from 2026-09-30 --to 2026-10-07
    python scripts/fetch_betfair_pro_dap.py --source local.tar --month 2026-10 --from ... --to ...

urls.jsonl: one JSON object per line: path, month ("2026-10"), size (bytes), url, expires_utc.
The presigned ``url`` is a SECRET: it is never printed or logged (error text from the HTTP library,
which embeds it, is deliberately dropped).

Per member: (a) OPT-IN path prefilter (``--path-pad-days N``): the path date (``/2026/Oct/1/``) is
checked against [from - N, to + N] and members outside are skipped without being decompressed
(their bytes still pass through the stream - a tar cannot be skipped over). It is OFF by default
because the folder date is the archive/settlement date, NOT marketTime: e.g. member
``PRO/2026/Nov/3/36202316/1.264004005.bz2`` has marketTime 2026-10-23T18:45Z (11+ days earlier), so
enabling the prefilter can silently drop in-window markets. With it off every member is peeked;
(b) the first marketDefinition is peeked
(bz2 decompressed incrementally, first 5 lines / 2 MB); (c) kept iff the market type is wanted and
marketTime's UTC date is within [from, to]. Resume = just re-run: matching files that already exist
with the same size are counted ``already`` and not rewritten (``kept`` counts new writes only).
A month is only marked done (``_done_<month>.json``) when the stream delivered exactly ``size`` bytes.

Exit codes: 0 ok | 2 bad args | 3 TRUNCATED stream / corrupt tar (no done-marker) |
4 presigned URL expired (re-mint; message lists the months still to do) |
5 free disk below --min-free-gb | 6 kept bytes exceeded --max-gb.
"""
import argparse
import bz2
import json
import os
import re
import shutil
import tarfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GB = 1e9
DEFAULT_TYPES = ("MATCH_ODDS,OVER_UNDER_05,OVER_UNDER_15,OVER_UNDER_25,OVER_UNDER_35,OVER_UNDER_45,"
                 "OVER_UNDER_55,OVER_UNDER_65,OVER_UNDER_75,OVER_UNDER_85,ASIAN_HANDICAP,ALT_TOTAL_GOALS,"
                 "DRAW_NO_BET,DOUBLE_CHANCE,BOTH_TEAMS_TO_SCORE,HALF_TIME,FIRST_HALF_GOALS_05,"
                 "FIRST_HALF_GOALS_15,FIRST_HALF_GOALS_25,HANDICAP,MONEY_LINE")
_MONTHS = {m: i for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split(), 1)}
_PATH_DATE = re.compile(r"(?:^|/)(20\d\d)/([A-Za-z]{3})/(\d{1,2})/")
PEEK_LINES, PEEK_BYTES = 5, 2_000_000


class UrlExpired(Exception):
    """HTTP 403 on a presigned URL: re-mint needed. Never retried."""


class Stop(Exception):
    """Run must stop with exit code ``code`` (3 truncated, 5 disk low, 6 cap exceeded)."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


@dataclass
class Cfg:
    out_dir: Path
    date_from: date
    date_to: date
    types: frozenset | None = None  # None = every market type
    pad_days: int | None = None  # None = path-date prefilter disabled (peek every member)
    max_bytes: float = 8.0 * GB
    min_free: float = 10.0 * GB
    run_kept: int = 0  # bytes newly written so far this run (all tars); checked against max_bytes


# --------------------------------------------------------------------------- reader
class ResumableReader:
    """File-like ``read(n)`` over ``opener(offset) -> file-like``; reconnects from the current byte."""

    def __init__(self, opener, size: int, max_retries: int = 8):
        self.opener, self.size, self.max_retries = opener, size, max_retries
        self.pos, self._f, self._fails = 0, None, 0

    def close(self):
        f, self._f = self._f, None
        if f is not None and hasattr(f, "close"):
            try:
                f.close()
            except Exception:
                pass

    def _reconnect(self, why: str):
        self.close()
        self._fails += 1
        if self._fails > self.max_retries:
            raise Stop(3, f"TRUNCATED: stream lost at byte {self.pos} of {self.size} "
                          f"after {self.max_retries} reconnects ({why})") from None
        delay = min(5 * 2 ** (self._fails - 1), 60)
        print(f"  reconnect {self._fails}/{self.max_retries} at byte {self.pos} ({why}); wait {delay}s", flush=True)
        time.sleep(delay)

    def read(self, n: int = 1 << 20) -> bytes:
        while True:
            try:
                if self._f is None:
                    self._f = self.opener(self.pos)
                b = self._f.read(n)
            except UrlExpired:
                raise
            except Exception as exc:  # never print str(exc): HTTP errors embed the presigned URL
                self._reconnect(type(exc).__name__)
                continue
            if not b and self.pos < self.size:
                self._reconnect("short stream")
                continue
            self.pos += len(b)
            if b:
                self._fails = 0  # progress made: the retry budget is for consecutive failures
            return b


def http_opener(url: str):
    def opener(offset: int):
        import requests

        resp = requests.get(url, stream=True, headers={"Range": f"bytes={offset}-"} if offset else {},
                            timeout=(30, 300))
        if resp.status_code == 403:
            resp.close()
            raise UrlExpired("HTTP 403")
        if resp.status_code != (206 if offset else 200):
            code = resp.status_code
            resp.close()
            raise RuntimeError(f"unexpected HTTP status {code}")
        resp.raw.decode_content = False
        return resp.raw
    return opener


def local_opener(path):
    def opener(offset: int):
        f = open(path, "rb")
        f.seek(offset)
        return f
    return opener


# --------------------------------------------------------------------------- members
def path_date(name: str) -> date | None:
    m = _PATH_DATE.search(name)
    mon = _MONTHS.get(m.group(2).lower()) if m else None
    if not mon:
        return None
    try:
        return date(int(m.group(1)), mon, int(m.group(3)))
    except ValueError:
        return None


def peek_definition(data: bytes):
    """(market_id, marketDefinition) of the first mc carrying one in the first lines, else None."""
    buf = data[:PEEK_BYTES]
    if data[:3] == b"BZh":
        d, buf, pos = bz2.BZ2Decompressor(), b"", 0
        try:
            while not d.eof and len(buf) < PEEK_BYTES and buf.count(b"\n") < PEEK_LINES:
                chunk = b""
                if d.needs_input:
                    chunk, pos = data[pos:pos + 65536], pos + 65536
                    if not chunk:
                        break
                buf += d.decompress(chunk, PEEK_BYTES - len(buf))
        except (OSError, ValueError, EOFError):
            return None
    for line in buf.split(b"\n")[:PEEK_LINES]:
        try:
            msg = json.loads(line)
            for mc in msg.get("mc") or []:
                if mc.get("marketDefinition"):
                    return mc.get("id"), mc["marketDefinition"]
        except Exception:
            continue
    return None


def market_date(md: dict) -> date | None:
    try:
        return datetime.fromisoformat(md["marketTime"].replace("Z", "+00:00")).astimezone(timezone.utc).date()
    except (KeyError, ValueError, AttributeError):
        return None


def _safe(x) -> str:
    return re.sub(r"[^\w.\-]", "_", str(x))


def _check_free(cfg: Cfg):
    free = shutil.disk_usage(cfg.out_dir).free
    if free < cfg.min_free:
        raise Stop(5, f"DISK LOW: {free / GB:.1f} GB free < {cfg.min_free / GB:.1f} GB minimum")


def _progress(month, s, reader, t0, tag=""):
    el = max(time.monotonic() - t0, 1e-9)
    rate = reader.pos / el
    eta = (reader.size - reader.pos) / rate if rate else 0
    print(f"[{month}]{tag} {reader.pos / GB:.2f}/{reader.size / GB:.2f} GB {100 * reader.pos / max(reader.size, 1):.1f}% "
          f"{rate / 1e6:.1f} MB/s | members {s['members_seen']} in_window {s['in_path_window']} "
          f"kept {s['kept']} ({s['kept_bytes'] / 1e6:.1f} MB) already {s['already']} | "
          f"elapsed {el / 60:.1f}m eta {eta / 60:.1f}m", flush=True)


def extract_tar(opener, size: int, month: str, cfg: Cfg) -> dict:
    """Stream one tar, save matching members, return stats. Raises Stop / UrlExpired."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _check_free(cfg)
    s = dict(members_seen=0, in_path_window=0, kept=0, already=0, no_def=0, kept_bytes=0, bytes_read=0)
    if cfg.pad_days is not None:
        lo, hi = cfg.date_from - timedelta(days=cfg.pad_days), cfg.date_to + timedelta(days=cfg.pad_days)
    manifest = cfg.out_dir / f"manifest_{month}.jsonl"
    have: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            try:
                have.add(json.loads(line)["saved"])
            except Exception:
                pass
    reader, t0, last, rows = ResumableReader(opener, size), time.monotonic(), time.monotonic(), 0
    try:
        with open(manifest, "a", encoding="utf-8") as mf, \
                tarfile.open(fileobj=reader, mode="r|", bufsize=1 << 20) as tf:
            while (m := tf.next()) is not None:
                tf.members.clear()  # stream mode keeps every TarInfo otherwise
                if not m.isfile():
                    continue
                s["members_seen"] += 1
                if time.monotonic() - last >= 30:
                    last = time.monotonic()
                    _progress(month, s, reader, t0)
                if cfg.pad_days is not None:
                    pd = path_date(m.name)
                    if pd is not None and not lo <= pd <= hi:
                        continue
                s["in_path_window"] += 1
                data = tf.extractfile(m).read()
                found = peek_definition(data)
                if found is None or market_date(found[1]) is None:
                    s["no_def"] += 1
                    continue
                mid, md = found
                if (cfg.types is not None and md.get("marketType") not in cfg.types) \
                        or not cfg.date_from <= market_date(md) <= cfg.date_to:
                    continue
                dest = (cfg.out_dir / market_date(md).isoformat() / _safe(md.get("eventId") or "unknown")
                        / _safe(m.name.replace("\\", "/").rsplit("/", 1)[-1]))
                if dest.exists() and dest.stat().st_size == len(data):
                    s["already"] += 1
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".part")
                    tmp.write_bytes(data)
                    os.replace(tmp, dest)
                    s["kept"] += 1
                    s["kept_bytes"] += len(data)
                    cfg.run_kept += len(data)
                    if s["kept"] % 200 == 0:
                        _check_free(cfg)
                saved = dest.relative_to(cfg.out_dir).as_posix()
                if saved not in have:
                    have.add(saved)
                    mf.write(json.dumps({
                        "market_id": mid, "event_id": md.get("eventId"), "event_name": md.get("eventName"),
                        "market_type": md.get("marketType"), "market_time": md.get("marketTime"),
                        "country_code": md.get("countryCode"), "bytes": len(data), "member": m.name,
                        "saved": saved}) + "\n")
                    rows += 1
                    if rows % 200 == 0:
                        mf.flush()
                if cfg.run_kept > cfg.max_bytes:
                    raise Stop(6, f"CAP: kept {cfg.run_kept / GB:.2f} GB this run > --max-gb {cfg.max_bytes / GB:.2f}")
            while reader.read(1 << 20):  # drain padding / trailing bytes to EOF
                pass
    except (tarfile.ReadError, EOFError):
        raise Stop(3, f"TRUNCATED: tar unreadable at byte {reader.pos} of {size}") from None
    finally:
        reader.close()
    s["bytes_read"] = reader.pos
    _progress(month, s, reader, t0, " done")
    if reader.pos != size:
        raise Stop(3, f"TRUNCATED: read {reader.pos} bytes, expected {size}")
    return s


# --------------------------------------------------------------------------- main
def _marker_ok(marker: Path, window: dict, types: list) -> bool:
    try:
        d = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return d.get("window") == window and d.get("market_types") == types


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--urls", help="urls.jsonl (path, month, size, url, expires_utc)")
    ap.add_argument("--source", help="local .tar to read instead of HTTP (tests); needs --month")
    ap.add_argument("--month", help="month label for --source, e.g. 2026-10")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "betfair_pro"))
    ap.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD, inclusive (marketTime UTC date)")
    ap.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD, inclusive")
    ap.add_argument("--market-types", default=DEFAULT_TYPES, help="comma list, or ALL")
    ap.add_argument("--path-pad-days", type=int, default=None,
                    help="OPT-IN prefilter: skip members whose path date is outside [from-N, to+N]. Default: off "
                         "(every member is peeked). The path date is the archive/settlement folder date, NOT "
                         "marketTime (e.g. PRO/2026/Nov/3/36202316/1.264004005.bz2 has marketTime 2026-10-23), "
                         "so enabling this risks dropping in-window markets")
    ap.add_argument("--max-gb", type=float, default=8.0)
    ap.add_argument("--min-free-gb", type=float, default=10.0)
    a = ap.parse_args(argv)
    if not (a.urls or (a.source and a.month)):
        ap.error("give --urls, or --source together with --month")
    d_from, d_to = date.fromisoformat(a.date_from), date.fromisoformat(a.date_to)
    wanted = sorted({t.strip() for t in a.market_types.split(",") if t.strip()})
    all_types = wanted == ["ALL"]
    cfg = Cfg(Path(a.out_dir), d_from, d_to, None if all_types else frozenset(wanted), a.path_pad_days,
              a.max_gb * GB, a.min_free_gb * GB)
    window = {"from": d_from.isoformat(), "to": d_to.isoformat()}
    types = ["ALL"] if all_types else wanted

    if a.source:
        rows = [{"month": a.month, "size": os.path.getsize(a.source), "opener": local_opener(a.source)}]
    else:
        with open(a.urls, encoding="utf-8") as fh:
            rows = [json.loads(ln) for ln in fh if ln.strip()]
        for r in rows:
            r["opener"] = http_opener(r["url"])
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r["month"] for r in rows if not _marker_ok(cfg.out_dir / f"_done_{r['month']}.json", window, types)]
    for r in rows:
        month, marker = r["month"], cfg.out_dir / f"_done_{r['month']}.json"
        if month not in todo:
            print(f"[{month}] already done for this window/market types - skipping", flush=True)
            continue
        try:
            stats = extract_tar(r["opener"], int(r["size"]), month, cfg)
        except UrlExpired:
            print(f"URL EXPIRED (HTTP 403) while reading {month}: re-mint the presigned URL(s) for: "
                  f"{', '.join(todo[todo.index(month):])}", flush=True)
            return 4
        except Stop as exc:
            print(f"STOP [{month}]: {exc}", flush=True)
            return exc.code
        stats.update(window=window, market_types=types, path_pad_days=cfg.pad_days,
                     finished_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
        tmp = marker.with_name(marker.name + ".tmp")
        tmp.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        os.replace(tmp, marker)
        todo.remove(month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
