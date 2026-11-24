"""Reader for Betfair Exchange historical stream files (historicdata.betfair.com).

Handles the BASIC plan (last traded price, ~1 minute granularity) as well as
ADVANCED (top-3 ladder ``batb``/``batl``) and PRO (full ladder ``atb``/``atl``,
``trd``) files. Input may be a directory tree, a .tar (as downloaded from the
website), individual .bz2/.gz files, a .zip, or plain JSON-lines files.
Archives are read in memory; nothing is extracted to disk.

Each line of a file is one Exchange Stream API "mcm" message:
    {"op":"mcm","clk":"...","pt":<publish time ms UTC>,
     "mc":[{"id":"1.234","marketDefinition":{...},"rc":[{"id":123,"ltp":2.1}]}]}
"""
from __future__ import annotations

import bisect
import bz2
import gzip
import io
import json
import lzma
import os
import tarfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Iterator

try:  # optional speed-up
    import orjson as _orjson  # type: ignore

    def _loads(b: bytes):
        return _orjson.loads(b)
except Exception:  # pragma: no cover - fallback
    def _loads(b: bytes):
        return json.loads(b)


EVENT_TYPE_NAMES = {
    "1": "Soccer",
    "2": "Tennis",
    "3": "Golf",
    "4": "Cricket",
    "5": "Rugby Union",
    "6": "Boxing",
    "7": "Horse Racing",
    "8": "Motor Sport",
    "11": "Cycling",
    "1477": "Rugby League",
    "3503": "Darts",
    "4339": "Greyhound Racing",
    "6231": "Financial Bets",
    "6422": "Snooker",
    "6423": "American Football",
    "7511": "Baseball",
    "7522": "Basketball",
    "7524": "Ice Hockey",
    "61420": "Australian Rules",
    "468328": "Handball",
    "998917": "Volleyball",
    "2152880": "Gaelic Games",
    "2378961": "Politics",
    "26420387": "Mixed Martial Arts",
    "27454571": "Esports",
    "10": "Special Bets",
}


def iso_to_ms(s: str | None) -> int | None:
    if not s:
        return None
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_dt(ms: int | None) -> datetime | None:
    if ms is None:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


RunnerKey = tuple  # (selection_id: int, handicap: float)


@dataclass
class RunnerSeries:
    selection_id: int
    handicap: float = 0.0
    name: str | None = None
    sort_priority: int | None = None
    status: str | None = None  # last seen status (final status once CLOSED)
    adjustment_factor: float | None = None
    removal_date: str | None = None
    bsp: float | None = None
    # last traded price series
    ltp_t: list[int] = field(default_factory=list)
    ltp_p: list[float] = field(default_factory=list)
    # traded volume series (ADVANCED/PRO)
    tv_t: list[int] = field(default_factory=list)
    tv_v: list[float] = field(default_factory=list)
    # best available back/lay series (ADVANCED/PRO); NaN when side empty
    bb_t: list[int] = field(default_factory=list)
    bb_p: list[float] = field(default_factory=list)
    bb_s: list[float] = field(default_factory=list)
    bl_t: list[int] = field(default_factory=list)
    bl_p: list[float] = field(default_factory=list)
    bl_s: list[float] = field(default_factory=list)

    @property
    def key(self) -> RunnerKey:
        return (self.selection_id, self.handicap)


@dataclass
class MarketRecord:
    market_id: str
    source: str = ""
    event_id: str | None = None
    event_type_id: str | None = None
    event_name: str | None = None
    market_name: str | None = None
    market_type: str | None = None
    betting_type: str | None = None
    country_code: str | None = None
    timezone: str | None = None
    venue: str | None = None
    open_date_ms: int | None = None
    market_time_ms: int | None = None
    suspend_time_ms: int | None = None
    settled_time_ms: int | None = None
    market_base_rate: float | None = None
    number_of_winners: int | None = None
    turn_in_play_enabled: bool | None = None
    in_play_start_ms: int | None = None
    close_ms: int | None = None
    first_pt: int | None = None
    last_pt: int | None = None
    final_status: str | None = None
    data_level: str = "UNKNOWN"  # BASIC / ADVANCED / PRO
    # (pt, status, inPlay) whenever either changes
    status_changes: list[tuple[int, str, bool]] = field(default_factory=list)
    runners: dict[RunnerKey, RunnerSeries] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def sport(self) -> str | None:
        return EVENT_TYPE_NAMES.get(str(self.event_type_id), self.event_type_id)

    @property
    def market_time(self) -> datetime | None:
        return ms_to_dt(self.market_time_ms)

    @property
    def open_date(self) -> datetime | None:
        return ms_to_dt(self.open_date_ms)

    def off_ms(self) -> int | None:
        """Best estimate of the moment the event started ("the off").

        In-play turn time if the market went in-play; otherwise the first
        suspension/closure at or after (scheduled start - 30 min); otherwise the
        scheduled start.
        """
        if self.in_play_start_ms is not None:
            return self.in_play_start_ms
        if self.market_time_ms is not None:
            floor = self.market_time_ms - 30 * 60_000
            for pt, status, _ip in self.status_changes:
                if pt >= floor and status in ("SUSPENDED", "CLOSED"):
                    return pt
            return self.market_time_ms
        return None

    def status_at(self, t_ms: int) -> tuple[str | None, bool | None]:
        """(status, inPlay) in force at time t (last change at or before t)."""
        if not self.status_changes:
            return (None, None)
        times = [c[0] for c in self.status_changes]
        i = bisect.bisect_right(times, t_ms) - 1
        if i < 0:
            return (None, None)
        return (self.status_changes[i][1], self.status_changes[i][2])

    def runner_by_name(self, name: str) -> list[RunnerSeries]:
        n = name.strip().lower()
        return [r for r in self.runners.values() if (r.name or "").strip().lower() == n]


# ---------------------------------------------------------------------------
# Source iteration
# ---------------------------------------------------------------------------

def _decompress(data: bytes) -> bytes:
    if data[:3] == b"BZh":
        return bz2.decompress(data)
    if data[:2] == b"\x1f\x8b":
        return gzip.decompress(data)
    if data[:6] == b"\xfd7zXZ\x00":
        return lzma.decompress(data)
    return data


def _iter_tar(fileobj, label: str) -> Iterator[tuple[str, bytes]]:
    with tarfile.open(fileobj=fileobj, mode="r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            f = tf.extractfile(m)
            if f is None:
                continue
            data = f.read()
            yield from _iter_blob(f"{label}::{m.name}", data)


def _iter_zip(data: bytes, label: str) -> Iterator[tuple[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            yield from _iter_blob(f"{label}::{info.filename}", zf.read(info))


def _looks_like_tar(data: bytes) -> bool:
    return len(data) > 262 and data[257:262] == b"ustar"


def _iter_blob(label: str, data: bytes) -> Iterator[tuple[str, bytes]]:
    """Yield (label, json-lines bytes) for a blob that may be compressed/archived."""
    if data[:4] == b"PK\x03\x04":
        yield from _iter_zip(data, label)
        return
    raw = _decompress(data)
    if _looks_like_tar(raw):
        yield from _iter_tar(io.BytesIO(raw), label)
        return
    yield (label, raw)


def iter_sources(paths: str | os.PathLike | Iterable[str | os.PathLike]) -> Iterator[tuple[str, bytes]]:
    """Yield (label, decompressed json-lines bytes) for every market file found."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    for p in paths:
        p = os.fspath(p)
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for fn in sorted(files):
                    if fn.startswith("."):
                        continue
                    fp = os.path.join(root, fn)
                    with open(fp, "rb") as fh:
                        yield from _iter_blob(fp, fh.read())
        else:
            with open(p, "rb") as fh:
                yield from _iter_blob(p, fh.read())


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class _MarketBuilder:
    def __init__(self, market_id: str, source: str):
        self.rec = MarketRecord(market_id=market_id, source=source)
        self._last_status: tuple[str, bool] | None = None
        # ladder state: key -> {"batb": {level: (p, s)}, "batl": {...}, "atb": {p: s}, "atl": {p: s}}
        self._ladders: dict[RunnerKey, dict[str, dict]] = {}
        self._last_best: dict[RunnerKey, tuple] = {}
        self._has_batb = False
        self._has_atb = False

    def _runner(self, sid: int, hc: float) -> RunnerSeries:
        key = (int(sid), float(hc or 0.0))
        r = self.rec.runners.get(key)
        if r is None:
            r = RunnerSeries(selection_id=int(sid), handicap=float(hc or 0.0))
            self.rec.runners[key] = r
        return r

    def apply_definition(self, pt: int, md: dict) -> None:
        rec = self.rec
        rec.event_id = str(md.get("eventId", rec.event_id) or rec.event_id or "") or None
        rec.event_type_id = str(md.get("eventTypeId", rec.event_type_id) or rec.event_type_id or "") or None
        rec.event_name = md.get("eventName", rec.event_name)
        rec.market_name = md.get("name", rec.market_name)
        rec.market_type = md.get("marketType", rec.market_type)
        rec.betting_type = md.get("bettingType", rec.betting_type)
        rec.country_code = md.get("countryCode", rec.country_code)
        rec.timezone = md.get("timezone", rec.timezone)
        rec.venue = md.get("venue", rec.venue)
        if md.get("openDate"):
            rec.open_date_ms = iso_to_ms(md["openDate"])
        if md.get("marketTime"):
            rec.market_time_ms = iso_to_ms(md["marketTime"])
        if md.get("suspendTime"):
            rec.suspend_time_ms = iso_to_ms(md["suspendTime"])
        if md.get("settledTime"):
            rec.settled_time_ms = iso_to_ms(md["settledTime"])
        if md.get("marketBaseRate") is not None:
            rec.market_base_rate = float(md["marketBaseRate"])
        if md.get("numberOfWinners") is not None:
            rec.number_of_winners = int(md["numberOfWinners"])
        if md.get("turnInPlayEnabled") is not None:
            rec.turn_in_play_enabled = bool(md["turnInPlayEnabled"])
        status = md.get("status")
        in_play = bool(md.get("inPlay", False))
        if status is not None:
            rec.final_status = status
            cur = (status, in_play)
            if cur != self._last_status:
                rec.status_changes.append((pt, status, in_play))
                self._last_status = cur
            if in_play and rec.in_play_start_ms is None:
                rec.in_play_start_ms = pt
            if status == "CLOSED" and rec.close_ms is None:
                rec.close_ms = pt
        for rd in md.get("runners", []) or []:
            r = self._runner(rd["id"], rd.get("hc", 0.0))
            if rd.get("name") is not None:
                r.name = rd["name"]
            if rd.get("sortPriority") is not None:
                r.sort_priority = int(rd["sortPriority"])
            if rd.get("status") is not None:
                r.status = rd["status"]
            if rd.get("adjustmentFactor") is not None:
                r.adjustment_factor = float(rd["adjustmentFactor"])
            if rd.get("removalDate") is not None:
                r.removal_date = rd["removalDate"]
            if rd.get("bsp") not in (None, "NaN", "Infinity"):
                try:
                    r.bsp = float(rd["bsp"])
                except (TypeError, ValueError):
                    pass

    def _update_best(self, pt: int, key: RunnerKey, r: RunnerSeries) -> None:
        lad = self._ladders.get(key)
        if lad is None:
            return
        nan = float("nan")
        if self._has_atb:
            atb = lad.get("atb", {})
            atl = lad.get("atl", {})
            bb = max((p for p, s in atb.items() if s > 0), default=None)
            bl = min((p for p, s in atl.items() if s > 0), default=None)
            bbs = atb.get(bb, 0.0) if bb is not None else 0.0
            bls = atl.get(bl, 0.0) if bl is not None else 0.0
        else:
            b0 = lad.get("batb", {}).get(0)
            l0 = lad.get("batl", {}).get(0)
            bb, bbs = (b0 if b0 and b0[1] > 0 else (None, 0.0))
            bl, bls = (l0 if l0 and l0[1] > 0 else (None, 0.0))
        cur = (bb, bbs, bl, bls)
        if self._last_best.get(key) == cur:
            return
        prev = self._last_best.get(key)
        self._last_best[key] = cur
        if prev is None or (prev[0], prev[1]) != (bb, bbs):
            r.bb_t.append(pt)
            r.bb_p.append(bb if bb is not None else nan)
            r.bb_s.append(float(bbs))
        if prev is None or (prev[2], prev[3]) != (bl, bls):
            r.bl_t.append(pt)
            r.bl_p.append(bl if bl is not None else nan)
            r.bl_s.append(float(bls))

    def apply_runner_change(self, pt: int, rc: dict, img: bool) -> None:
        key = (int(rc["id"]), float(rc.get("hc", 0.0) or 0.0))
        r = self._runner(*key)
        if "ltp" in rc and rc["ltp"] is not None:
            p = float(rc["ltp"])
            if not r.ltp_p or r.ltp_t[-1] != pt:
                r.ltp_t.append(pt)
                r.ltp_p.append(p)
            else:  # same publish time: keep the latest value
                r.ltp_p[-1] = p
        if "tv" in rc and rc["tv"] is not None:
            r.tv_t.append(pt)
            r.tv_v.append(float(rc["tv"]))
        touched = False
        lad = self._ladders.setdefault(key, {})
        if img:
            lad.clear()
        for fld in ("batb", "batl"):
            if fld in rc and rc[fld] is not None:
                self._has_batb = True
                book = lad.setdefault(fld, {})
                for level, price, size in rc[fld]:
                    if size == 0:
                        book.pop(int(level), None)
                    else:
                        book[int(level)] = (float(price), float(size))
                touched = True
        for fld in ("atb", "atl"):
            if fld in rc and rc[fld] is not None:
                self._has_atb = True
                book = lad.setdefault(fld, {})
                for price, size in rc[fld]:
                    if size == 0:
                        book.pop(float(price), None)
                    else:
                        book[float(price)] = float(size)
                touched = True
        if touched:
            self._update_best(pt, key, r)

    def apply(self, pt: int, mc: dict) -> None:
        rec = self.rec
        if rec.first_pt is None or pt < rec.first_pt:
            rec.first_pt = pt
        if rec.last_pt is None or pt > rec.last_pt:
            rec.last_pt = pt
        md = mc.get("marketDefinition")
        if md:
            self.apply_definition(pt, md)
        img = bool(mc.get("img", False))
        for rc in mc.get("rc", []) or []:
            self.apply_runner_change(pt, rc, img)

    def finish(self) -> MarketRecord:
        if self._has_atb:
            self.rec.data_level = "PRO"
        elif self._has_batb:
            self.rec.data_level = "ADVANCED"
        elif any(r.ltp_t for r in self.rec.runners.values()):
            self.rec.data_level = "BASIC"
        return self.rec


def parse_stream_bytes(data: bytes, source: str = "",
                       market_filter: Callable[[str], bool] | None = None) -> list[MarketRecord]:
    """Parse one decompressed file (JSON lines) into MarketRecords."""
    builders: dict[str, _MarketBuilder] = {}
    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            msg = _loads(line)
        except Exception:
            continue  # tolerate a corrupt/truncated line
        if msg.get("op") != "mcm":
            continue
        pt = int(msg.get("pt") or 0)
        for mc in msg.get("mc", []) or []:
            mid = mc.get("id")
            if mid is None:
                continue
            if market_filter is not None and not market_filter(mid):
                continue
            b = builders.get(mid)
            if b is None:
                b = builders[mid] = _MarketBuilder(mid, source)
            b.apply(pt, mc)
    return [b.finish() for b in builders.values()]


def load_markets(paths, market_filter: Callable[[str], bool] | None = None,
                 on_error: str = "warn") -> dict[str, MarketRecord]:
    """Load every market found under ``paths``. Later files override earlier ones
    for the same market id only if they contain more data (longer time span)."""
    out: dict[str, MarketRecord] = {}
    for label, raw in iter_sources(paths):
        try:
            recs = parse_stream_bytes(raw, source=label, market_filter=market_filter)
        except Exception as exc:  # pragma: no cover - defensive
            if on_error == "raise":
                raise
            print(f"[betfair_files] failed to parse {label}: {exc}")
            continue
        for rec in recs:
            prev = out.get(rec.market_id)
            if prev is None:
                out[rec.market_id] = rec
            else:
                span_prev = (prev.last_pt or 0) - (prev.first_pt or 0)
                span_new = (rec.last_pt or 0) - (rec.first_pt or 0)
                if span_new > span_prev:
                    out[rec.market_id] = rec
    return out
