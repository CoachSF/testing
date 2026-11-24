"""Lazy, manifest-based access to the PRO market files written by scripts/fetch_betfair_pro_dap.py.

``<out_dir>/manifest_<month>.jsonl`` lists every extracted market (market_id, event_id, event_name,
market_type, market_time, country_code, event_type_id, bytes, member, saved). The manifest alone is enough to build
the event index; the (large) market files are parsed one at a time, on first access, by MarketStore.
Several folders can be combined: every row remembers its ``root`` folder and that folder's ``default_sport``
(used for rows whose manifest lacks ``event_type_id``).
"""
from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pandas as pd

from .betfair_files import MarketRecord, iso_to_ms, iter_sources, parse_stream_bytes
from .matching import EventInfo
from .textnorm import split_event_name

COLUMNS = ["market_id", "event_id", "event_name", "market_type", "market_time", "country_code",
           "bytes", "member", "saved", "event_type_id", "market_time_ms", "root", "default_sport"]
EVENT_TYPE_SPORT = {"1": "Soccer", "2": "Tennis", "7522": "Basketball", "7524": "Ice Hockey", "7511": "Baseball",
                    "468328": "Handball", "998917": "Volleyball", "6423": "American Football", "4": "Cricket",
                    "5": "Rugby Union", "1477": "Rugby League", "3": "Golf", "6": "Boxing", "3503": "Darts",
                    "6422": "Snooker", "26420387": "Mixed Martial Arts"}


def _ms(s):
    try:
        return iso_to_ms(s)
    except (ValueError, TypeError, AttributeError):
        return None


def load_manifest(out_dirs, default_sports=None) -> pd.DataFrame:
    """All manifest_*.jsonl rows of one folder or a list of folders (bad lines skipped), one row per market_id
    (largest ``bytes``). ``root`` = the folder a row came from; ``default_sport`` = that folder's sport (from
    ``default_sports``, a list parallel to the folders or a dict keyed by folder; default "Soccer")."""
    dirs = [out_dirs] if isinstance(out_dirs, (str, Path)) else list(out_dirs)
    if isinstance(default_sports, dict):
        sports = [default_sports.get(d, default_sports.get(str(d), "Soccer")) for d in dirs]
    else:
        sports = list(default_sports or []) + ["Soccer"] * (len(dirs) - len(default_sports or []))
    rows = []
    for d, sport in zip(dirs, sports):
        for p in sorted(Path(d).glob("manifest_*.jsonl")):
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if isinstance(r, dict) and r.get("market_id"):
                    rows.append({**r, "root": str(d), "default_sport": sport})
    df = pd.DataFrame(rows, columns=[c for c in COLUMNS if c != "market_time_ms"])
    df["event_type_id"] = pd.Series([None if pd.isna(v) else str(v) for v in df["event_type_id"]], index=df.index, dtype=object)
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce")
    df = df.sort_values("bytes", ascending=False, kind="stable").drop_duplicates("market_id")
    df["market_time_ms"] = pd.array([_ms(s) for s in df["market_time"]], dtype="Int64")
    return df[COLUMNS].reset_index(drop=True)


def _first(g: pd.DataFrame, col: str):
    return next((str(v) for v in g[col] if isinstance(v, (str, int)) and str(v)), None) if col in g else None


def build_event_index_from_manifest(df: pd.DataFrame, sport: str = "Soccer") -> dict[str, EventInfo]:
    """Sport per event: its event_type_id (EVENT_TYPE_SPORT), else its folder's default_sport, else ``sport``."""
    idx: dict[str, EventInfo] = {}
    for eid, g in df.groupby("event_id", sort=False):
        etid = _first(g, "event_type_id")
        sp_name = EVENT_TYPE_SPORT.get(etid, f"eventType:{etid}") if etid else (_first(g, "default_sport") or sport)
        name = next((n for n in g["event_name"] if isinstance(n, str) and n), None)
        sp = split_event_name(name) if name else None
        mo = g.loc[g["market_type"] == "MATCH_ODDS", "market_time_ms"].dropna()
        t = mo.min() if len(mo) else g["market_time_ms"].min()
        ev = idx[str(eid)] = EventInfo(
            event_id=str(eid), sport=sp_name, event_name=name, open_date_ms=None if pd.isna(t) else int(t),
            country_code=next((c for c in g["country_code"] if isinstance(c, str) and c), None),
            home=sp[0] if sp else None, away=sp[1] if sp else None)
        for mt, mid in zip(g["market_type"], g["market_id"]):
            ev.markets.setdefault(mt if isinstance(mt, str) and mt else "UNKNOWN", []).append(mid)
    return idx


class MarketStore:
    """Mapping-style, lazy view of the manifest's markets: ``in`` checks the file exists, ``[]`` parses
    that one file on first use. At most ``max_cached`` parsed markets are kept (LRU); ``loads`` counts parses."""

    def __init__(self, out_dir, df: pd.DataFrame, max_cached: int = 32):
        self.out_dir, self.max_cached, self.loads = Path(out_dir), max_cached, 0
        self._saved = dict(zip(df["market_id"], df["saved"]))
        self._root = dict(zip(df["market_id"], df["root"])) if "root" in df else {}  # per-row folder (several PRO dirs)
        self._cache: OrderedDict[str, MarketRecord] = OrderedDict()

    def _path(self, market_id) -> Path | None:
        s, root = self._saved.get(market_id), self._root.get(market_id)
        return Path(root if isinstance(root, str) and root else self.out_dir) / s if isinstance(s, str) and s else None

    def __contains__(self, market_id) -> bool:
        p = self._path(market_id)
        return p is not None and p.is_file()

    def __len__(self) -> int:
        return len(self._saved)

    def __getitem__(self, market_id) -> MarketRecord:
        if market_id in self._cache:
            self._cache.move_to_end(market_id)
            return self._cache[market_id]
        if market_id not in self:
            raise KeyError(market_id)
        rec = next((r for lbl, raw in iter_sources(self._path(market_id))
                    for r in parse_stream_bytes(raw, source=lbl, market_filter=lambda m: m == market_id)), None)
        if rec is None:
            raise KeyError(market_id)
        self.loads += 1
        self._cache[market_id] = rec
        if len(self._cache) > self.max_cached:
            self._cache.popitem(last=False)
        return rec

    def get(self, market_id, default=None):
        try:
            return self[market_id]
        except KeyError:
            return default
