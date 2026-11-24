"""Synthetic Betfair-stream fixtures for unit tests (NOT real data)."""
from __future__ import annotations

import bz2
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

T0 = 1_790_000_000_000  # arbitrary epoch ms
MIN = 60_000


def md(status="OPEN", in_play=False, runner_status=("ACTIVE", "ACTIVE", "ACTIVE"), market_type="MATCH_ODDS",
       event_name="Arsenal v Chelsea", open_ms=T0 + 120 * MIN, settled=False):
    from datetime import datetime, timezone
    iso = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    d = {
        "eventId": "34000001", "eventTypeId": "1", "marketType": market_type, "bettingType": "ODDS",
        "marketTime": iso, "openDate": iso, "status": status, "inPlay": in_play, "numberOfWinners": 1,
        "marketBaseRate": 5.0, "turnInPlayEnabled": True, "countryCode": "GB", "timezone": "GMT",
        "eventName": event_name, "name": "Match Odds",
        "runners": [
            {"id": 1001, "name": "Arsenal", "sortPriority": 1, "status": runner_status[0]},
            {"id": 1002, "name": "Chelsea", "sortPriority": 2, "status": runner_status[1]},
            {"id": 58805, "name": "The Draw", "sortPriority": 3, "status": runner_status[2]},
        ],
    }
    if settled:
        d["settledTime"] = iso
    return d


def basic_market_lines(market_id="1.200000001"):
    """Pre-off LTP path for Arsenal: 2.50 -> 2.40 (t0+10) -> 2.30 (t0+30) -> in-play at t0+120 -> Arsenal wins."""
    L = []
    L.append({"op": "mcm", "clk": "1", "pt": T0 - 60 * MIN, "mc": [{"id": market_id, "marketDefinition": md()}]})
    L.append({"op": "mcm", "clk": "2", "pt": T0 - 30 * MIN, "mc": [{"id": market_id, "rc": [
        {"id": 1001, "ltp": 2.5}, {"id": 1002, "ltp": 3.1}, {"id": 58805, "ltp": 3.4}]}]})
    L.append({"op": "mcm", "clk": "3", "pt": T0 + 10 * MIN, "mc": [{"id": market_id, "rc": [{"id": 1001, "ltp": 2.4}]}]})
    L.append({"op": "mcm", "clk": "4", "pt": T0 + 30 * MIN, "mc": [{"id": market_id, "rc": [{"id": 1001, "ltp": 2.3}]}]})
    L.append({"op": "mcm", "clk": "5", "pt": T0 + 119 * MIN, "mc": [{"id": market_id, "rc": [{"id": 1001, "ltp": 2.32}]}]})
    L.append({"op": "mcm", "clk": "6", "pt": T0 + 120 * MIN, "mc": [{"id": market_id, "marketDefinition": md(in_play=True)}]})
    L.append({"op": "mcm", "clk": "7", "pt": T0 + 125 * MIN, "mc": [{"id": market_id, "rc": [{"id": 1001, "ltp": 1.8}]}]})
    L.append({"op": "mcm", "clk": "8", "pt": T0 + 240 * MIN, "mc": [{"id": market_id, "marketDefinition": md(
        status="CLOSED", in_play=True, runner_status=("WINNER", "LOSER", "LOSER"), settled=True)}]})
    return L


def to_bytes(lines) -> bytes:
    return ("\n".join(json.dumps(x) for x in lines) + "\n").encode()


@pytest.fixture
def basic_bz2(tmp_path):
    p = tmp_path / "1.200000001.bz2"
    p.write_bytes(bz2.compress(to_bytes(basic_market_lines())))
    return p


@pytest.fixture
def basic_tar(tmp_path):
    p = tmp_path / "data.tar"
    with tarfile.open(p, "w") as tf:
        for mid in ("1.200000001", "1.200000002"):
            raw = bz2.compress(to_bytes(basic_market_lines(mid)))
            ti = tarfile.TarInfo(name=f"BASIC/2026/Oct/1/34000001/{mid}.bz2")
            ti.size = len(raw)
            tf.addfile(ti, io.BytesIO(raw))
    return p
