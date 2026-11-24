#!/usr/bin/env python3
"""Download the Betfair historical (BASIC) price files needed for the ShootingBets analysis.

Run this on YOUR OWN computer (Betfair blocks cloud/overseas IPs). Standard library only.

    python fetch_betfair_basic.py --spec betfair_spec.json
    python fetch_betfair_basic.py --spec betfair_spec.json --dry-run   # check plans & sizes only

It will ask for your Betfair session token ("ssoid") - it is used only for requests
to historicdata.betfair.com and is never written to disk.

How to get the ssoid: log in at https://historicdata.betfair.com in Chrome/Edge/Firefox,
press F12 -> Application (Chrome/Edge) or Storage (Firefox) -> Cookies ->
https://historicdata.betfair.com -> copy the Value of the cookie named "ssoid".

What it does
  1. Lists your data plans (GetMyData) and warns if a needed sport/month is missing
     (BASIC data is free: add it via "Purchase data" on the website, then re-run).
  2. For each day in the spec, downloads every MATCH_ODDS file for the sport (used to
     find the alert events and their Betfair names/ids).
  3. Matches the spec's events (lenient - the final matching is done later), then
     downloads the extra market types (e.g. OVER_UNDER_25, ASIAN_HANDICAP) for those
     events only.
  4. Writes everything into one archive (default: shootingbets_betfair_basic.tar)
     plus a manifest. Upload that file to Google Drive.
Re-running resumes: files already in the cache folder are not downloaded again.
"""
from __future__ import annotations

import argparse
import bz2
import concurrent.futures as cf
import datetime as dt
import difflib
import getpass
import json
import os
import re
import sys
import tarfile
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

API = "https://historicdata.betfair.com/api/"
UA = "shootingbets-analysis/1.0 (+personal research)"


# --------------------------------------------------------------------------- http
class Client:
    def __init__(self, ssoid: str, timeout: int = 60):
        self.ssoid = ssoid
        self.timeout = timeout

    def _headers(self) -> dict:
        return {"ssoid": self.ssoid, "Content-Type": "application/json", "Accept": "application/json",
                "User-Agent": UA}

    def _open(self, req: urllib.request.Request, retries: int = 5) -> bytes:
        delay = 2.0
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as e:
                body = e.read()[:300].decode("utf-8", "replace")
                if e.code in (401, 403):
                    raise SystemExit(f"\nBetfair refused the request (HTTP {e.code}). Your ssoid may be wrong or expired"
                                     f" - log in again and copy a fresh one.\n{body}")
                if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise
        raise RuntimeError("unreachable")

    def post(self, method: str, payload: dict):
        req = urllib.request.Request(API + method, data=json.dumps(payload).encode(), headers=self._headers(),
                                     method="POST")
        raw = self._open(req)
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise SystemExit(f"Unexpected (non-JSON) answer from Betfair for {method}: {raw[:200]!r}")

    def download(self, file_path: str) -> bytes:
        url = API + "DownloadFile?filePath=" + urllib.parse.quote(file_path, safe="/")
        req = urllib.request.Request(url, headers={"ssoid": self.ssoid, "User-Agent": UA})
        return self._open(req)


def day_filter(sport: str, plan: str, day: dt.date, market_types=None, countries=None) -> dict:
    f = {"sport": sport, "plan": plan,
         "fromDay": day.day, "fromMonth": day.month, "fromYear": day.year,
         "toDay": day.day, "toMonth": day.month, "toYear": day.year,
         "fileTypeCollection": ["M"]}
    # omit empty collections (= no filter) rather than sending [] which could mean "nothing"
    if market_types:
        f["marketTypesCollection"] = list(market_types)
    if countries:
        f["countriesCollection"] = list(countries)
    return f


# ----------------------------------------------------------------------- matching
def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"\b(fc|cf|sc|afc|ac|as|cd|sd|ud|fk|sk|club|de|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / max(1, len(ta | tb))
    sub = 0.85 if (ta <= tb or tb <= ta) else 0.0
    return max(r, jac, sub)


def split_event(name: str):
    if " @ " in name:
        a, h = name.split(" @ ", 1)
        return h, a
    parts = re.split(r"\s+(?:v|vs|-)\s+", name, maxsplit=1)
    return (parts[0], parts[1]) if len(parts) == 2 else (name, "")


def first_definition(raw: bytes) -> dict | None:
    data = bz2.decompress(raw) if raw[:3] == b"BZh" else raw
    for line in data.splitlines():
        try:
            msg = json.loads(line)
        except Exception:
            continue
        for mc in msg.get("mc", []) or []:
            md = mc.get("marketDefinition")
            if md:
                return md
    return None


def iso_ms(s: str) -> int:
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", required=True, help="betfair_spec.json produced by the analysis")
    ap.add_argument("--out", default="shootingbets_betfair_basic.tar")
    ap.add_argument("--cache", default="betfair_cache", help="folder for downloaded files (allows resume)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="only check plans and file counts")
    args = ap.parse_args()

    with open(args.spec, "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    plan = spec.get("plan", "Basic Plan")

    ssoid = os.environ.get("BETFAIR_SSOID") or getpass.getpass("Paste your Betfair ssoid (input hidden): ").strip()
    if not ssoid:
        raise SystemExit("No ssoid given.")
    cl = Client(ssoid)

    print("Checking your Betfair historical data plans ...")
    mine = cl.post("GetMyData", {})
    have = {(p.get("sport"), p.get("plan"), (p.get("forDate") or "")[:7]) for p in mine}
    sports_have = sorted({(s, p) for s, p, _ in have})
    print("  plans on your account:", ", ".join(f"{s} [{p}]" for s, p in sports_have) or "none")
    missing = []
    for rq in spec["requests"]:
        d0, d1 = dt.date.fromisoformat(rq["from"]), dt.date.fromisoformat(rq["to"])
        months = sorted({(d0 + dt.timedelta(days=i)).strftime("%Y-%m") for i in range((d1 - d0).days + 1)})
        for mth in months:
            if (rq["sport"], plan, mth) not in have:
                missing.append((rq["sport"], mth))
    if missing:
        print("\n  WARNING - not on your account yet (add the free BASIC plan on historicdata.betfair.com):")
        for s, mth in missing:
            print(f"    - {s}  {mth}")
        print("  Continuing; days without a plan will simply return no files.\n")

    os.makedirs(args.cache, exist_ok=True)
    manifest = {"plan": plan, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "requests": [],
                "matched_events": []}
    all_files: list[str] = []

    def fetch(path: str) -> str:
        local = os.path.join(args.cache, path.strip("/").replace("/", "__"))
        if not os.path.exists(local):
            data = cl.download(path)
            tmp = local + ".part"
            with open(tmp, "wb") as fh:
                fh.write(data)
            os.replace(tmp, local)
        return local

    for rq in spec["requests"]:
        sport = rq["sport"]
        d0, d1 = dt.date.fromisoformat(rq["from"]), dt.date.fromisoformat(rq["to"])
        discovery_type = rq.get("discovery_market_type", "MATCH_ODDS")
        extra_types = [t for t in rq.get("market_types", []) if t != discovery_type]
        events = rq.get("events", [])
        for i in range((d1 - d0).days + 1):
            day = d0 + dt.timedelta(days=i)
            paths = cl.post("DownloadListOfFiles", day_filter(sport, plan, day, [discovery_type], rq.get("countries")))
            if not isinstance(paths, list):
                print(f"  warning: unexpected answer for {sport} {day}: {str(paths)[:200]}")
                paths = []
            print(f"{sport} {day}: {len(paths)} {discovery_type} files", flush=True)
            manifest["requests"].append({"sport": sport, "day": day.isoformat(), "type": discovery_type, "n": len(paths)})
            if args.dry_run or not paths:
                continue
            with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
                locals_ = list(ex.map(fetch, paths))
            all_files += paths
            # lenient matching of the spec's events to these markets
            def on_day(e: dict) -> bool:
                if e.get("day"):
                    return e["day"] == day.isoformat()
                if e.get("kickoff_utc"):
                    noon = iso_ms(day.isoformat() + "T12:00:00Z")
                    return abs(iso_ms(e["kickoff_utc"]) - noon) < 36 * 3_600_000
                return True
            day_events = [e for e in events if on_day(e)]
            matched_ids = set()
            for path, local in zip(paths, locals_):
                with open(local, "rb") as fh:
                    md = first_definition(fh.read())
                if not md:
                    continue
                h, a = split_event(md.get("eventName", ""))
                runners = [r.get("name", "") for r in md.get("runners", [])]
                t_ev = iso_ms(md["openDate"]) if md.get("openDate") else None
                for e in day_events:
                    if e.get("kickoff_utc") and t_ev is not None and abs(t_ev - iso_ms(e["kickoff_utc"])) > 90 * 60_000:
                        continue
                    s1 = max([sim(e["home"], h)] + [sim(e["home"], r) for r in runners])
                    s2 = max([sim(e["away"], a)] + [sim(e["away"], r) for r in runners])
                    if min(s1, s2) >= 0.45 or max(s1, s2) >= 0.9:
                        matched_ids.add(str(md.get("eventId")))
                        manifest["matched_events"].append({"spec_event": e, "eventId": md.get("eventId"),
                                                           "eventName": md.get("eventName"), "score": [s1, s2]})
            if extra_types and matched_ids:
                more = cl.post("DownloadListOfFiles", day_filter(sport, plan, day, extra_types, rq.get("countries")))
                more = [p for p in (more if isinstance(more, list) else [])
                        if any(f"/{eid}/" in p for eid in matched_ids)]
                print(f"   matched {len(matched_ids)} events -> {len(more)} extra market files", flush=True)
                with cf.ThreadPoolExecutor(max_workers=args.threads) as ex:
                    list(ex.map(fetch, more))
                all_files += more

    if args.dry_run:
        print("\nDry run finished - no files downloaded.")
        return
    man_path = os.path.join(args.cache, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with tarfile.open(args.out, "w") as tf:
        for p in sorted(set(all_files)):
            local = os.path.join(args.cache, p.strip("/").replace("/", "__"))
            if os.path.exists(local):
                tf.add(local, arcname=p.strip("/"))
        tf.add(man_path, arcname="manifest.json")
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\nDone: {len(set(all_files))} files -> {args.out} ({size_mb:.1f} MB). Upload this file to Google Drive.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted - re-run the same command to resume.")
