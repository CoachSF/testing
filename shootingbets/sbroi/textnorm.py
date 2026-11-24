"""Team / player name normalisation and fuzzy comparison across bookmakers."""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

# Tokens that carry no identity (club-type prefixes/suffixes).
_DROP = {
    "fc", "cf", "sc", "afc", "ac", "as", "ss", "us", "cd", "sd", "ud", "rc", "rcd", "sk", "fk",
    "nk", "bk", "ik", "if", "ff", "sv", "tsv", "fsv", "vfb", "vfl", "club", "de", "the", "cp",
    "ca", "cs", "ks", "mfk", "ofk", "fbc", "sad", "kv", "krc", "ssc", "acf", "calcio",
    "football", "futbol", "clube", "esporte", "society", "deportivo",
}
# Tokens that distinguish teams and must be canonicalised, never dropped.
_CANON = [
    (r"\b(women|woman|ladies|fem|femenino|feminine|feminino|feminin|frauen|damen|dames)\b", " women "),
    (r"\(w\)|\bw\b", " women "),
    (r"\bunder[\s-]?(\d{2})\b|\bu[\s-]?(\d{2})\b|\bsub[\s-]?(\d{2})\b", None),  # handled in code
    (r"\b(reserves|reserve|res)\b", " res "),
    (r"\butd\b", " united "),
    (r"\bst\b\.?", " saint "),
    (r"&", " and "),
]
_ALIASES = {
    "man united": "manchester united",
    "man utd": "manchester united",
    "man city": "manchester city",
    "nottm forest": "nottingham forest",
    "sheff united": "sheffield united",
    "sheff wed": "sheffield wednesday",
    "wolves": "wolverhampton",
    "spurs": "tottenham",
    "qpr": "queens park rangers",
    "west brom": "west bromwich",
    "paris saint g": "paris saint germain",
    "paris sg": "paris saint germain",
    "psg": "paris saint germain",
    "atl madrid": "atletico madrid",
    "atletico de madrid": "atletico madrid",
    "inter milan": "inter",
    "internazionale": "inter",
    "bayern munchen": "bayern munich",
    "bayern muenchen": "bayern munich",
    "mgladbach": "monchengladbach",
    "b monchengladbach": "monchengladbach",
    "borussia monchengladbach": "monchengladbach",
    "borussia dortmund": "dortmund",
    "athletic bilbao": "athletic club",
    "ath bilbao": "athletic club",
    "ath madrid": "atletico madrid",
    "sporting lisbon": "sporting",
    "olympiacos": "olympiakos",
    "red star belgrade": "crvena zvezda",
}


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch)).replace("ø", "o").replace("ß", "ss").replace("ł", "l").replace("đ", "d")


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    s = strip_accents(str(name)).lower()
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[\.']", "", s)  # St. -> St, O'Neill -> ONeill
    for pat, rep in _CANON:
        if rep is None:
            s = re.sub(pat, lambda m: f" u{next(g for g in m.groups() if g)} ", s)
        else:
            s = re.sub(pat, rep, s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for k, v in _ALIASES.items():
        s = re.sub(rf"(^|\s){re.escape(k)}(\s|$)", rf"\1{v}\2", s)
    toks = [t for t in s.split() if t not in _DROP]
    return " ".join(toks) if toks else s


_SPLIT_RE = re.compile(r"\s+(?:v|vs|vs\.|versus|-|–|—|@|x)\s+", re.IGNORECASE)


def split_event_name(event_name: str) -> tuple[str, str, bool] | None:
    """Split 'Home v Away' / 'Away @ Home' into (home, away, was_at_format)."""
    if not event_name:
        return None
    if " @ " in event_name:
        a, h = event_name.split(" @ ", 1)
        return (h.strip(), a.strip(), True)
    parts = _SPLIT_RE.split(event_name, maxsplit=1)
    if len(parts) != 2:
        return None
    return (parts[0].strip(), parts[1].strip(), False)


def name_score(a: str | None, b: str | None) -> float:
    """0-100 similarity of two team/player names after normalisation."""
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 100.0
    # token_set_ratio scores 100 whenever one name's tokens are a subset of the
    # other's ("Real Madrid" vs "Real Madrid Castilla"), so cap it below an exact match.
    s1 = min(fuzz.token_set_ratio(na, nb), 92.0)
    s2 = fuzz.token_sort_ratio(na, nb)
    s3 = fuzz.partial_ratio(na, nb) if min(len(na), len(nb)) >= 4 else 0.0
    # distinguishing markers must agree (U21 vs senior, women vs men, B/II/res teams)
    markers = {"women", "res", "ii", "b", "u17", "u18", "u19", "u20", "u21", "u23"}
    ma = {t for t in na.split() if t in markers}
    mb = {t for t in nb.split() if t in markers}
    penalty = 25.0 if ma != mb else 0.0
    return max(0.0, max(s1, (s2 + s3) / 2) - penalty)


def surname_score(a: str | None, b: str | None) -> float:
    """Tennis-style comparison: Betfair event names often use surnames only."""
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return 0.0
    la, lb = na.split()[-1], nb.split()[-1]
    return max(name_score(a, b), 100.0 if la == lb and len(la) > 2 else 0.0)
