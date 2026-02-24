"""
Map Sleeper player names to Basketball Reference slugs.

Three-tier lookup:
1. Historical parquet data (924 players across 4 seasons — covers ~95%)
2. BBRef search API (for new players not in historical data)
3. Algorithmic slug construction (last resort)

Resolved mappings are cached to bbref_slug_cache.json.
"""

from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
SLUG_CACHE_PATH = Path(__file__).resolve().parent / "bbref_slug_cache.json"
BBREF_DELAY = 3.0

# In-memory caches (populated lazily)
_parquet_lookup: dict[str, str] | None = None
_slug_cache: dict[str, str] | None = None


def _normalize_name(name: str) -> str:
    """Normalize a player name for matching.

    Strips diacritics, lowercases, normalizes suffixes and whitespace.
    """
    # Strip diacritics (é → e, ć → c, etc.)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))

    # Lowercase and strip
    result = ascii_name.lower().strip()

    # Normalize common suffixes
    result = re.sub(r"\s+(jr\.?|sr\.?|ii|iii|iv)$", lambda m: " " + m.group(1).rstrip("."), result)

    # Collapse whitespace
    result = re.sub(r"\s+", " ", result)

    return result


def _build_parquet_lookup() -> dict[str, str]:
    """Build name→slug mapping from historical parquet files."""
    mappings: dict[str, str] = {}
    for f in sorted(DATA_DIR.glob("player_game_logs_*.parquet")):
        try:
            df = pd.read_parquet(f, columns=["PLAYER_NAME", "PLAYER_ID"])
            for _, row in df.drop_duplicates("PLAYER_ID").iterrows():
                norm = _normalize_name(str(row["PLAYER_NAME"]))
                mappings[norm] = str(row["PLAYER_ID"])
        except Exception as e:
            logger.warning(f"Could not read {f.name} for slug lookup: {e}")
    logger.info(f"Built parquet slug lookup: {len(mappings)} players")
    return mappings


def _get_parquet_lookup() -> dict[str, str]:
    """Get or build the parquet-derived lookup (lazy singleton)."""
    global _parquet_lookup
    if _parquet_lookup is None:
        _parquet_lookup = _build_parquet_lookup()
    return _parquet_lookup


def _load_slug_cache() -> dict[str, str]:
    """Load the persistent slug cache from disk."""
    global _slug_cache
    if _slug_cache is not None:
        return _slug_cache
    if SLUG_CACHE_PATH.exists():
        try:
            with open(SLUG_CACHE_PATH) as f:
                _slug_cache = json.load(f)
        except Exception:
            _slug_cache = {}
    else:
        _slug_cache = {}
    return _slug_cache


def _save_slug_cache() -> None:
    """Persist the slug cache to disk."""
    if _slug_cache is not None:
        with open(SLUG_CACHE_PATH, "w") as f:
            json.dump(_slug_cache, f, indent=2, sort_keys=True)


def _search_bbref_slug(full_name: str) -> str | None:
    """Use BBRef search to find a player's slug."""
    try:
        from basketball_reference_web_scraper import client as bbref
        from basketball_reference_web_scraper.data import League

        time.sleep(BBREF_DELAY)
        results = bbref.search(full_name)
        players = results.get("players", [])
        for p in players:
            leagues = p.get("leagues", set())
            if League.NATIONAL_BASKETBALL_ASSOCIATION in leagues:
                slug = p.get("identifier")
                if slug:
                    logger.info(f"BBRef search found slug for '{full_name}': {slug}")
                    return slug
    except Exception as e:
        logger.warning(f"BBRef search failed for '{full_name}': {e}")
    return None


def _construct_slug_guess(full_name: str) -> str:
    """Algorithmically construct a BBRef slug guess.

    Pattern: first 5 chars of last name + first 2 of first name + "01"
    Example: "LeBron James" → "jamesle01"
    """
    norm = _normalize_name(full_name)
    # Remove suffixes for slug construction
    norm = re.sub(r"\s+(jr|sr|ii|iii|iv)$", "", norm)
    parts = norm.split()
    if len(parts) < 2:
        return norm[:7] + "01"
    first = parts[0]
    last = parts[-1]
    return (last[:5] + first[:2] + "01").lower()


def get_bbref_slug(full_name: str) -> str | None:
    """Look up BBRef slug for a player name.

    Tries: persistent cache → parquet lookup → BBRef search → algorithmic guess.
    Returns None if all methods fail.
    """
    norm = _normalize_name(full_name)

    # 1. Check persistent cache
    cache = _load_slug_cache()
    if norm in cache:
        return cache[norm]

    # 2. Check parquet-derived lookup
    lookup = _get_parquet_lookup()
    if norm in lookup:
        slug = lookup[norm]
        cache[norm] = slug
        _save_slug_cache()
        return slug

    # 3. Try BBRef search API
    slug = _search_bbref_slug(full_name)
    if slug:
        cache[norm] = slug
        _save_slug_cache()
        return slug

    # 4. Algorithmic guess — validate by trying to fetch
    guess = _construct_slug_guess(full_name)
    try:
        from basketball_reference_web_scraper import client as bbref

        time.sleep(BBREF_DELAY)
        boxes = bbref.regular_season_player_box_scores(
            player_identifier=guess, season_end_year=2026,
        )
        if boxes:
            logger.info(f"Algorithmic slug guess worked for '{full_name}': {guess}")
            cache[norm] = guess
            _save_slug_cache()
            return guess
    except Exception:
        pass

    logger.warning(f"Could not resolve BBRef slug for '{full_name}'")
    return None
