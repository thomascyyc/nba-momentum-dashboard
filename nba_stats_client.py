"""
NBA stats client using Basketball Reference for per-game box score data.

Maps Sleeper player names to BBRef slugs, then fetches game logs.
Includes rate-limiting delays to respect BBRef scraping limits.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime

import requests
from lxml import html as lxml_html

from basketball_reference_web_scraper import client as bbref

from bbref_slug_map import get_bbref_slug
from config import NBA_API_SEASON
from src.data.nba_api_client import _team_to_abbrev

logger = logging.getLogger(__name__)

# Delay between BBRef requests (respectful scraping)
BBREF_DELAY = 2.0  # seconds

MAX_RETRIES = 2
RETRY_DELAYS = [5, 10]

# BBRef base URL
_BBREF_BASE = "https://www.basketball-reference.com"

# BBRef 3-letter abbreviations used on game log pages
_BBREF_TEAM_MAP = {
    "ATL": "ATL", "BOS": "BOS", "BRK": "BKN", "CHO": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GSW": "GSW",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NOP": "NOP", "NYK": "NYK",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHO": "PHX", "POR": "POR",
    "SAC": "SAC", "SAS": "SAS", "TOR": "TOR", "UTA": "UTA", "WAS": "WAS",
}


def _normalize_team(abbrev: str) -> str:
    """Normalize BBRef team abbreviation to standard 3-letter code."""
    return _BBREF_TEAM_MAP.get(abbrev, abbrev)


def find_nba_player_id(full_name):
    """
    Look up a player's BBRef slug by full name.

    Returns (slug, matched_name) or (None, None) if not found.
    Kept as find_nba_player_id for backward compatibility with cache.py.
    """
    slug = get_bbref_slug(full_name)
    if slug:
        return slug, full_name
    return None, None


def get_player_game_log(player_id, season=NBA_API_SEASON, last_n_games=0):
    """
    Fetch a player's game log from Basketball Reference.

    Args:
        player_id: BBRef player slug (e.g., "anunoog01")
        season: Season string like "2025-26"
        last_n_games: If > 0, only return this many recent games. 0 = all games.

    Returns list of dicts, each with:
        - date, matchup, result (game context)
        - pts, reb, ast, stl, blk, to, tpm (stats for scoring)
        - min (minutes played)
    """
    season_end_year = int(season.split("-")[0]) + 1

    # Try the library first (works for most players)
    games = _fetch_via_library(player_id, season_end_year)

    # If library fails (DNP/Inactive parsing bug), fall back to direct HTML scrape
    if games is None:
        logger.info(f"Library failed for {player_id}, falling back to HTML scrape")
        games = _fetch_via_html(player_id, season_end_year)

    if games is None:
        return []

    if last_n_games > 0:
        games = games[:last_n_games]

    return games


def _fetch_via_library(player_id, season_end_year):
    """Try fetching game log via the basketball_reference_web_scraper library."""
    try:
        time.sleep(BBREF_DELAY)
        boxes = bbref.regular_season_player_box_scores(
            player_identifier=player_id,
            season_end_year=season_end_year,
        )
    except Exception as e:
        logger.debug(f"Library fetch failed for {player_id}: {e}")
        return None

    games = []
    for b in boxes:
        if not b.get("active", True):
            continue

        team_abbrev = _team_to_abbrev(b["team"])
        opp_abbrev = _team_to_abbrev(b["opponent"])

        location = b.get("location")
        is_home = str(getattr(location, "value", location)) == "HOME"
        matchup = f"{team_abbrev} vs. {opp_abbrev}" if is_home else f"{team_abbrev} @ {opp_abbrev}"

        outcome = b.get("outcome")
        wl = "W" if str(getattr(outcome, "value", outcome)) == "WIN" else "L"

        seconds = b.get("seconds_played", 0) or 0
        minutes = round(seconds / 60, 1)

        game_date = b["date"]  # datetime.date object
        date_str = game_date.strftime("%b %d, %Y")

        oreb = b.get("offensive_rebounds", 0) or 0
        dreb = b.get("defensive_rebounds", 0) or 0

        game = {
            "date": date_str,
            "matchup": matchup,
            "result": wl,
            "min": minutes,
            "pts": b.get("points_scored", 0) or 0,
            "reb": oreb + dreb,
            "ast": b.get("assists", 0) or 0,
            "stl": b.get("steals", 0) or 0,
            "blk": b.get("blocks", 0) or 0,
            "to": b.get("turnovers", 0) or 0,
            "tpm": b.get("made_three_point_field_goals", 0) or 0,
        }
        games.append(game)

    # BBRef returns oldest first; reverse to newest first
    games.reverse()
    return games


def _fetch_via_html(player_id, season_end_year):
    """Fallback: directly scrape BBRef HTML for players the library can't handle.

    Handles pages with DNP/Inactive rows that crash the library's parser.
    Supports both old (pgl_basic) and new (player_game_log_reg) table formats.
    """
    url = (
        f"{_BBREF_BASE}/players/{player_id[0]}/{player_id}"
        f"/gamelog/{season_end_year}"
    )

    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(BBREF_DELAY)
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAYS[attempt])
            else:
                logger.error(f"HTML fetch failed for {player_id}: {e}")
                return None

    tree = lxml_html.fromstring(resp.content)

    # Try both table IDs (BBRef uses different IDs on different pages)
    table = None
    for table_id in ["pgl_basic", "player_game_log_reg"]:
        found = tree.xpath(f'//table[@id="{table_id}"]')
        if found:
            table = found[0]
            break

    if table is None:
        logger.warning(f"No game log table found for {player_id}")
        return []

    rows = table.xpath('.//tbody/tr')
    games = []

    # Column name mapping: old format → new format alternatives
    # (we check both; whichever exists wins)
    _STAT_NAMES = {
        "date": ["date_game", "date"],
        "team": ["team_id", "team_name_abbr"],
        "opp": ["opp_id", "opp_name_abbr"],
        "location": ["game_location"],
        "result": ["game_result"],
        "minutes": ["mp"],
        "pts": ["pts"],
        "reb": ["trb"],
        "ast": ["ast"],
        "stl": ["stl"],
        "blk": ["blk"],
        "tov": ["tov"],
        "fg3": ["fg3"],
        "reason": ["reason"],
        "starter": ["is_starter"],
    }

    def cell(row, key):
        for stat_name in _STAT_NAMES.get(key, [key]):
            found = row.xpath(f'.//td[@data-stat="{stat_name}"]')
            if found:
                return found[0].text_content().strip()
        return ""

    def safe_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    for row in rows:
        # Skip header rows within tbody
        cls = row.get("class", "")
        if "thead" in cls or "partial_table" in cls:
            continue

        # Skip inactive/DNP rows (check both formats)
        reason = cell(row, "reason")
        starter = cell(row, "starter")
        if reason or starter in ("Inactive", "Did Not Play", "Did Not Dress", "Not With Team"):
            continue

        mp = cell(row, "minutes")
        if not mp or ":" not in mp:
            continue

        date_str = cell(row, "date")
        if not date_str:
            continue

        team_abbrev = _normalize_team(cell(row, "team"))
        opp_abbrev = _normalize_team(cell(row, "opp"))
        location = cell(row, "location")
        is_home = location != "@"
        matchup = f"{team_abbrev} vs. {opp_abbrev}" if is_home else f"{team_abbrev} @ {opp_abbrev}"

        result_text = cell(row, "result")
        wl = "W" if result_text.startswith("W") else "L"

        # Parse minutes from MM:SS
        parts = mp.split(":")
        minutes = round(int(parts[0]) + int(parts[1]) / 60, 1)

        # Format date (BBRef may use "2026-02-22" or "Feb 22, 2026")
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_formatted = dt.strftime("%b %d, %Y")
        except ValueError:
            date_formatted = date_str

        game = {
            "date": date_formatted,
            "matchup": matchup,
            "result": wl,
            "min": minutes,
            "pts": safe_int(cell(row, "pts")),
            "reb": safe_int(cell(row, "reb")),
            "ast": safe_int(cell(row, "ast")),
            "stl": safe_int(cell(row, "stl")),
            "blk": safe_int(cell(row, "blk")),
            "to": safe_int(cell(row, "tov")),
            "tpm": safe_int(cell(row, "fg3")),
        }
        games.append(game)

    # Reverse to newest first
    games.reverse()
    logger.info(f"HTML scrape for {player_id}: {len(games)} active games")
    return games


def get_games_for_player(player_name, last_n_games=5, season=NBA_API_SEASON):
    """
    High-level function: look up a player by name and return recent game logs.

    Args:
        player_name: Full name as it appears in Sleeper (e.g. "Anthony Edwards")
        last_n_games: Number of recent games to return
        season: NBA season string

    Returns:
        (player_info, games) where:
        - player_info = {"nba_id": slug, "matched_name": str}
        - games = list of game stat dicts (see get_player_game_log)

    Returns (None, []) if the player can't be found.
    """
    slug, matched_name = find_nba_player_id(player_name)
    if slug is None:
        print(f"  Warning: Could not find '{player_name}' on Basketball Reference")
        return None, []

    player_info = {"nba_id": slug, "matched_name": matched_name}
    games = get_player_game_log(slug, season=season, last_n_games=last_n_games)
    return player_info, games
