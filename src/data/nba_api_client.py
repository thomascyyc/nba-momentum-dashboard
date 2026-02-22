"""
NBA data fetching for historical player game logs.

Primary source: Basketball Reference (via basketball_reference_web_scraper).
Fallback: nba_api per-player endpoint (when BBRef is unavailable).
Advanced/opponent stats: nba_api league-wide endpoints (optional, graceful fallback).

All data is cached as parquet files with API fallback when files are missing.
"""

from __future__ import annotations

import time
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BBREF_DELAY = 3.0  # seconds between BBRef requests (respectful scraping)
NBA_API_DELAY = 0.6  # seconds between nba_api calls
MAX_RETRIES = 3
RETRY_DELAYS = [3, 6, 12]

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
HISTORICAL_SEASONS = ["2021-22", "2022-23", "2023-24"]

# Map BBRef team enums to standard 3-letter abbreviations
_TEAM_ABBREV = {
    "ATLANTA HAWKS": "ATL", "BOSTON CELTICS": "BOS", "BROOKLYN NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA", "CHICAGO BULLS": "CHI", "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL", "DENVER NUGGETS": "DEN", "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW", "HOUSTON ROCKETS": "HOU", "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC", "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM", "MIAMI HEAT": "MIA", "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN", "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK", "OKLAHOMA CITY THUNDER": "OKC", "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI", "PHOENIX SUNS": "PHX", "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC", "SAN ANTONIO SPURS": "SAS", "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA", "WASHINGTON WIZARDS": "WAS",
}


def _team_to_abbrev(team_enum) -> str:
    """Convert BBRef Team enum to 3-letter abbreviation."""
    name = str(team_enum.value) if hasattr(team_enum, "value") else str(team_enum)
    return _TEAM_ABBREV.get(name, name[:3].upper())


# ── Parquet I/O ──────────────────────────────────────────────────

def _parquet_engine() -> str:
    """Pick parquet engine: prefer pyarrow, fall back to fastparquet."""
    try:
        import pyarrow  # noqa: F401
        from importlib.metadata import version as pkg_version
        ver = tuple(int(x) for x in pkg_version("pyarrow").split(".")[:2])
        if ver >= (10, 0):
            return "pyarrow"
    except Exception:
        pass
    return "fastparquet"


def save_to_parquet(df: pd.DataFrame, filename: str) -> Path:
    """Save DataFrame to parquet in data/ directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.to_parquet(path, index=False, engine=_parquet_engine())
    logger.info(f"Saved {len(df)} rows to {path}")
    return path


def load_from_parquet(filename: str) -> pd.DataFrame | None:
    """Load DataFrame from parquet if file exists, else None."""
    path = DATA_DIR / filename
    if path.exists():
        df = pd.read_parquet(path, engine=_parquet_engine())
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df
    return None


# ── Helpers ──────────────────────────────────────────────────────

def _parse_matchup(matchup: str) -> tuple[str, bool]:
    """Parse MATCHUP string into (opponent_abbrev, is_home).

    'LAL vs. GSW' -> ('GSW', True)
    'LAL @ GSW'   -> ('GSW', False)
    """
    is_home = "vs." in matchup
    opponent = matchup.split(" ")[-1]
    return opponent, is_home


def _compute_dd2_td3(pts, reb, ast, stl, blk) -> tuple[int, int]:
    """Compute double-double and triple-double flags from raw stats."""
    cats = sum(1 for v in [pts, reb, ast, stl, blk] if (v or 0) >= 10)
    return (1 if cats >= 2 else 0, 1 if cats >= 3 else 0)


def _season_end_year(season: str) -> int:
    """Convert '2023-24' to 2024."""
    return int(season.split("-")[0]) + 1


def _api_call_with_retry(fetch_fn, description: str, retries: int = MAX_RETRIES):
    """Execute an API call with rate limiting and exponential backoff retry."""
    for attempt in range(retries):
        try:
            time.sleep(NBA_API_DELAY)
            result = fetch_fn()
            return result
        except Exception as e:
            if attempt < retries - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                logger.warning(
                    f"{description}: attempt {attempt + 1} failed ({e}), "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"{description}: all {retries} attempts failed")
                raise


# ── Player Game Logs (Basketball Reference) ──────────────────────

def _get_game_dates(season_end_year: int) -> list[date]:
    """Get all unique game dates for a season from BBRef schedule."""
    from basketball_reference_web_scraper import client as bbref

    sched = bbref.season_schedule(season_end_year=season_end_year)

    # Filter to regular season only (before mid-April typically)
    # BBRef schedule includes playoffs; we filter by date range
    all_dates = sorted(set(s["start_time"].date() for s in sched))

    # Regular season typically ends mid-April
    # Filter out anything after April 20 (safe cutoff for regular season)
    cutoff = date(season_end_year, 4, 20)
    reg_dates = [d for d in all_dates if d <= cutoff]

    logger.info(f"  {len(reg_dates)} regular season game dates "
               f"({reg_dates[0]} to {reg_dates[-1]})")
    return reg_dates


def fetch_player_game_logs(season: str) -> pd.DataFrame:
    """Fetch all player game log rows for a season from Basketball Reference.

    Iterates over each game date and fetches all player box scores.
    ~180 game dates per season, ~3s between requests = ~9 minutes.
    """
    from basketball_reference_web_scraper import client as bbref

    year = _season_end_year(season)
    logger.info(f"Fetching player game logs for {season} from Basketball Reference...")

    game_dates = _get_game_dates(year)
    all_rows = []
    errors = 0

    for i, d in enumerate(game_dates):
        try:
            boxes = bbref.player_box_scores(day=d.day, month=d.month, year=d.year)

            for b in boxes:
                # Skip inactive players (DNP, etc.)
                if not b.get("active", True):
                    continue

                team_abbrev = _team_to_abbrev(b["team"])
                opp_abbrev = _team_to_abbrev(b["opponent"])
                is_home = str(b["location"].value) == "HOME" if hasattr(b["location"], "value") else b["location"] == "HOME"
                matchup = f"{team_abbrev} vs. {opp_abbrev}" if is_home else f"{team_abbrev} @ {opp_abbrev}"

                seconds = b.get("seconds_played", 0) or 0
                minutes = round(seconds / 60, 1)
                pts = b.get("points_scored", 0) or 0
                oreb = b.get("offensive_rebounds", 0) or 0
                dreb = b.get("defensive_rebounds", 0) or 0
                reb = oreb + dreb
                ast = b.get("assists", 0) or 0
                stl = b.get("steals", 0) or 0
                blk = b.get("blocks", 0) or 0
                tov = b.get("turnovers", 0) or 0
                fg3m = b.get("made_three_point_field_goals", 0) or 0

                dd2, td3 = _compute_dd2_td3(pts, reb, ast, stl, blk)

                wl = str(b.get("outcome", "")).split(".")[-1] if b.get("outcome") else ""
                if hasattr(b.get("outcome"), "value"):
                    wl = "W" if b["outcome"].value == "WIN" else "L"

                row = {
                    "PLAYER_NAME": b.get("name", ""),
                    "PLAYER_ID": b.get("slug", ""),  # BBRef slug as ID
                    "TEAM_ABBREVIATION": team_abbrev,
                    "SEASON_YEAR": season,
                    "GAME_DATE": d.strftime("%b %d, %Y"),
                    "MATCHUP": matchup,
                    "WL": wl,
                    "MIN": minutes,
                    "PTS": pts,
                    "REB": reb,
                    "AST": ast,
                    "STL": stl,
                    "BLK": blk,
                    "TOV": tov,
                    "FG3M": fg3m,
                    "FGM": b.get("made_field_goals", 0) or 0,
                    "FGA": b.get("attempted_field_goals", 0) or 0,
                    "FG3A": b.get("attempted_three_point_field_goals", 0) or 0,
                    "FTM": b.get("made_free_throws", 0) or 0,
                    "FTA": b.get("attempted_free_throws", 0) or 0,
                    "OREB": oreb,
                    "DREB": dreb,
                    "PF": b.get("personal_fouls", 0) or 0,
                    "PLUS_MINUS": b.get("plus_minus", 0) or 0,
                    "DD2": dd2,
                    "TD3": td3,
                    "NBA_FANTASY_PTS": np.nan,
                    "game_date_parsed": pd.Timestamp(d),
                    "opponent": opp_abbrev,
                    "is_home": is_home,
                }

                # Compute shooting percentages
                fga = row["FGA"]
                row["FG_PCT"] = round(row["FGM"] / fga, 3) if fga > 0 else 0.0
                fg3a = row["FG3A"]
                row["FG3_PCT"] = round(row["FG3M"] / fg3a, 3) if fg3a > 0 else 0.0
                fta = row["FTA"]
                row["FT_PCT"] = round(row["FTM"] / fta, 3) if fta > 0 else 0.0

                all_rows.append(row)

            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i+1}/{len(game_dates)} dates, "
                           f"{len(all_rows)} box scores so far")

        except Exception as e:
            logger.warning(f"  Error fetching {d}: {e}")
            errors += 1

        time.sleep(BBREF_DELAY)

    if not all_rows:
        raise RuntimeError(f"No game logs fetched for {season}")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)

    n_players = df["PLAYER_ID"].nunique()
    logger.info(f"  {len(df)} player-game rows for {season} "
               f"({n_players} players, {errors} date errors)")
    return df


def get_player_game_logs(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """Get player game logs with parquet-first, API fallback."""
    filename = f"player_game_logs_{season}.parquet"
    if not force_refresh:
        df = load_from_parquet(filename)
        if df is not None:
            return df
    df = fetch_player_game_logs(season)
    save_to_parquet(df, filename)
    return df


# ── Advanced Player Stats (nba_api, optional) ────────────────────

def fetch_player_advanced_stats(season: str) -> pd.DataFrame:
    """Fetch season-level advanced stats for all players from NBA.com."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    logger.info(f"Fetching advanced player stats for {season}...")

    def _call():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=120,
        ).get_data_frames()[0]

    df = _api_call_with_retry(_call, f"LeagueDashPlayerStats(Advanced, {season})")
    logger.info(f"  {len(df)} players with advanced stats for {season}")
    return df


def get_player_advanced_stats(season: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """Get advanced player stats with parquet-first, API fallback.

    Returns None if data is unavailable (endpoint blocked).
    """
    filename = f"player_advanced_stats_{season}.parquet"
    if not force_refresh:
        df = load_from_parquet(filename)
        if df is not None:
            return df
    try:
        df = fetch_player_advanced_stats(season)
        save_to_parquet(df, filename)
        return df
    except Exception as e:
        logger.warning(f"Could not fetch advanced stats for {season}: {e}")
        return None


# ── Team Opponent Stats (nba_api, optional) ──────────────────────

def fetch_team_opponent_stats(season: str) -> pd.DataFrame:
    """Fetch team-level opponent stats from NBA.com."""
    from nba_api.stats.endpoints import leaguedashteamstats

    logger.info(f"Fetching team opponent stats for {season}...")

    def _call():
        return leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            measure_type_detailed_defense="Opponent",
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=120,
        ).get_data_frames()[0]

    df = _api_call_with_retry(_call, f"LeagueDashTeamStats(Opponent, {season})")
    logger.info(f"  {len(df)} teams with opponent stats for {season}")
    return df


def get_team_opponent_stats(season: str, force_refresh: bool = False) -> pd.DataFrame | None:
    """Get team opponent stats with parquet-first, API fallback.

    Returns None if data is unavailable (endpoint blocked).
    """
    filename = f"team_opponent_stats_{season}.parquet"
    if not force_refresh:
        df = load_from_parquet(filename)
        if df is not None:
            return df
    try:
        df = fetch_team_opponent_stats(season)
        save_to_parquet(df, filename)
        return df
    except Exception as e:
        logger.warning(f"Could not fetch team opponent stats for {season}: {e}")
        return None
