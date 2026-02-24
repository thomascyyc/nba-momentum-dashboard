"""
Fetch advanced player stats and compute team defense ratings from Basketball Reference.

Replaces the NBA.com bulk endpoints that are blocked.
Advanced stats: BBRef's players_advanced_season_totals endpoint.
Team defense: computed from existing game log data (no extra API calls).
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

import pandas as pd

from src.data.nba_api_client import (
    save_to_parquet, load_from_parquet, _team_to_abbrev, BBREF_DELAY,
)

logger = logging.getLogger(__name__)


# ── Player Advanced Stats ────────────────────────────────────────

def fetch_player_advanced_stats_bbref(season: str) -> pd.DataFrame:
    """Fetch season-level advanced stats from Basketball Reference.

    Returns DataFrame with PLAYER_ID and advanced stat columns matching
    the naming convention used by feature_engineering.py.
    """
    from basketball_reference_web_scraper import client as bbref

    year = int(season.split("-")[0]) + 1
    time.sleep(BBREF_DELAY)

    data = bbref.players_advanced_season_totals(season_end_year=year)
    df = pd.DataFrame(data)

    # Filter out combined totals rows (players traded mid-season)
    if "is_combined_totals" in df.columns:
        df = df[df["is_combined_totals"] == False].copy()

    # Rename BBRef columns to match nba_api naming used in feature_engineering.py
    rename_map = {
        "slug": "PLAYER_ID",
        "usage_percentage": "USG_PCT",
        "true_shooting_percentage": "TS_PCT",
        "player_efficiency_rating": "PIE",  # PER as proxy for PIE
        "assist_percentage": "AST_PCT",
        "offensive_box_plus_minus": "OFF_RATING",  # OBPM as proxy
        "defensive_box_plus_minus": "DEF_RATING",  # DBPM as proxy
        "box_plus_minus": "NET_RATING",  # BPM as proxy
    }
    actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=actual_rename)

    # Keep only columns we need (deduplicate since PLAYER_ID is in rename_map values)
    keep = list(dict.fromkeys(
        ["PLAYER_ID"] + [v for v in rename_map.values() if v in df.columns]
    ))
    df = df[[c for c in keep if c in df.columns]].copy()

    # Drop duplicates (keep first, i.e. the team they played most games for)
    df = df.drop_duplicates(subset="PLAYER_ID", keep="first")

    logger.info(f"BBRef advanced stats for {season}: {len(df)} players")
    return df


def get_player_advanced_stats_bbref(
    season: str, force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Get advanced stats with parquet cache, BBRef fallback."""
    filename = f"player_advanced_stats_bbref_{season}.parquet"
    if not force_refresh:
        df = load_from_parquet(filename)
        if df is not None:
            return df
    try:
        df = fetch_player_advanced_stats_bbref(season)
        save_to_parquet(df, filename)
        return df
    except Exception as e:
        logger.warning(f"Could not fetch BBRef advanced stats for {season}: {e}")
        return None


# ── Team Defense Ratings (computed from game logs) ────────────────

def compute_team_defense_from_game_logs(
    season: str, force_refresh: bool = False,
) -> pd.DataFrame | None:
    """Compute team defense ratings from existing player game log data.

    For each team, calculates average stats scored by opponents per game.
    No extra API calls needed — uses cached parquet game logs.

    Returns DataFrame indexed by TEAM_ABBREVIATION with columns:
    PTS, REB, AST, STL, BLK, TOV, FG3M, FG_PCT, FG3_PCT
    """
    filename = f"team_defense_ratings_{season}.parquet"
    if not force_refresh:
        df = load_from_parquet(filename)
        if df is not None:
            return df

    # Load raw game logs
    game_logs = load_from_parquet(f"player_game_logs_{season}.parquet")
    if game_logs is None:
        logger.warning(f"No game logs found for {season}, cannot compute team defense")
        return None

    # For each game (game_date + opponent), sum all player stats
    # "opponent" is the team being played against — their defense is what we measure
    stat_cols = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
                 "FGM", "FGA", "FG3A"]

    # Group by game date + opponent to get per-game totals scored against each team
    game_totals = (
        game_logs
        .groupby(["game_date_parsed", "opponent"])[stat_cols]
        .sum()
        .reset_index()
    )

    # Now average across all games for each team
    team_defense = (
        game_totals
        .groupby("opponent")[stat_cols]
        .mean()
        .reset_index()
    )

    # Compute shooting percentages
    team_defense["FG_PCT"] = (
        team_defense["FGM"] / team_defense["FGA"]
    ).round(3)
    team_defense["FG3_PCT"] = (
        team_defense["FG3M"] / team_defense["FG3A"]
    ).round(3)

    # Rename opponent to TEAM_ABBREVIATION for joining
    team_defense = team_defense.rename(columns={"opponent": "TEAM_ABBREVIATION"})

    # Drop intermediate columns
    team_defense = team_defense.drop(columns=["FGM", "FGA", "FG3A"], errors="ignore")

    save_to_parquet(team_defense, filename)
    logger.info(f"Computed team defense ratings for {season}: {len(team_defense)} teams")
    return team_defense
