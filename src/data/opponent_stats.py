"""
Opponent defense data: build lookup tables and join to game logs.

Season-level opponent stats represent what opponents score against each team
(e.g., opp_def_pts = average points opponents score against this team).
"""

from __future__ import annotations

import logging

import pandas as pd

from src.data.nba_api_client import get_team_opponent_stats, _parse_matchup

logger = logging.getLogger(__name__)

# Columns to keep from opponent stats, with opp_def_ prefix
OPP_STAT_COLUMNS = {
    "PTS": "opp_def_pts",
    "REB": "opp_def_reb",
    "AST": "opp_def_ast",
    "STL": "opp_def_stl",
    "BLK": "opp_def_blk",
    "TOV": "opp_def_tov",
    "FG3M": "opp_def_fg3m",
    "FG_PCT": "opp_def_fg_pct",
    "FG3_PCT": "opp_def_fg3_pct",
}


def build_opponent_defense_lookup(season: str) -> pd.DataFrame | None:
    """Build a lookup table of opponent defense metrics indexed by team abbreviation.

    Prefers computing from game logs (matches our data exactly), falls back to NBA.com.
    Returns DataFrame with TEAM_ABBREVIATION as index and opp_def_* columns,
    or None if data is unavailable.
    """
    df = None

    # Try computing from our own game logs first (no extra API calls, guaranteed match)
    try:
        from src.data.bbref_advanced import compute_team_defense_from_game_logs
        df = compute_team_defense_from_game_logs(season)
    except Exception as e:
        logger.warning(f"Could not compute team defense for {season}: {e}")

    # Fallback to NBA.com endpoint
    if df is None:
        df = get_team_opponent_stats(season)

    if df is None:
        logger.warning(f"Team opponent stats unavailable for {season}")
        return None

    # Rename stat columns with opp_def_ prefix
    rename_map = {k: v for k, v in OPP_STAT_COLUMNS.items() if k in df.columns}
    result = df[["TEAM_ABBREVIATION"] + list(rename_map.keys())].copy()
    result = result.rename(columns=rename_map)

    result = result.set_index("TEAM_ABBREVIATION")
    logger.info(f"Built opponent defense lookup: {len(result)} teams for {season}")
    return result


def join_opponent_defense(game_logs: pd.DataFrame, season: str) -> pd.DataFrame:
    """Join opponent defense stats to game log rows.

    Each game row gets the season-average defensive stats of the opponent team.
    Returns game_logs unchanged if opponent data is unavailable.
    """
    # Ensure opponent column exists
    if "opponent" not in game_logs.columns:
        parsed = game_logs["MATCHUP"].apply(_parse_matchup)
        game_logs = game_logs.copy()
        game_logs["opponent"] = parsed.apply(lambda x: x[0])

    lookup = build_opponent_defense_lookup(season)

    if lookup is None:
        logger.warning(f"Opponent defense data unavailable for {season}, skipping join")
        return game_logs

    # Left join on opponent abbreviation
    result = game_logs.merge(
        lookup,
        left_on="opponent",
        right_index=True,
        how="left",
    )

    missing = result["opp_def_pts"].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} rows missing opponent defense data")

    return result
