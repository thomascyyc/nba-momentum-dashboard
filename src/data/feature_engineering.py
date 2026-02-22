"""
Feature engineering for ML-ready fantasy basketball dataset.

Transforms raw NBA game logs into features: fantasy points (using the league's
custom scoring), rolling averages, trend metrics, contextual features (rest,
home/away, B2B), and joins for advanced stats and opponent defense.
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root so we can import scoring.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scoring import calculate_fantasy_points
from src.data.nba_api_client import get_player_advanced_stats
from src.data.opponent_stats import join_opponent_defense

logger = logging.getLogger(__name__)

# Map nba_api columns to Sleeper-compatible stat keys for scoring
# Matches pattern from nba_stats_client.py lines 18-26
STAT_COLUMN_MAP = {
    "PTS": "pts",
    "REB": "reb",
    "AST": "ast",
    "STL": "stl",
    "BLK": "blk",
    "TOV": "to",
    "FG3M": "tpm",
}

ROLLING_WINDOWS = [3, 5, 10, 15]

ROLLING_STAT_COLS = ["fpts", "PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"]


# ── Fantasy Points ───────────────────────────────────────────────

def compute_fantasy_points(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate fantasy points for each game row using the league's scoring.

    Reuses scoring.calculate_fantasy_points() from the project root to ensure
    consistency with the existing dashboard.
    """
    df = df.copy()

    def _calc_row(row):
        stats = {}
        for nba_col, sleeper_key in STAT_COLUMN_MAP.items():
            stats[sleeper_key] = row.get(nba_col, 0) or 0
        # DD2/TD3 columns from nba_api (1/0 flags)
        if "DD2" in row and pd.notna(row["DD2"]):
            stats["dd"] = int(row["DD2"])
        if "TD3" in row and pd.notna(row["TD3"]):
            stats["td"] = int(row["TD3"])
        total, _ = calculate_fantasy_points(stats)
        return total

    df["fpts"] = df.apply(_calc_row, axis=1)
    logger.info(f"Computed fantasy points: mean={df['fpts'].mean():.1f}, "
                f"median={df['fpts'].median():.1f}")
    return df


# ── Contextual Features ─────────────────────────────────────────

def compute_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest, schedule, and location features.

    Requires game_date_parsed column and data sorted by [PLAYER_ID, game_date_parsed].
    """
    df = df.copy()

    # Home/away (may already exist from nba_api_client)
    if "is_home" not in df.columns:
        df["is_home"] = df["MATCHUP"].str.contains("vs.", na=False)

    if "opponent" not in df.columns:
        df["opponent"] = df["MATCHUP"].str.split(" ").str[-1]

    # Days rest and back-to-back (per player)
    df["prev_game_date"] = df.groupby("PLAYER_ID")["game_date_parsed"].shift(1)
    df["days_rest"] = (df["game_date_parsed"] - df["prev_game_date"]).dt.days
    df["is_back_to_back"] = df["days_rest"] == 1
    df.drop(columns=["prev_game_date"], inplace=True)

    # Calendar features
    df["day_of_week"] = df["game_date_parsed"].dt.dayofweek
    df["month"] = df["game_date_parsed"].dt.month

    return df


# ── Rolling Averages ─────────────────────────────────────────────

def compute_rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling averages and standard deviations per player.

    Uses shift(1) to prevent data leakage — rolling window only includes
    prior games, not the current game. min_periods=1 so early-season rows
    still get values.
    """
    df = df.copy()

    # Games played count per player within current season
    df["games_played_season"] = df.groupby(
        ["PLAYER_ID", "SEASON_YEAR"]
    ).cumcount() + 1

    for window in ROLLING_WINDOWS:
        suffix = f"_roll_{window}"

        for col in ROLLING_STAT_COLS:
            if col not in df.columns:
                continue

            # Shift by 1 to exclude current game, then rolling mean
            shifted = df.groupby("PLAYER_ID")[col].shift(1)
            df[f"{col.lower()}{suffix}"] = shifted.groupby(
                df["PLAYER_ID"]
            ).transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )

        # Rolling std of fpts (consistency/volatility metric)
        shifted_fpts = df.groupby("PLAYER_ID")["fpts"].shift(1)
        df[f"fpts_std_{window}"] = shifted_fpts.groupby(
            df["PLAYER_ID"]
        ).transform(
            lambda x: x.rolling(window, min_periods=2).std()
        )

    return df


# ── Trend Features ───────────────────────────────────────────────

def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend and momentum features from rolling averages.

    Requires rolling averages to already be computed.
    """
    df = df.copy()

    # Short vs medium trend (positive = heating up)
    if "fpts_roll_3" in df.columns and "fpts_roll_10" in df.columns:
        df["fpts_trend_3v10"] = df["fpts_roll_3"] - df["fpts_roll_10"]

    # Medium vs long trend
    if "fpts_roll_5" in df.columns and "fpts_roll_15" in df.columns:
        df["fpts_trend_5v15"] = df["fpts_roll_5"] - df["fpts_roll_15"]

    # Over/under performance vs recent average
    if "fpts_roll_5" in df.columns:
        df["fpts_last_game_diff"] = df["fpts"] - df["fpts_roll_5"]

    # Hot streak indicator: L3 > L10
    if "fpts_roll_3" in df.columns and "fpts_roll_10" in df.columns:
        df["is_hot_streak"] = df["fpts_roll_3"] > df["fpts_roll_10"]

    return df


# ── Season Context ───────────────────────────────────────────────

def compute_season_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add season phase and career game count features."""
    df = df.copy()

    # Season phase based on games played
    def _season_phase(gp):
        if gp <= 20:
            return "early"
        elif gp <= 55:
            return "mid"
        elif gp <= 72:
            return "late"
        else:
            return "final_stretch"

    if "games_played_season" in df.columns:
        df["season_phase"] = df["games_played_season"].apply(_season_phase)

    # Career game count across all seasons in dataset
    df["career_game_count"] = df.groupby("PLAYER_ID").cumcount() + 1

    return df


# ── Advanced Stats Join ──────────────────────────────────────────

# Columns to keep from advanced stats, with adv_ prefix
ADV_STAT_COLUMNS = [
    "USG_PCT", "TS_PCT", "OFF_RATING", "DEF_RATING", "NET_RATING",
    "PACE", "PIE", "AST_PCT",
]


def join_advanced_stats(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Join season-level advanced stats to game log rows.

    Returns df unchanged if advanced stats are unavailable.
    """
    adv_df = get_player_advanced_stats(season)

    if adv_df is None:
        logger.warning(f"Advanced stats unavailable for {season}, skipping join")
        return df

    # Select only columns we need and rename with adv_ prefix
    keep_cols = ["PLAYER_ID"]
    rename_map = {}
    for col in ADV_STAT_COLUMNS:
        if col in adv_df.columns:
            keep_cols.append(col)
            rename_map[col] = f"adv_{col.lower()}"

    adv_subset = adv_df[keep_cols].copy().rename(columns=rename_map)

    result = df.merge(adv_subset, on="PLAYER_ID", how="left")

    missing = result[f"adv_{ADV_STAT_COLUMNS[0].lower()}"].isna().sum()
    if missing > 0:
        logger.warning(f"{missing} rows missing advanced stats")

    return result


# ── Orchestrator ─────────────────────────────────────────────────

def engineer_all_features(df: pd.DataFrame, season: str) -> pd.DataFrame:
    """Apply all feature transforms in dependency order.

    Args:
        df: Raw game log DataFrame from nba_api_client.get_player_game_logs()
        season: Season string (e.g., '2023-24')

    Returns:
        Fully-featured DataFrame ready for ML or EDA.
    """
    logger.info(f"Engineering features for {season} ({len(df)} rows)...")

    # 1. Fantasy points (needs raw stats)
    df = compute_fantasy_points(df)

    # 2. Contextual features (needs MATCHUP, game_date_parsed)
    df = compute_contextual_features(df)

    # 3. Rolling averages (needs fpts from step 1, sorted order)
    df = compute_rolling_averages(df)

    # 4. Trend features (needs rolling averages from step 3)
    df = compute_trend_features(df)

    # 5. Season context (needs games_played_season from step 3)
    df = compute_season_context(df)

    # 6. Advanced stats join (independent, needs PLAYER_ID)
    df = join_advanced_stats(df, season)

    # 7. Opponent defense join (needs opponent column from step 2)
    df = join_opponent_defense(df, season)

    logger.info(f"Feature engineering complete: {df.shape[1]} columns")
    return df
