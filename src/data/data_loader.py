"""
Data orchestrator: load, join, and produce the final ML-ready dataset.

Handles multi-season loading, combining, and parquet persistence.
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path

import pandas as pd

from src.data.nba_api_client import get_player_game_logs, DATA_DIR, HISTORICAL_SEASONS, CURRENT_SEASON, _parquet_engine
from src.data.feature_engineering import engineer_all_features

logger = logging.getLogger(__name__)

COMBINED_FILENAME = "features_combined.parquet"


def load_season(season: str, force_refresh: bool = False) -> pd.DataFrame:
    """Fetch game logs for a season and apply all feature engineering.

    Args:
        season: NBA season string (e.g., '2023-24')
        force_refresh: If True, re-download from API even if parquet exists

    Returns:
        Fully-featured DataFrame for one season.
    """
    df = get_player_game_logs(season, force_refresh=force_refresh)
    df = engineer_all_features(df, season)
    return df


def load_all_seasons(
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load and combine multiple seasons of featured data.

    Args:
        seasons: List of season strings. Defaults to HISTORICAL_SEASONS.
        force_refresh: If True, re-download all data from API.

    Returns:
        Combined DataFrame across all seasons.
    """
    if seasons is None:
        seasons = HISTORICAL_SEASONS

    dfs = []
    for season in seasons:
        logger.info(f"Loading season {season}...")
        df = load_season(season, force_refresh=force_refresh)
        dfs.append(df)

    # Normalize datetime precision before concat (avoids ns/ms mismatch)
    for df in dfs:
        if "game_date_parsed" in df.columns:
            df["game_date_parsed"] = pd.to_datetime(df["game_date_parsed"]).dt.floor("s")

    combined = pd.concat(dfs, ignore_index=True)

    # Recompute career game count across all seasons
    combined = combined.sort_values(
        ["PLAYER_ID", "game_date_parsed"]
    ).reset_index(drop=True)
    combined["career_game_count"] = combined.groupby("PLAYER_ID").cumcount() + 1

    logger.info(
        f"Combined dataset: {len(combined)} rows, {combined.shape[1]} columns, "
        f"seasons: {combined['SEASON_YEAR'].nunique()}, "
        f"players: {combined['PLAYER_ID'].nunique()}"
    )
    return combined


def build_and_save_dataset(
    seasons: list[str] | None = None,
    force_refresh: bool = False,
) -> Path:
    """Build the full featured dataset and save to parquet.

    Returns:
        Path to the saved parquet file.
    """
    df = load_all_seasons(seasons=seasons, force_refresh=force_refresh)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / COMBINED_FILENAME
    df.to_parquet(path, index=False, engine=_parquet_engine())

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved combined dataset to {path} ({size_mb:.1f} MB)")

    print(f"\nDataset summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {df.shape[1]}")
    print(f"  Seasons: {sorted(df['SEASON_YEAR'].unique())}")
    print(f"  Players: {df['PLAYER_ID'].nunique():,}")
    print(f"  Date range: {df['game_date_parsed'].min()} to {df['game_date_parsed'].max()}")
    print(f"  File size: {size_mb:.1f} MB")

    return path


def load_dataset() -> pd.DataFrame | None:
    """Load the pre-built combined dataset from parquet.

    Returns:
        DataFrame if file exists, None otherwise.
    """
    path = DATA_DIR / COMBINED_FILENAME
    if path.exists():
        df = pd.read_parquet(path, engine=_parquet_engine())
        logger.info(f"Loaded combined dataset: {len(df)} rows from {path}")
        return df
    logger.warning(f"Combined dataset not found at {path}")
    return None


def is_dataset_stale(max_age_hours: float = 6.0) -> bool:
    """Check if the combined dataset parquet is older than max_age_hours.

    Returns True if stale or missing.
    """
    path = DATA_DIR / COMBINED_FILENAME
    if not path.exists():
        return True
    mtime = path.stat().st_mtime
    age_hours = (time.time() - mtime) / 3600
    return age_hours > max_age_hours
