#!/usr/bin/env python3
"""
Fetch historical NBA data for Phase 1A ML pipeline.

Downloads 3 seasons of player game logs, advanced stats, and team defense
stats from NBA.com via nba_api. Saves to data/ as parquet files, then
builds the combined feature dataset.

Usage:
    python scripts/fetch_historical_data.py              # skip if files exist
    python scripts/fetch_historical_data.py --force      # re-download all
    python scripts/fetch_historical_data.py --season 2023-24  # single season
"""

import argparse
import sys
import time
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import nba_api_client
from src.data.data_loader import build_and_save_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

HISTORICAL_SEASONS = ["2021-22", "2022-23", "2023-24"]


def main():
    parser = argparse.ArgumentParser(description="Fetch historical NBA data")
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if files exist"
    )
    parser.add_argument(
        "--season", type=str, help="Fetch a single season only (e.g., 2023-24)"
    )
    args = parser.parse_args()

    seasons = [args.season] if args.season else HISTORICAL_SEASONS

    print(f"NBA Historical Data Fetch")
    print(f"Seasons: {', '.join(seasons)}")
    print(f"Force refresh: {args.force}")
    print()

    for season in seasons:
        print(f"{'=' * 50}")
        print(f"Season: {season}")
        print(f"{'=' * 50}")

        # Step 1: Player game logs (per-player, ~500 API calls)
        print(f"  [1/3] Player game logs...", end=" ", flush=True)
        df_games = nba_api_client.get_player_game_logs(season, force_refresh=args.force)
        print(f"{len(df_games):,} rows")

        # Step 2: Advanced player stats (~500 rows) — may fail
        print(f"  [2/3] Advanced player stats...", end=" ", flush=True)
        df_adv = nba_api_client.get_player_advanced_stats(
            season, force_refresh=args.force
        )
        if df_adv is not None:
            print(f"{len(df_adv):,} players")
        else:
            print("unavailable (will skip in features)")

        # Step 3: Team opponent stats (30 rows) — may fail
        print(f"  [3/3] Team opponent stats...", end=" ", flush=True)
        df_opp = nba_api_client.get_team_opponent_stats(
            season, force_refresh=args.force
        )
        if df_opp is not None:
            print(f"{len(df_opp)} teams")
        else:
            print("unavailable (will skip in features)")

        # Extra buffer between seasons
        time.sleep(1)

    # Build combined feature dataset
    print(f"\n{'=' * 50}")
    print(f"Building combined feature dataset...")
    print(f"{'=' * 50}")
    path = build_and_save_dataset(seasons=seasons, force_refresh=args.force)
    print(f"\nSaved to: {path}")
    print("Done!")


if __name__ == "__main__":
    main()
