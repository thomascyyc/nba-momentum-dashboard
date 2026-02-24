"""Unit tests for BBRef advanced stats module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.bbref_advanced import (
    compute_team_defense_from_game_logs,
)


def _make_game_logs():
    """Create synthetic game log data for team defense computation."""
    rows = []
    teams = ["LAL", "GSW", "BOS"]
    for game_num in range(10):
        date = pd.Timestamp(f"2024-01-{game_num + 1:02d}")
        # Each game: team A vs team B
        team_a = teams[game_num % 3]
        team_b = teams[(game_num + 1) % 3]
        for pid in range(5):  # 5 players per team
            # Team A player stats (scoring against team B)
            rows.append({
                "PLAYER_ID": f"player_a{pid}",
                "TEAM_ABBREVIATION": team_a,
                "game_date_parsed": date,
                "opponent": team_b,
                "PTS": 20, "REB": 5, "AST": 3,
                "STL": 1, "BLK": 1, "TOV": 2, "FG3M": 2,
                "FGM": 8, "FGA": 15, "FG3A": 5,
            })
            # Team B player stats (scoring against team A)
            rows.append({
                "PLAYER_ID": f"player_b{pid}",
                "TEAM_ABBREVIATION": team_b,
                "game_date_parsed": date,
                "opponent": team_a,
                "PTS": 18, "REB": 4, "AST": 2,
                "STL": 1, "BLK": 0, "TOV": 3, "FG3M": 1,
                "FGM": 7, "FGA": 16, "FG3A": 4,
            })
    return pd.DataFrame(rows)


class TestTeamDefense:
    def test_computes_all_teams(self):
        """Team defense should have an entry for each team that was an opponent."""
        logs = _make_game_logs()
        # Save temporarily for the function to load
        from src.data.nba_api_client import save_to_parquet, DATA_DIR
        save_to_parquet(logs, "player_game_logs_test-season.parquet")

        try:
            df = compute_team_defense_from_game_logs("test-season", force_refresh=True)
            assert df is not None
            assert len(df) == 3  # LAL, GSW, BOS
            assert "TEAM_ABBREVIATION" in df.columns
            assert "PTS" in df.columns
        finally:
            # Cleanup
            (DATA_DIR / "player_game_logs_test-season.parquet").unlink(missing_ok=True)
            (DATA_DIR / "team_defense_ratings_test-season.parquet").unlink(missing_ok=True)

    def test_defense_values_are_per_game_averages(self):
        """Defense stats should be averaged per game, not per player."""
        logs = _make_game_logs()
        from src.data.nba_api_client import save_to_parquet, DATA_DIR
        save_to_parquet(logs, "player_game_logs_test-season.parquet")

        try:
            df = compute_team_defense_from_game_logs("test-season", force_refresh=True)
            # Each team faces 5 players scoring 20 pts each = 100 pts per game
            # (or 18 pts each = 90 pts per game depending on matchup)
            # Values should be in the 90-100 range (team totals), not 18-20 (per player)
            assert df is not None
            for _, row in df.iterrows():
                assert row["PTS"] > 50  # Must be team-level, not per-player
        finally:
            (DATA_DIR / "player_game_logs_test-season.parquet").unlink(missing_ok=True)
            (DATA_DIR / "team_defense_ratings_test-season.parquet").unlink(missing_ok=True)

    def test_returns_none_without_game_logs(self):
        """Should return None if no game logs parquet exists."""
        df = compute_team_defense_from_game_logs("nonexistent-season", force_refresh=True)
        assert df is None
