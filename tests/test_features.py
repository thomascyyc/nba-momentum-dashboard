"""
Unit tests for feature engineering.

Uses mock DataFrames — no API calls needed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.feature_engineering import (
    STAT_COLUMN_MAP,
    compute_fantasy_points,
    compute_contextual_features,
    compute_rolling_averages,
    compute_trend_features,
    compute_season_context,
)
from src.data.nba_api_client import _parse_matchup


# ── Fixtures ─────────────────────────────────────────────────────


def _make_game_row(
    player_id=1,
    date="2024-01-15",
    matchup="LAL vs. GSW",
    season="2023-24",
    pts=20,
    reb=5,
    ast=4,
    stl=1,
    blk=0,
    tov=2,
    fg3m=2,
    minutes=30.0,
    dd2=0,
    td3=0,
):
    """Create a single game row matching nba_api PlayerGameLogs schema."""
    return {
        "PLAYER_ID": player_id,
        "PLAYER_NAME": f"Player_{player_id}",
        "TEAM_ABBREVIATION": matchup.split(" ")[0],
        "SEASON_YEAR": season,
        "GAME_DATE": date,
        "MATCHUP": matchup,
        "WL": "W",
        "MIN": minutes,
        "PTS": pts,
        "REB": reb,
        "AST": ast,
        "STL": stl,
        "BLK": blk,
        "TOV": tov,
        "FG3M": fg3m,
        "FGM": 8,
        "FGA": 16,
        "FG_PCT": 0.500,
        "FG3A": 5,
        "FG3_PCT": 0.400,
        "FTM": 4,
        "FTA": 5,
        "FT_PCT": 0.800,
        "OREB": 1,
        "DREB": 4,
        "PF": 2,
        "PLUS_MINUS": 5,
        "DD2": dd2,
        "TD3": td3,
        "NBA_FANTASY_PTS": 35.0,
        "game_date_parsed": pd.Timestamp(date),
        "opponent": matchup.split(" ")[-1],
        "is_home": "vs." in matchup,
    }


def _make_player_games(player_id, n_games, base_fpts=25.0, trend=0.0, start_date="2024-01-01"):
    """Create n game rows for one player with optional trend."""
    rows = []
    for i in range(n_games):
        date = pd.Timestamp(start_date) + pd.Timedelta(days=i * 2)  # every 2 days
        pts = int(base_fpts * 0.8 + trend * i)  # rough pts from fpts
        rows.append(_make_game_row(
            player_id=player_id,
            date=date.strftime("%Y-%m-%d"),
            pts=pts,
            reb=5,
            ast=4,
            stl=1,
            blk=1,
            tov=2,
            fg3m=2,
        ))
    return rows


# ── Fantasy Points Tests ─────────────────────────────────────────


class TestComputeFantasyPoints:
    def test_basic_scoring(self):
        """Standard stat line: PTS*0.5 + REB*1 + AST*1 + STL*2 + BLK*2 + TO*-1 + 3PM*0.5"""
        df = pd.DataFrame([_make_game_row(pts=20, reb=5, ast=4, stl=1, blk=0, tov=2, fg3m=2)])
        result = compute_fantasy_points(df)
        # 20*0.5 + 5*1 + 4*1 + 1*2 + 0*2 + 2*-1 + 2*0.5 = 10+5+4+2+0-2+1 = 20.0
        assert result["fpts"].iloc[0] == pytest.approx(20.0)

    def test_double_double(self):
        """10+ in two categories triggers DD bonus (+1.0)."""
        df = pd.DataFrame([_make_game_row(pts=20, reb=12, ast=3, stl=0, blk=0, tov=1, fg3m=1, dd2=1)])
        result = compute_fantasy_points(df)
        # 20*0.5 + 12*1 + 3*1 + 0 + 0 + 1*-1 + 1*0.5 + 1*1(DD) = 10+12+3-1+0.5+1 = 25.5
        assert result["fpts"].iloc[0] == pytest.approx(25.5)

    def test_triple_double(self):
        """10+ in three categories triggers both DD (+1) and TD (+2)."""
        df = pd.DataFrame([_make_game_row(pts=20, reb=12, ast=11, stl=0, blk=0, tov=3, fg3m=1, dd2=1, td3=1)])
        result = compute_fantasy_points(df)
        # 20*0.5 + 12*1 + 11*1 + 0 + 0 + 3*-1 + 1*0.5 + 1(DD) + 2(TD) = 10+12+11-3+0.5+1+2 = 33.5
        assert result["fpts"].iloc[0] == pytest.approx(33.5)

    def test_40pt_bonus(self):
        """40+ points triggers +2 bonus."""
        df = pd.DataFrame([_make_game_row(pts=42, reb=5, ast=3, stl=1, blk=0, tov=2, fg3m=3)])
        result = compute_fantasy_points(df)
        # 42*0.5 + 5*1 + 3*1 + 1*2 + 0 + 2*-1 + 3*0.5 + 2(40+bonus) = 21+5+3+2-2+1.5+2 = 32.5
        assert result["fpts"].iloc[0] == pytest.approx(32.5)

    def test_50pt_bonus_stacks(self):
        """50+ triggers both 40+ and 50+ bonuses (+4 total)."""
        df = pd.DataFrame([_make_game_row(pts=52, reb=5, ast=3, stl=1, blk=0, tov=2, fg3m=3)])
        result = compute_fantasy_points(df)
        # 52*0.5 + 5+3+2-2+1.5 + 2(40+) + 2(50+) = 26+9.5+4 = 39.5
        assert result["fpts"].iloc[0] == pytest.approx(39.5)


# ── Rolling Average Tests ────────────────────────────────────────


class TestRollingAverages:
    def _build_df(self, n=10):
        """Build a simple n-game DataFrame for one player."""
        rows = _make_player_games(player_id=1, n_games=n, base_fpts=25.0)
        df = pd.DataFrame(rows)
        df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)
        return compute_fantasy_points(df)

    def test_shift_prevents_leakage(self):
        """Current game's fpts should NOT be in its own rolling average."""
        df = self._build_df(n=10)
        result = compute_rolling_averages(df)

        # For the last game (index 9), fpts_roll_3 should be mean of games 6,7,8
        # NOT including game 9 itself
        last_fpts = result["fpts"].iloc[9]
        roll_3 = result["fpts_roll_3"].iloc[9]
        prior_3_mean = result["fpts"].iloc[6:9].mean()
        assert roll_3 == pytest.approx(prior_3_mean, abs=0.01)

    def test_first_game_has_nan_rolling(self):
        """First game has no prior games, so rolling should be NaN."""
        df = self._build_df(n=5)
        result = compute_rolling_averages(df)
        assert pd.isna(result["fpts_roll_3"].iloc[0])

    def test_min_periods(self):
        """Second game should still get a rolling value (min_periods=1)."""
        df = self._build_df(n=5)
        result = compute_rolling_averages(df)
        # Game index 1: only game 0 is prior, so roll_3 = game 0's fpts
        assert not pd.isna(result["fpts_roll_3"].iloc[1])
        assert result["fpts_roll_3"].iloc[1] == pytest.approx(result["fpts"].iloc[0])

    def test_games_played_season(self):
        """Games played counter should increment per player per season."""
        df = self._build_df(n=5)
        result = compute_rolling_averages(df)
        assert list(result["games_played_season"]) == [1, 2, 3, 4, 5]


# ── Trend Feature Tests ──────────────────────────────────────────


class TestTrendFeatures:
    def test_ascending_trend_positive(self):
        """Ascending fpts should produce positive trend."""
        rows = _make_player_games(player_id=1, n_games=15, base_fpts=15.0, trend=1.0)
        df = pd.DataFrame(rows)
        df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)
        df = compute_fantasy_points(df)
        df = compute_rolling_averages(df)
        df = compute_trend_features(df)
        # Last few games: L3 avg should be > L10 avg (ascending)
        last_trend = df["fpts_trend_3v10"].iloc[-1]
        assert last_trend > 0

    def test_hot_streak_indicator(self):
        """Hot streak = L3 > L10."""
        rows = _make_player_games(player_id=1, n_games=15, base_fpts=15.0, trend=1.0)
        df = pd.DataFrame(rows)
        df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)
        df = compute_fantasy_points(df)
        df = compute_rolling_averages(df)
        df = compute_trend_features(df)
        assert df["is_hot_streak"].iloc[-1] is True or df["is_hot_streak"].iloc[-1] == True


# ── Contextual Feature Tests ─────────────────────────────────────


class TestContextualFeatures:
    def test_back_to_back_detection(self):
        """Games on consecutive days should be flagged as B2B."""
        rows = [
            _make_game_row(player_id=1, date="2024-01-15"),
            _make_game_row(player_id=1, date="2024-01-16"),  # B2B
            _make_game_row(player_id=1, date="2024-01-18"),  # 2 days rest
        ]
        df = pd.DataFrame(rows)
        df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)
        result = compute_contextual_features(df)
        assert result["is_back_to_back"].iloc[1] == True
        assert result["is_back_to_back"].iloc[2] == False

    def test_days_rest(self):
        """Days rest should be correctly computed."""
        rows = [
            _make_game_row(player_id=1, date="2024-01-15"),
            _make_game_row(player_id=1, date="2024-01-16"),
            _make_game_row(player_id=1, date="2024-01-19"),
        ]
        df = pd.DataFrame(rows)
        df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)
        result = compute_contextual_features(df)
        assert pd.isna(result["days_rest"].iloc[0])  # first game, no prior
        assert result["days_rest"].iloc[1] == 1
        assert result["days_rest"].iloc[2] == 3

    def test_home_away_parsing(self):
        """Home vs away should be correctly detected from MATCHUP."""
        rows = [
            _make_game_row(matchup="LAL vs. GSW"),
            _make_game_row(matchup="LAL @ BOS"),
        ]
        df = pd.DataFrame(rows)
        result = compute_contextual_features(df)
        assert result["is_home"].iloc[0] == True
        assert result["is_home"].iloc[1] == False
        assert result["opponent"].iloc[0] == "GSW"
        assert result["opponent"].iloc[1] == "BOS"


# ── Season Context Tests ─────────────────────────────────────────


class TestSeasonContext:
    def test_season_phase(self):
        """Season phase should categorize by games played."""
        rows = []
        for i in range(80):
            date = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            rows.append(_make_game_row(player_id=1, date=date.strftime("%Y-%m-%d")))
        df = pd.DataFrame(rows)
        df = df.sort_values(["PLAYER_ID", "game_date_parsed"]).reset_index(drop=True)
        df = compute_fantasy_points(df)
        df = compute_rolling_averages(df)
        df = compute_season_context(df)

        assert df["season_phase"].iloc[4] == "early"     # game 5
        assert df["season_phase"].iloc[39] == "mid"      # game 40
        assert df["season_phase"].iloc[59] == "late"     # game 60
        assert df["season_phase"].iloc[79] == "final_stretch"  # game 80


# ── Matchup Parsing Tests ────────────────────────────────────────


class TestMatchupParsing:
    def test_home_game(self):
        opp, is_home = _parse_matchup("LAL vs. GSW")
        assert opp == "GSW"
        assert is_home is True

    def test_away_game(self):
        opp, is_home = _parse_matchup("LAL @ BOS")
        assert opp == "BOS"
        assert is_home is False
