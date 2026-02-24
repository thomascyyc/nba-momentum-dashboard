"""Unit tests for ML model module."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.model import (
    EXCLUDE_COLS,
    get_feature_columns,
    prepare_features,
    temporal_split,
    train_model,
    evaluate_model,
    predict_fpts,
    save_model,
    load_model,
    _compute_residual_std,
)


def _make_dataset(n_players=5, games_per_player=30, seasons=None):
    """Create a synthetic dataset matching the feature_engineering output schema."""
    if seasons is None:
        seasons = ["2021-22", "2022-23", "2023-24"]

    rng = np.random.RandomState(42)
    rows = []
    for season in seasons:
        for pid in range(n_players):
            base_fpts = rng.uniform(5, 25)
            for g in range(games_per_player):
                fpts = base_fpts + rng.normal(0, 3)
                row = {
                    "PLAYER_NAME": f"Player_{pid}",
                    "PLAYER_ID": f"player{pid:03d}",
                    "TEAM_ABBREVIATION": "LAL",
                    "SEASON_YEAR": season,
                    "GAME_DATE": f"2023-01-{(g % 28) + 1:02d}",
                    "MATCHUP": "LAL vs. GSW" if g % 2 == 0 else "LAL @ BOS",
                    "WL": "W",
                    "MIN": rng.uniform(10, 35),
                    "PTS": rng.randint(0, 40),
                    "REB": rng.randint(0, 15),
                    "AST": rng.randint(0, 12),
                    "STL": rng.randint(0, 5),
                    "BLK": rng.randint(0, 4),
                    "TOV": rng.randint(0, 6),
                    "FG3M": rng.randint(0, 8),
                    "FGM": rng.randint(0, 15),
                    "FGA": rng.randint(5, 25),
                    "FG3A": rng.randint(0, 12),
                    "FTM": rng.randint(0, 10),
                    "FTA": rng.randint(0, 12),
                    "OREB": rng.randint(0, 5),
                    "DREB": rng.randint(0, 10),
                    "PF": rng.randint(0, 6),
                    "PLUS_MINUS": rng.uniform(-20, 20),
                    "DD2": 0,
                    "TD3": 0,
                    "NBA_FANTASY_PTS": np.nan,
                    "game_date_parsed": pd.Timestamp(f"2023-01-{(g % 28) + 1:02d}"),
                    "opponent": "GSW",
                    "is_home": g % 2 == 0,
                    "FG_PCT": rng.uniform(0.3, 0.6),
                    "FG3_PCT": rng.uniform(0.2, 0.5),
                    "FT_PCT": rng.uniform(0.5, 1.0),
                    "fpts": fpts,
                    "days_rest": rng.choice([1, 2, 3]),
                    "is_back_to_back": rng.choice([True, False]),
                    "day_of_week": rng.randint(0, 7),
                    "month": rng.randint(10, 13) % 12 + 1,
                    "games_played_season": g + 1,
                    "season_phase": rng.choice(["early", "mid", "late"]),
                    "career_game_count": g + 1,
                    "is_hot_streak": rng.choice([True, False]),
                }
                # Rolling features (NaN for first game)
                for window in [3, 5, 10, 15]:
                    val = base_fpts + rng.normal(0, 1) if g > 0 else np.nan
                    row[f"fpts_roll_{window}"] = val
                    row[f"fpts_std_{window}"] = abs(rng.normal(2, 1)) if g > 1 else np.nan
                    for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "min"]:
                        row[f"{stat}_roll_{window}"] = rng.uniform(0, 10) if g > 0 else np.nan

                row["fpts_trend_3v10"] = rng.normal(0, 2) if g > 0 else np.nan
                row["fpts_trend_5v15"] = rng.normal(0, 1) if g > 0 else np.nan
                row["fpts_last_game_diff"] = rng.normal(0, 3)

                rows.append(row)

    return pd.DataFrame(rows)


class TestGetFeatureColumns:
    def test_excludes_target(self):
        df = _make_dataset(n_players=1, games_per_player=5)
        cols = get_feature_columns(df)
        assert "fpts" not in cols

    def test_excludes_identifiers(self):
        df = _make_dataset(n_players=1, games_per_player=5)
        cols = get_feature_columns(df)
        for c in ["PLAYER_NAME", "PLAYER_ID", "SEASON_YEAR", "GAME_DATE", "MATCHUP"]:
            assert c not in cols

    def test_excludes_leaky_stats(self):
        df = _make_dataset(n_players=1, games_per_player=5)
        cols = get_feature_columns(df)
        for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV", "MIN"]:
            assert c not in cols

    def test_includes_rolling_features(self):
        df = _make_dataset(n_players=1, games_per_player=5)
        cols = get_feature_columns(df)
        assert "fpts_roll_10" in cols
        assert "fpts_roll_3" in cols
        assert "fpts_std_5" in cols

    def test_includes_contextual_features(self):
        df = _make_dataset(n_players=1, games_per_player=5)
        cols = get_feature_columns(df)
        assert "days_rest" in cols
        assert "is_back_to_back" in cols
        assert "day_of_week" in cols
        assert "games_played_season" in cols


class TestPrepareFeatures:
    def test_drops_first_game_nans(self):
        df = _make_dataset(n_players=2, games_per_player=10)
        X, y = prepare_features(df)
        # First game per player per season has NaN rolling features and should be dropped
        assert not X["fpts_roll_3"].isna().any()

    def test_encodes_season_phase(self):
        df = _make_dataset(n_players=1, games_per_player=10)
        X, y = prepare_features(df)
        if "season_phase" in X.columns:
            assert X["season_phase"].dtype in [np.int64, np.int32, int]

    def test_converts_booleans(self):
        df = _make_dataset(n_players=1, games_per_player=10)
        X, y = prepare_features(df)
        for col in X.columns:
            assert X[col].dtype != bool


class TestTemporalSplit:
    def test_no_data_leakage(self):
        df = _make_dataset(n_players=3, games_per_player=20)
        X_train, X_test, y_train, y_test = temporal_split(
            df, ["2021-22", "2022-23"], ["2023-24"]
        )
        # Train indices should not overlap with test
        assert len(set(X_train.index) & set(X_test.index)) == 0

    def test_correct_sizes(self):
        df = _make_dataset(n_players=3, games_per_player=20)
        X_train, X_test, y_train, y_test = temporal_split(
            df, ["2021-22", "2022-23"], ["2023-24"]
        )
        # 2 seasons of train, 1 of test
        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestTrainModel:
    def test_train_returns_model_and_scores(self):
        df = _make_dataset(n_players=3, games_per_player=30)
        X, y = prepare_features(df)
        model, cv_scores, fi = train_model(X, y, n_cv_splits=3)

        assert model is not None
        assert len(cv_scores["mae"]) == 3
        assert len(fi) == len(X.columns)
        assert all(cv_scores["mae"][i] > 0 for i in range(3))

    def test_predict_returns_dict(self):
        df = _make_dataset(n_players=3, games_per_player=30)
        X, y = prepare_features(df)
        model, _, _ = train_model(X, y, n_cv_splits=2)

        result = predict_fpts(model, X.iloc[0].to_dict(), list(X.columns))
        assert "predicted_fpts" in result
        assert "confidence_low" in result
        assert "confidence_high" in result
        assert result["confidence_low"] <= result["predicted_fpts"]
        assert result["predicted_fpts"] <= result["confidence_high"]


class TestEvaluateModel:
    def test_returns_metrics(self):
        df = _make_dataset(n_players=3, games_per_player=30)
        X_train, X_test, y_train, y_test = temporal_split(
            df, ["2021-22", "2022-23"], ["2023-24"]
        )
        model, _, _ = train_model(X_train, y_train, n_cv_splits=2)
        metrics = evaluate_model(model, X_test, y_test)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert metrics["mae"] > 0
        assert metrics["rmse"] >= metrics["mae"]


class TestSaveLoadModel:
    def test_roundtrip(self, tmp_path, monkeypatch):
        import src.ml.model as model_module
        monkeypatch.setattr(model_module, "MODEL_DIR", tmp_path)

        df = _make_dataset(n_players=3, games_per_player=20)
        X, y = prepare_features(df)
        model, cv_scores, fi = train_model(X, y, n_cv_splits=2)
        residual_std = _compute_residual_std(model, X, y)
        metrics = {"mae": 4.0, "rmse": 5.0, "r2": 0.4}

        save_model(model, list(X.columns), metrics, fi, cv_scores, residual_std)

        loaded = load_model()
        assert loaded is not None
        loaded_model, metadata = loaded

        assert metadata["n_features"] == len(X.columns)
        assert metadata["metrics"]["mae"] == 4.0

        # Predictions should match
        orig_pred = model.predict(X[:1])
        loaded_pred = loaded_model.predict(X[:1])
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
