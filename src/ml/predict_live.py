"""
Generate ML predictions from live game log cache data.

Bridges the gap between the dashboard's JSON cache format and the
ML model's expected feature format. Computes rolling averages from
cached game lists to match the training feature schema.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np

from src.ml.model import load_model, predict_fpts

logger = logging.getLogger(__name__)

_model_cache = None


def _get_model():
    """Load model with simple caching (avoid reloading on every call)."""
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model()
    return _model_cache


def cache_games_to_features(games: list[dict], player_context: dict | None = None) -> dict:
    """Convert a player's cached game list into features matching the model.

    Args:
        games: list of game dicts from cache (newest first), each with keys:
               fpts, pts, reb, ast, stl, blk, to, tpm, min, date, matchup
        player_context: optional dict with is_home, days_rest, is_back_to_back
                        for the upcoming game

    Returns:
        dict of feature_name -> value, matching training feature columns.
    """
    if not games or len(games) < 2:
        return {}

    context = player_context or {}
    features = {}

    # Rolling averages for each window
    for window in [3, 5, 10, 15]:
        # FPTS rolling
        fpts_vals = [g.get("fpts", 0) for g in games[:window]]
        if fpts_vals:
            features[f"fpts_roll_{window}"] = np.mean(fpts_vals)
            if len(fpts_vals) >= 2:
                features[f"fpts_std_{window}"] = float(np.std(fpts_vals, ddof=1))
            else:
                features[f"fpts_std_{window}"] = 0.0

        # Component stat rolling averages
        stat_map = {
            "pts": "pts", "reb": "reb", "ast": "ast",
            "stl": "stl", "blk": "blk", "to": "tov", "min": "min",
        }
        for cache_key, feature_prefix in stat_map.items():
            vals = [g.get(cache_key, 0) for g in games[:window]]
            if vals:
                features[f"{feature_prefix}_roll_{window}"] = np.mean(vals)

    # Trend features
    r3 = features.get("fpts_roll_3")
    r5 = features.get("fpts_roll_5")
    r10 = features.get("fpts_roll_10")
    r15 = features.get("fpts_roll_15")

    if r3 is not None and r10 is not None:
        features["fpts_trend_3v10"] = r3 - r10
    if r5 is not None and r15 is not None:
        features["fpts_trend_5v15"] = r5 - r15

    # Hot streak
    if r3 is not None and r10 is not None:
        features["is_hot_streak"] = int(r3 > r10)

    # Contextual features
    features["is_home"] = int(context.get("is_home", False))
    features["is_back_to_back"] = int(context.get("is_back_to_back", False))
    features["days_rest"] = context.get("days_rest", 2)
    features["games_played_season"] = len(games)
    features["career_game_count"] = len(games)  # approximate

    # Calendar
    today = date.today()
    features["day_of_week"] = today.weekday()
    features["month"] = today.month

    # Season phase
    gp = len(games)
    if gp <= 20:
        features["season_phase"] = 0  # early
    elif gp <= 55:
        features["season_phase"] = 1  # mid
    elif gp <= 72:
        features["season_phase"] = 2  # late
    else:
        features["season_phase"] = 3  # final_stretch

    return features


def predict_for_player(games: list[dict], player_context: dict | None = None) -> dict | None:
    """Generate prediction for a single player.

    Args:
        games: game list from cache (newest first)
        player_context: optional {is_home, days_rest, is_back_to_back}

    Returns:
        {predicted_fpts, confidence_low, confidence_high} or None if unavailable.
    """
    model_data = _get_model()
    if model_data is None:
        return None

    model, metadata = model_data
    feature_cols = metadata["feature_columns"]
    residual_std = metadata.get("residual_std_by_tier")

    features = cache_games_to_features(games, player_context)
    if not features:
        return None

    return predict_fpts(model, features, feature_cols, residual_std)


def predict_for_roster(cache: dict, roster_id, players: list[dict]) -> dict:
    """Generate predictions for all players on a roster.

    Args:
        cache: the loaded v2 game log cache
        roster_id: team's roster ID (str or int)
        players: list of {player_id, name, team, position}

    Returns:
        dict of player_id -> {predicted_fpts, confidence_low, confidence_high}
    """
    from cache import get_player_games

    predictions = {}
    for player in players:
        pid = player["player_id"]
        games = get_player_games(cache, pid, roster_id=str(roster_id))
        if not games:
            continue

        result = predict_for_player(games)
        if result is not None:
            predictions[pid] = result

    return predictions


def predict_for_available(cache: dict, players: list[dict]) -> dict:
    """Generate predictions for waiver wire players.

    Args:
        cache: the loaded v2 game log cache
        players: list of {player_id, name, team, position}

    Returns:
        dict of player_id -> {predicted_fpts, confidence_low, confidence_high}
    """
    from cache import get_player_games

    predictions = {}
    for player in players:
        pid = player["player_id"]
        games = get_player_games(cache, pid, source="available")
        if not games:
            continue

        result = predict_for_player(games)
        if result is not None:
            predictions[pid] = result

    return predictions
