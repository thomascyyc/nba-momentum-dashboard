"""
FPTS prediction model: train, evaluate, predict.

Uses LightGBM (preferred) or XGBoost for gradient-boosted tree regression.
Temporal train/test split by season. Model and metadata saved to data/model/.
"""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "model"

# ── Feature Selection ────────────────────────────────────────────

# Columns to EXCLUDE from features (target, identifiers, current-game stats)
EXCLUDE_COLS = {
    # Target
    "fpts",
    # Derived from current game (leakage)
    "fpts_last_game_diff",
    # Identifiers / metadata
    "PLAYER_NAME", "PLAYER_ID", "TEAM_ABBREVIATION", "SEASON_YEAR",
    "GAME_DATE", "MATCHUP", "WL", "game_date_parsed", "opponent",
    # Current-game box score stats (leakage — not available pre-game)
    "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M",
    "FGM", "FGA", "FG3A", "FTM", "FTA", "OREB", "DREB",
    "PF", "PLUS_MINUS", "DD2", "TD3", "MIN",
    "FG_PCT", "FG3_PCT", "FT_PCT",
    # Always null in our dataset
    "NBA_FANTASY_PTS",
}

# Categorical features that need encoding
CATEGORICAL_COLS = {"season_phase"}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return ordered list of feature columns, excluding target/IDs/leaky cols."""
    return sorted(c for c in df.columns if c not in EXCLUDE_COLS)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target y.

    - Drops rows where fpts is NaN
    - Drops rows where rolling features are all NaN (first game per player)
    - Encodes season_phase as integer
    - Converts booleans to int

    Returns (X, y) with aligned indices.
    """
    df = df.copy()

    # Drop rows without target
    df = df.dropna(subset=["fpts"])

    # Drop first game per player (no rolling features available)
    df = df.dropna(subset=["fpts_roll_3"])

    # Encode season_phase as ordinal
    phase_map = {"early": 0, "mid": 1, "late": 2, "final_stretch": 3}
    if "season_phase" in df.columns:
        df["season_phase"] = df["season_phase"].map(phase_map).fillna(0).astype(int)

    # Convert booleans to int for the model
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].copy()
    y = df["fpts"].copy()

    return X, y


def temporal_split(
    df: pd.DataFrame,
    train_seasons: list[str],
    test_seasons: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data by season for temporal train/test.

    Ensures no future data leaks into training set.
    """
    train_mask = df["SEASON_YEAR"].isin(train_seasons)
    test_mask = df["SEASON_YEAR"].isin(test_seasons)

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    logger.info(f"Train: {len(X_train)} rows ({train_seasons})")
    logger.info(f"Test:  {len(X_test)} rows ({test_seasons})")

    return X_train, X_test, y_train, y_test


def _get_model_class():
    """Get LightGBM regressor, falling back to XGBoost, then sklearn."""
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor, "lightgbm"
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        return XGBRegressor, "xgboost"
    except ImportError:
        pass
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor, "sklearn_gbr"


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_cv_splits: int = 5,
) -> tuple:
    """Train gradient-boosted model with TimeSeriesSplit cross-validation.

    Returns (model, cv_results, feature_importance_df).
    """
    ModelClass, model_name = _get_model_class()
    logger.info(f"Training with {model_name} on {X_train.shape[1]} features...")

    # Hyperparameters (reasonable defaults for tabular regression)
    if model_name == "lightgbm":
        model = ModelClass(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=-1,
        )
    elif model_name == "xgboost":
        model = ModelClass(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbosity=0,
        )
    else:
        model = ModelClass(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=42,
        )

    # Cross-validation
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_scores = {"mae": [], "rmse": []}

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_cv_train = X_train.iloc[train_idx]
        y_cv_train = y_train.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_val = y_train.iloc[val_idx]

        model.fit(X_cv_train, y_cv_train)
        preds = model.predict(X_cv_val)

        mae = mean_absolute_error(y_cv_val, preds)
        rmse = np.sqrt(mean_squared_error(y_cv_val, preds))
        cv_scores["mae"].append(mae)
        cv_scores["rmse"].append(rmse)
        logger.info(f"  Fold {fold + 1}: MAE={mae:.3f}, RMSE={rmse:.3f}")

    logger.info(f"CV Mean MAE: {np.mean(cv_scores['mae']):.3f} "
                f"(+/- {np.std(cv_scores['mae']):.3f})")

    # Final fit on all training data
    model.fit(X_train, y_train)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
    else:
        fi = pd.DataFrame({"feature": X_train.columns, "importance": 0})

    return model, cv_scores, fi


def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series,
) -> dict:
    """Evaluate on held-out test set. Returns metrics dict."""
    preds = model.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "mean_actual": float(y_test.mean()),
        "mean_predicted": float(preds.mean()),
        "n_test": len(y_test),
    }

    # Per-tier accuracy (bucket by actual FPTS)
    tier_bins = [(-np.inf, 5), (5, 15), (15, 25), (25, np.inf)]
    tier_names = ["bench", "rotation", "starter", "star"]
    for (lo, hi), name in zip(tier_bins, tier_names):
        mask = (y_test >= lo) & (y_test < hi)
        if mask.sum() > 0:
            metrics[f"mae_{name}"] = float(mean_absolute_error(
                y_test[mask], preds[mask]
            ))

    return metrics


def _compute_residual_std(model, X_train, y_train):
    """Compute residual std by player tier for confidence intervals."""
    preds = model.predict(X_train)
    residuals = y_train - preds

    # Bin by predicted FPTS
    bins = pd.cut(preds, bins=[-np.inf, 5, 15, 25, np.inf],
                  labels=["bench", "rotation", "starter", "star"])
    tier_std = {}
    for tier in ["bench", "rotation", "starter", "star"]:
        mask = bins == tier
        if mask.sum() > 10:
            tier_std[tier] = float(np.std(residuals[mask]))
        else:
            tier_std[tier] = float(np.std(residuals))

    return tier_std


def save_model(
    model,
    feature_cols: list[str],
    metrics: dict,
    feature_importance: pd.DataFrame,
    cv_scores: dict,
    residual_std: dict,
) -> Path:
    """Save model + metadata to data/model/ directory."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    model_path = MODEL_DIR / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Metadata
    metadata = {
        "feature_columns": feature_cols,
        "metrics": metrics,
        "cv_mae_mean": float(np.mean(cv_scores["mae"])),
        "cv_mae_std": float(np.std(cv_scores["mae"])),
        "residual_std_by_tier": residual_std,
        "training_date": datetime.now().isoformat(),
        "n_features": len(feature_cols),
        "model_type": type(model).__name__,
    }
    with open(MODEL_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Feature importance
    feature_importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    logger.info(f"Model saved to {MODEL_DIR}")
    return MODEL_DIR


def load_model() -> tuple | None:
    """Load saved model and metadata. Returns (model, metadata) or None."""
    model_path = MODEL_DIR / "model.pkl"
    meta_path = MODEL_DIR / "metadata.json"

    if not model_path.exists() or not meta_path.exists():
        logger.warning(f"No saved model found at {MODEL_DIR}")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(meta_path) as f:
        metadata = json.load(f)

    return model, metadata


def predict_fpts(
    model,
    features: dict | pd.DataFrame,
    feature_cols: list[str],
    residual_std: dict | None = None,
) -> dict:
    """Predict next-game FPTS for a player.

    Args:
        model: trained model
        features: dict or single-row DataFrame of feature values
        feature_cols: ordered list of feature names the model expects
        residual_std: optional dict of tier -> std for confidence intervals

    Returns dict with predicted_fpts, confidence_low, confidence_high.
    """
    if isinstance(features, dict):
        row = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_cols}])
    else:
        row = features[feature_cols].copy()

    pred = float(model.predict(row)[0])

    # Confidence interval based on residual std by tier
    if residual_std:
        if pred >= 25:
            tier = "star"
        elif pred >= 15:
            tier = "starter"
        elif pred >= 5:
            tier = "rotation"
        else:
            tier = "bench"
        std = residual_std.get(tier, 5.0)
    else:
        std = 5.0  # fallback

    return {
        "predicted_fpts": round(pred, 1),
        "confidence_low": round(max(0, pred - std), 1),
        "confidence_high": round(pred + std, 1),
    }
