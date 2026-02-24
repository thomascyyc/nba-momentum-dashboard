#!/usr/bin/env python3
"""
Train the FPTS prediction model.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --force   # retrain even if model exists
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data_loader import load_dataset
from src.ml.model import (
    temporal_split,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    _compute_residual_std,
    MODEL_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train FPTS prediction model")
    parser.add_argument("--force", action="store_true", help="Retrain even if model exists")
    args = parser.parse_args()

    # Check if model already exists
    if not args.force and (MODEL_DIR / "model.pkl").exists():
        print(f"Model already exists at {MODEL_DIR}. Use --force to retrain.")
        return

    # Load dataset
    df = load_dataset()
    if df is None:
        print("No dataset found. Run scripts/fetch_historical_data.py first.")
        sys.exit(1)

    print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Seasons: {sorted(df['SEASON_YEAR'].unique())}")
    print()

    # Determine train/test split based on available seasons
    seasons = sorted(df["SEASON_YEAR"].unique())
    if len(seasons) >= 3:
        # Train on all but last season, test on last
        train_seasons = seasons[:-1]
        test_seasons = [seasons[-1]]
    else:
        # Only 1-2 seasons: use 80/20 split within
        train_seasons = seasons
        test_seasons = seasons

    print(f"Train seasons: {train_seasons}")
    print(f"Test season:   {test_seasons}")
    print()

    # Split
    X_train, X_test, y_train, y_test = temporal_split(
        df, train_seasons, test_seasons
    )
    print(f"Features: {X_train.shape[1]}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows:  {len(X_test):,}")
    print()

    # Train
    print("=" * 50)
    print("Training model...")
    print("=" * 50)
    model, cv_scores, feature_importance = train_model(X_train, y_train)

    # Evaluate
    print()
    print("=" * 50)
    print("Evaluating on test set...")
    print("=" * 50)
    metrics = evaluate_model(model, X_test, y_test)

    print(f"  MAE:  {metrics['mae']:.3f} FPTS")
    print(f"  RMSE: {metrics['rmse']:.3f} FPTS")
    print(f"  R2:   {metrics['r2']:.3f}")
    print(f"  Mean actual:    {metrics['mean_actual']:.1f}")
    print(f"  Mean predicted: {metrics['mean_predicted']:.1f}")
    print()

    for tier in ["bench", "rotation", "starter", "star"]:
        key = f"mae_{tier}"
        if key in metrics:
            print(f"  MAE ({tier}): {metrics[key]:.3f}")

    # Compute residual std for confidence intervals
    residual_std = _compute_residual_std(model, X_train, y_train)

    # Save
    print()
    print("=" * 50)
    print("Top 15 features:")
    print("=" * 50)
    print(feature_importance.head(15).to_string(index=False))

    save_model(
        model,
        feature_cols=list(X_train.columns),
        metrics=metrics,
        feature_importance=feature_importance,
        cv_scores=cv_scores,
        residual_std=residual_std,
    )

    print()
    print(f"Model saved to {MODEL_DIR}")
    print("Done!")


if __name__ == "__main__":
    main()
