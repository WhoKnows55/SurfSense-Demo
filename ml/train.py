"""
XGBoost training script for the surf condition model (Section 3.3.5).

Workflow:
  1. Load train / val splits from ml/data/processed/.
  2. Apply synthetic labels (ml/labels.py).
  3. Extract features (app/ml/feature_extractor.py).
  4. Fit a SimpleImputer on training data.
  5. Grid-search XGBoost with TimeSeriesSplit (5-fold).
  6. Evaluate best model on the val set.
  7. Persist model + imputer + metadata to ml/models/.

Usage:
    python -m ml.train
    python -m ml.train --no-search   # skip grid search, use default params
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from app.ml.feature_extractor import FEATURE_NAMES, ForecastPointFeatureExtractor
from ml.labels import compute_synthetic_score
from ml.splits import split

MODELS_DIR = Path("ml/models")
PROCESSED   = Path("ml/data/processed")

PARAM_GRID = {
    "max_iter":         [100, 300, 500],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.01, 0.05, 0.1],
    "min_samples_leaf": [10, 20, 50],
}

DEFAULT_PARAMS = {
    "max_iter":          300,
    "max_depth":         5,
    "learning_rate":     0.05,
    "min_samples_leaf":  20,
    "l2_regularization": 0.1,
    "random_state":      42,
}


def _prepare(df: pd.DataFrame, extractor: ForecastPointFeatureExtractor) -> np.ndarray:
    """Extract features and add synthetic labels."""
    X = extractor.transform_batch(df, skill_level="intermediate")
    return X


def _add_labels(df: pd.DataFrame) -> pd.Series:
    return df.apply(compute_synthetic_score, axis=1)


def _dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()[:12]


def train(grid_search: bool = True) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading splits …")
    train_df, val_df, _ = split()
    print(f"  train={len(train_df):,}  val={len(val_df):,}")

    extractor = ForecastPointFeatureExtractor()

    print("Extracting features …")
    X_train = _prepare(train_df, extractor)
    X_val   = _prepare(val_df,   extractor)

    print("Computing synthetic labels …")
    y_train = _add_labels(train_df).values
    y_val   = _add_labels(val_df).values

    print("Fitting imputer on training data …")
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_val   = imputer.transform(X_val)

    if grid_search:
        print("Grid-searching hyperparameters (TimeSeriesSplit 5-fold) …")
        base = HistGradientBoostingRegressor(random_state=42)
        tscv = TimeSeriesSplit(n_splits=5)
        gs = GridSearchCV(
            base, PARAM_GRID, cv=tscv,
            scoring="r2", n_jobs=-1, verbose=1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        model = gs.best_estimator_
        best_params = gs.best_params_
        cv_r2 = float(gs.best_score_)
        print(f"  Best CV R²: {cv_r2:.4f}")
        print(f"  Best params: {best_params}")
    else:
        print("Training with default parameters …")
        model = HistGradientBoostingRegressor(**DEFAULT_PARAMS)
        model.fit(X_train, y_train)
        best_params = DEFAULT_PARAMS
        cv_r2 = float("nan")

    print("Evaluating on validation set …")
    y_pred = model.predict(X_val)
    val_r2  = r2_score(y_val, y_pred)
    val_mae = mean_absolute_error(y_val, y_pred)
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    print(f"  Val R²={val_r2:.4f}  MAE={val_mae:.2f}  RMSE={val_rmse:.2f}")

    model_path   = MODELS_DIR / "surf_condition_model.joblib"
    imputer_path = MODELS_DIR / "imputer.joblib"
    meta_path    = MODELS_DIR / "model_metadata.json"

    joblib.dump(model,   model_path)
    joblib.dump(imputer, imputer_path)

    metadata = {
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_hash":       _dataset_hash(train_df),
        "feature_list":       FEATURE_NAMES,
        "best_params":        best_params,
        "cv_r2_mean":         cv_r2,
        "val_r2":             val_r2,
        "val_mae":            val_mae,
        "val_rmse":           val_rmse,
        "train_rows":         len(train_df),
        "val_rows":           len(val_df),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:")
    print(f"  {model_path}   ({model_path.stat().st_size // 1024} KB)")
    print(f"  {imputer_path}")
    print(f"  {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-search", action="store_true",
                        help="Skip grid search and use default hyperparameters")
    args = parser.parse_args()
    train(grid_search=not args.no_search)
