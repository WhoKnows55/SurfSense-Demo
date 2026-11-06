"""
ML inference wrapper for the trained XGBoost surf condition model.

Loads model + imputer on instantiation, exposes predict / predict_batch /
get_feature_contributions.  SHAP values are computed with TreeExplainer
(exact, O(ms) per call – no caching needed).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from app.forecasting.models import ForecastPoint
from app.ml.feature_extractor import FEATURE_NAMES, ForecastPointFeatureExtractor

logger = logging.getLogger(__name__)


class SurfConditionModel:
    """Load and serve the trained XGBoost surf condition model.

    Args:
        model_path: Path to surf_condition_model.joblib.  The imputer
            (imputer.joblib) and metadata (model_metadata.json) are expected
            in the same directory.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ImportError: If joblib or shap are not installed.
    """

    def __init__(self, model_path: str) -> None:
        import joblib  # deferred – not installed until pip install runs
        import shap

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                "Run `python -m ml.train` to train the model first."
            )

        self._model = joblib.load(path)
        self._extractor = ForecastPointFeatureExtractor()

        imputer_path = path.parent / "imputer.joblib"
        self._imputer = joblib.load(imputer_path) if imputer_path.exists() else None

        meta_path = path.parent / "model_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._metadata = json.load(f)
            logger.info(
                "SurfConditionModel loaded | version=%s r2_val=%.3f",
                self._metadata.get("training_timestamp", "unknown"),
                self._metadata.get("cv_r2_mean", float("nan")),
            )
        else:
            self._metadata = {}

        self._explainer = shap.TreeExplainer(self._model)

    # -- Internal ------------------------------------------------------------

    def _prepare(self, fp: ForecastPoint, skill_level: str) -> np.ndarray:
        X = self._extractor.transform_point(fp, skill_level).reshape(1, -1)
        if self._imputer is not None:
            X = self._imputer.transform(X)
        return X

    # -- Public API ----------------------------------------------------------

    def predict(self, forecast: ForecastPoint, skill_level: str = "intermediate") -> float:
        """Return a single 0–100 quality score."""
        score = float(self._model.predict(self._prepare(forecast, skill_level))[0])
        return max(0.0, min(100.0, score))

    def predict_batch(
        self, forecasts: list[ForecastPoint], skill_level: str = "intermediate"
    ) -> list[float]:
        """Return a list of 0–100 quality scores for multiple ForecastPoints."""
        if not forecasts:
            return []
        X = np.vstack([self._prepare(fp, skill_level) for fp in forecasts])
        return [max(0.0, min(100.0, float(s))) for s in self._model.predict(X)]

    def get_feature_contributions(
        self, forecast: ForecastPoint, skill_level: str = "intermediate"
    ) -> dict[str, float]:
        """Return signed SHAP contributions for one prediction.

        Positive value = feature raised the score; negative = lowered it.
        """
        X = self._prepare(forecast, skill_level)
        shap_vals = self._explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        return dict(zip(FEATURE_NAMES, shap_vals[0].tolist()))
