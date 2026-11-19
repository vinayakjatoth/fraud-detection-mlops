"""Deployable model bundle: feature engineering + XGBoost + decision threshold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from .features import FraudFeatureEngineer


@dataclass
class FraudDetector:
    feature_engineer: FraudFeatureEngineer
    model: ClassifierMixin
    threshold: float = 0.5
    metadata: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability that each transaction is fraud."""
        feats = self.feature_engineer.transform(X)[self.feature_engineer.feature_names_]
        return self.model.predict_proba(feats)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Binary fraud decision using the tuned threshold."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> FraudDetector:
        # joblib uses pickle: only load model files you trained or trust.
        return joblib.load(path)
