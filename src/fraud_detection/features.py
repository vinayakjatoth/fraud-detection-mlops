"""Domain-driven feature engineering for card transactions.

The public dataset anonymises 28 PCA components (V1..V28) and exposes only `Time` and
`Amount` in raw form, so the engineered features focus on fraud behaviour that can still be
expressed from those columns:

* temporal  - hour-of-day (cyclical) and an off-peak "night" flag; fraud skews to quiet hours.
              NOTE: `Time` is seconds since the first transaction, not a wall clock, so
              `is_night` is a proxy that relies on the dataset starting near midnight.
* amount    - log / z-scored amount, "micro" amounts (card-testing probes) and round amounts.
* component - L2 norm of the PCA vector (distance from typical behaviour) and composite
              signals built from the V-components most correlated with fraud.

The transformer learns its statistics (amount mean/std and which V-components form the
composites) from the TRAINING split only, so there is no leakage into validation/test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .synthetic import V_COLUMNS


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        top_k_signal: int = 4,
        night_start_hour: int = 0,
        night_end_hour: int = 6,
        micro_amount: float = 2.0,
        engineer_features: bool = True,
    ):
        self.engineer_features = engineer_features
        self.top_k_signal = top_k_signal
        self.night_start_hour = night_start_hour
        self.night_end_hour = night_end_hour
        self.micro_amount = micro_amount

    def fit(self, X: pd.DataFrame, y) -> FraudFeatureEngineer:
        X = pd.DataFrame(X)
        self.feature_medians_ = X[["Time", *V_COLUMNS, "Amount"]].median().to_dict()
        X = X.fillna(self.feature_medians_)
        log_amount = np.log1p(X["Amount"].clip(lower=0))
        std = float(log_amount.std())
        self.log_amount_mean_ = float(log_amount.mean())
        self.log_amount_std_ = std if np.isfinite(std) and std > 0 else 1.0

        # Pick the V-components most (anti-)correlated with fraud, using training data only.
        corr = X[V_COLUMNS].corrwith(pd.Series(np.asarray(y), index=X.index)).fillna(0.0)
        self.negative_signal_cols_ = list(corr[corr < 0].nsmallest(self.top_k_signal).index)
        self.positive_signal_cols_ = list(corr[corr > 0].nlargest(self.top_k_signal).index)

        self.feature_names_ = list(self.transform(X.iloc[:5]).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "log_amount_mean_")
        X = pd.DataFrame(X)
        X = X.fillna(self.feature_medians_)
        if not self.engineer_features:  # raw columns only (notebook behaviour)
            return X[["Time", *V_COLUMNS, "Amount"]].copy()
        out = X[[*V_COLUMNS, "Amount"]].copy()

        # --- temporal -------------------------------------------------------------------
        hour = (X["Time"] // 3600) % 24
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["is_night"] = ((hour >= self.night_start_hour) & (hour < self.night_end_hour)).astype(
            int
        )

        # --- amount ---------------------------------------------------------------------
        amount = X["Amount"].clip(lower=0)
        log_amount = np.log1p(amount)
        out["log_amount"] = log_amount
        out["amount_zscore"] = (log_amount - self.log_amount_mean_) / self.log_amount_std_
        out["is_micro_amount"] = (amount < self.micro_amount).astype(int)
        out["is_round_amount"] = ((amount > 0) & (amount % 1 == 0)).astype(int)

        # --- PCA-component aggregates ---------------------------------------------------
        v = X[V_COLUMNS]
        out["v_l2_norm"] = np.sqrt((v**2).sum(axis=1))
        out["v_neg_signal"] = (
            v[self.negative_signal_cols_].mean(axis=1) if self.negative_signal_cols_ else 0.0
        )
        out["v_pos_signal"] = (
            v[self.positive_signal_cols_].mean(axis=1) if self.positive_signal_cols_ else 0.0
        )
        out["signal_gap"] = out["v_pos_signal"] - out["v_neg_signal"]
        return out
