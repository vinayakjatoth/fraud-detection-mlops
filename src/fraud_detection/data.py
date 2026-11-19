"""Data loading, schema validation and train/validation/test splitting."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .synthetic import V_COLUMNS, make_synthetic_transactions

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["Time", *V_COLUMNS, "Amount"]


class Splits(NamedTuple):
    X_train: pd.DataFrame
    X_val: pd.DataFrame | None  # None when val_size is 0
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series | None
    y_test: pd.Series


def download_openml(openml_id: int, dest: str | Path) -> Path:
    """Download the credit-card fraud dataset from OpenML and store it as CSV."""
    from sklearn.datasets import fetch_openml

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading OpenML dataset %s ...", openml_id)
    frame = fetch_openml(data_id=openml_id, as_frame=True, parser="auto").frame.copy()
    frame["Class"] = frame["Class"].astype(int)
    frame.to_csv(dest, index=False)
    logger.info("Saved %d rows to %s", len(frame), dest)
    return dest


def validate_schema(df: pd.DataFrame, target: str) -> None:
    missing = [c for c in [*FEATURE_COLUMNS, target] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[target].isna().any():
        raise ValueError("Target column contains missing values")
    numeric = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    nonnumeric = df[FEATURE_COLUMNS].notna() & numeric.isna()
    if nonnumeric.any().any():
        raise ValueError("Feature columns contain non-numeric values")
    if not numeric.map(lambda value: abs(float(value)) != float("inf")).all().all():
        raise ValueError("Feature columns contain infinite values")
    if not set(df[target].unique()) <= {0, 1}:
        raise ValueError("Target must be binary (0 = legit, 1 = fraud)")
    if df[target].sum() == 0:
        raise ValueError("Dataset contains no fraud examples")


def load_dataset(
    cfg: dict, synthetic: bool = False, n_synthetic: int = 50_000, fraud_rate: float = 0.005
) -> pd.DataFrame:
    """Load the real dataset (downloading it if needed) or generate synthetic data."""
    target = cfg["data"]["target"]
    if synthetic:
        logger.warning("Using SYNTHETIC data - metrics are for smoke-testing only.")
        df = make_synthetic_transactions(
            n_samples=n_synthetic, fraud_rate=fraud_rate, seed=cfg["data"]["random_state"]
        )
    else:
        # FRAUD_DATA_PATH lets you point at a CSV elsewhere (e.g. /kaggle/input/... on Kaggle)
        path = Path(os.getenv("FRAUD_DATA_PATH") or cfg["data"]["raw_path"])
        if not path.exists():
            download_openml(cfg["data"]["openml_id"], path)
        df = pd.read_csv(path)
    validate_schema(df, target)
    if cfg["data"].get("drop_duplicates", False):
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Dropped %d duplicate rows (%d -> %d)", before - len(df), before, len(df))
    return df


def split_data(df: pd.DataFrame, cfg: dict) -> Splits:
    """Stratified train / validation / test split (deterministic given random_state)."""
    d = cfg["data"]
    X = df[FEATURE_COLUMNS]
    y = df[d["target"]].astype(int)
    seed = d["random_state"]
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=d["test_size"], stratify=y, random_state=seed
    )
    val_size = d.get("val_size", 0.0)
    if val_size and val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X_rest, y_rest, test_size=val_size, stratify=y_rest, random_state=seed
        )
    else:  # no validation split: train on the whole non-test portion (as the notebook does)
        X_train, y_train, X_val, y_val = X_rest, y_rest, None, None
    return Splits(X_train, X_val, X_test, y_train, y_val, y_test)
