"""The pipeline must be able to reproduce the notebook's setup."""

import json

import pandas as pd

from fraud_detection.config import load_config
from fraud_detection.data import FEATURE_COLUMNS, load_dataset, split_data
from fraud_detection.evaluate import evaluate_model
from fraud_detection.train import train_model

BASELINES = ["configs/baseline_logreg.yaml", "configs/baseline_xgb_smote.yaml"]


def test_extends_merges_and_null_unsets():
    cfg = load_config("configs/baseline_xgb_smote.yaml")
    assert cfg["data"]["val_size"] == 0.0 and cfg["data"]["test_size"] == 0.20  # merged
    xgb = cfg["model"]["xgboost"]
    assert (xgb["n_estimators"], xgb["max_depth"], xgb["gamma"]) == (500, 4, 1)  # notebook params
    assert xgb["early_stopping_rounds"] is None and xgb["eval_metric"] is None  # unset


def test_duplicates_are_dropped(cfg, tmp_path):
    from fraud_detection.synthetic import make_synthetic_transactions

    base = make_synthetic_transactions(2_000, fraud_rate=0.02, seed=2)
    dup = pd.concat([base, base.head(50)], ignore_index=True)
    path = tmp_path / "cc.csv"
    dup.to_csv(path, index=False)
    local = {**cfg, "data": {**cfg["data"], "raw_path": str(path)}}
    assert len(load_dataset(local)) == len(base)
    local["data"]["drop_duplicates"] = False
    assert len(load_dataset(local)) == len(base) + 50


def test_test_split_independent_of_validation_split(cfg):
    """Same test rows whether or not a validation split is carved out (matches the notebook)."""
    from fraud_detection.synthetic import make_synthetic_transactions

    df = make_synthetic_transactions(5_000, fraud_rate=0.02, seed=4)
    with_val = split_data(df, cfg)
    no_val = split_data(df, {**cfg, "data": {**cfg["data"], "val_size": 0.0}})
    assert no_val.X_val is None
    assert list(with_val.X_test.index) == list(no_val.X_test.index)
    assert len(no_val.X_train) == len(with_val.X_train) + len(with_val.X_val)


def test_baselines_train_and_evaluate(tmp_path):
    for i, path in enumerate(BASELINES):
        cfg = load_config(path)
        cfg["model"]["xgboost"]["n_estimators"] = 30
        model_dir, rep_dir = tmp_path / f"m{i}", tmp_path / f"r{i}"
        detector = train_model(cfg, synthetic=True, n_synthetic=6_000, model_dir=model_dir)
        assert detector.threshold == 0.5  # fixed, as in the notebook
        assert detector.metadata["imbalance_strategy"] == "smote"
        assert (
            list(
                detector.feature_engineer.transform(
                    pd.DataFrame(columns=FEATURE_COLUMNS, dtype=float).reindex(range(1)).fillna(0)
                ).columns
            )
            == FEATURE_COLUMNS
        )  # raw columns only, no engineered features
        m = evaluate_model(cfg, model_dir / "fraud_model.joblib", rep_dir, True, 6_000)
        assert 0.0 <= m["roc_auc"] <= 1.0
        assert json.loads((rep_dir / "metrics.json").read_text())["data_source"] == "synthetic"
