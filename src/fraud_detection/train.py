"""Train the fraud model.

python -m fraud_detection.train                                  # production pipeline
python -m fraud_detection.train --config configs/baseline_xgb_smote.yaml   # notebook baseline
python -m fraud_detection.train --synthetic                      # synthetic smoke test
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import datetime, timezone
from pathlib import Path

import sklearn
import xgboost
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .config import load_config
from .data import load_dataset, split_data
from .evaluate import select_threshold
from .features import FraudFeatureEngineer
from .model import FraudDetector

logger = logging.getLogger(__name__)


def _build_model(cfg: dict, seed: int, scale_pos_weight: float | None, has_val: bool):
    """Create the estimator named in cfg['model']['type'] (None-valued params are dropped)."""
    mtype = cfg["model"].get("type", "xgboost")
    if mtype == "xgboost":
        params = {k: v for k, v in cfg["model"]["xgboost"].items() if v is not None}
        if not has_val:  # early stopping needs a validation set
            params.pop("early_stopping_rounds", None)
        if scale_pos_weight is not None:
            params["scale_pos_weight"] = scale_pos_weight
        params.update(random_state=seed, n_jobs=-1)
        return XGBClassifier(**params), params
    if mtype == "logistic_regression":
        params = {k: v for k, v in cfg["model"]["logistic_regression"].items() if v is not None}
        if scale_pos_weight is not None:
            params["class_weight"] = "balanced"
        return LogisticRegression(**params), params
    raise ValueError(f"Unknown model type: {mtype}")


def train_model(
    cfg: dict,
    synthetic: bool = False,
    n_synthetic: int = 50_000,
    model_dir: str | Path | None = None,
) -> FraudDetector:
    seed = cfg["data"]["random_state"]
    model_dir = Path(model_dir or cfg["artifacts"]["model_dir"])

    df = load_dataset(cfg, synthetic=synthetic, n_synthetic=n_synthetic)
    s = split_data(df, cfg)
    has_val = s.X_val is not None
    logger.info(
        "Rows: train=%d val=%s test=%d | fraud rate (train)=%.4f%%",
        len(s.X_train),
        len(s.X_val) if has_val else "-",
        len(s.X_test),
        100 * s.y_train.mean(),
    )

    # Feature engineering is fit on the training split only.
    fe = FraudFeatureEngineer(**cfg.get("features", {})).fit(s.X_train, s.y_train)
    X_train = fe.transform(s.X_train)
    X_val = fe.transform(s.X_val) if has_val else None
    y_fit = s.y_train

    # Class-imbalance handling
    strategy = cfg.get("imbalance", {}).get("strategy", "scale_pos_weight")
    n_pos, n_neg = int(s.y_train.sum()), int((s.y_train == 0).sum())
    spw = None
    if strategy == "scale_pos_weight":
        spw = n_neg / max(n_pos, 1)
    elif strategy == "smote":
        from imblearn.over_sampling import SMOTE  # only needed for this strategy

        X_train, y_fit = SMOTE(random_state=seed).fit_resample(X_train, s.y_train)
        logger.info("SMOTE: %d -> %d training rows", len(s.y_train), len(y_fit))
    elif strategy != "none":
        raise ValueError(f"Unknown imbalance strategy: {strategy}")

    model, params = _build_model(cfg, seed, spw, has_val)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if isinstance(model, XGBClassifier):
            fit_kwargs = {"verbose": False}
            if has_val and "early_stopping_rounds" in params:
                fit_kwargs["eval_set"] = [(X_val, s.y_val)]
            model.fit(X_train, y_fit, **fit_kwargs)
        else:
            model.fit(X_train, y_fit)

    # Decision threshold: fixed, or tuned on the validation split (the test split stays untouched).
    th_cfg = cfg.get("threshold", {})
    th_strategy = th_cfg.get("strategy", "max_f1")
    if th_strategy == "fixed":
        threshold = float(th_cfg.get("value", 0.5))
    elif not has_val:
        raise ValueError("Threshold tuning needs a validation split (set data.val_size > 0)")
    else:
        val_proba = model.predict_proba(X_val)[:, 1]
        threshold = select_threshold(
            s.y_val, val_proba, th_strategy, th_cfg.get("target_precision", 0.90)
        )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "synthetic" if synthetic else "real",
        "model_type": cfg["model"].get("type", "xgboost"),
        "imbalance_strategy": strategy,
        "engineer_features": bool(cfg.get("features", {}).get("engineer_features", True)),
        "n_train": len(s.X_train),
        "n_val": len(s.X_val) if has_val else 0,
        "n_test": len(s.X_test),
        "train_fraud_rate": float(s.y_train.mean()),
        "scale_pos_weight": spw,
        "best_iteration": (
            int(model.best_iteration)
            if getattr(model, "best_iteration", None) is not None
            else None
        ),
        "threshold": threshold,
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
    }
    detector = FraudDetector(
        feature_engineer=fe, model=model, threshold=threshold, metadata=metadata
    )

    path = detector.save(model_dir / "fraud_model.joblib")
    (model_dir / "train_info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved model to %s (threshold=%.4f)", path, threshold)
    return detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fraud detection model.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data (smoke test only)"
    )
    parser.add_argument("--n-synthetic", type=int, default=50_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train_model(load_config(args.config), args.synthetic, args.n_synthetic, args.model_dir)


if __name__ == "__main__":
    main()
