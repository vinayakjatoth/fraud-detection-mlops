"""Metrics, threshold selection, report figures and the `evaluate` CLI."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import load_config  # noqa: E402
from .data import load_dataset, split_data  # noqa: E402
from .model import FraudDetector  # noqa: E402

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- thresholding
def select_threshold(
    y_true, proba, strategy: str = "max_f1", target_precision: float = 0.90
) -> float:
    """Choose the decision threshold on a validation set (never on the test set)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]  # align with `thresholds`
    if strategy == "max_f1":
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        return float(thresholds[int(np.argmax(f1))])
    if strategy == "recall_at_precision":
        ok = precision >= target_precision
        if not ok.any():
            logger.warning(
                "No threshold reaches precision %.2f; falling back to max_f1", target_precision
            )
            return select_threshold(y_true, proba, "max_f1")
        best = np.where(ok, recall, -1.0)
        return float(thresholds[int(np.argmax(best))])
    raise ValueError(f"Unknown threshold strategy: {strategy}")


# --------------------------------------------------------------------------- metrics
def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }


# --------------------------------------------------------------------------- figures
def save_report_figures(y_true, proba, detector: FraudDetector, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true)
    thr = detector.threshold
    pred = (np.asarray(proba) >= thr).astype(int)

    # Precision-recall curve with the chosen operating point
    p, r, _ = precision_recall_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(r, p, lw=2, label=f"PR-AUC = {average_precision_score(y_true, proba):.3f}")
    ax.scatter(
        [recall_score(y_true, pred, zero_division=0)],
        [precision_score(y_true, pred, zero_division=0)],
        c="red",
        zorder=5,
        label=f"operating point (thr={thr:.2f})",
    )
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test set)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, proba)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc_score(y_true, proba):.3f}")
    ax.plot([0, 1], [0, 1], "--", c="grey")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true, pred, display_labels=["Legit", "Fraud"], cmap="Blues", values_format="d", ax=ax
    )
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # Feature importance (gain)
    gain = (
        detector.model.get_booster().get_score(importance_type="gain")
        if hasattr(detector.model, "get_booster")
        else {}
    )
    if gain:
        top = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:15][::-1]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh([k for k, _ in top], [v for _, v in top], color="#4c72b0")
        ax.set(xlabel="Average gain", title="Top 15 features (XGBoost gain)")
        fig.tight_layout()
        fig.savefig(out / "feature_importance.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- CLI
def evaluate_model(
    cfg: dict,
    model_path: str | Path,
    reports_dir: str | Path,
    synthetic: bool = False,
    n_synthetic: int = 50_000,
) -> dict:
    """Evaluate a saved model on the held-out test split and write metrics + figures."""
    detector = FraudDetector.load(model_path)
    df = load_dataset(cfg, synthetic=synthetic, n_synthetic=n_synthetic)
    splits = split_data(df, cfg)

    proba = detector.predict_proba(splits.X_test)
    metrics = compute_metrics(splits.y_test, proba, detector.threshold)
    metrics["data_source"] = "synthetic" if synthetic else "real"
    metrics["model_metadata"] = detector.metadata

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_report_figures(splits.y_test, proba, detector, reports_dir / "figures")
    return metrics


def print_summary(metrics: dict) -> None:
    cm = metrics["confusion_matrix"]
    print("\n=== Test-set evaluation ===")
    print(f"samples: {metrics['n_samples']:,}   frauds: {metrics['n_fraud']:,}")
    print(f"threshold : {metrics['threshold']:.4f}")
    print(f"precision : {metrics['precision']:.4f}")
    print(f"recall    : {metrics['recall']:.4f}")
    print(f"f1        : {metrics['f1']:.4f}")
    print(f"roc_auc   : {metrics['roc_auc']:.4f}")
    print(f"pr_auc    : {metrics['pr_auc']:.4f}")
    print(f"confusion : TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test split.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model", default=None, help="Path to model (default: <model_dir>/fraud_model.joblib)"
    )
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic data (smoke test only)"
    )
    parser.add_argument("--n-synthetic", type=int, default=50_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    model_path = args.model or Path(cfg["artifacts"]["model_dir"]) / "fraud_model.joblib"
    reports_dir = args.reports_dir or cfg["artifacts"]["reports_dir"]
    metrics = evaluate_model(cfg, model_path, reports_dir, args.synthetic, args.n_synthetic)
    print_summary(metrics)
    print(f"\nWrote {Path(reports_dir) / 'metrics.json'} and figures/")


if __name__ == "__main__":
    main()
