import json

import numpy as np

from fraud_detection.evaluate import compute_metrics, evaluate_model, select_threshold
from fraud_detection.model import FraudDetector


def test_model_is_saved_and_reloadable(trained):
    detector, model_dir = trained
    assert (model_dir / "fraud_model.joblib").exists()
    assert (model_dir / "train_info.json").exists()
    reloaded = FraudDetector.load(model_dir / "fraud_model.joblib")
    assert reloaded.threshold == detector.threshold


def test_evaluation_writes_metrics_and_figures(cfg, trained, tmp_path):
    _, model_dir = trained
    metrics = evaluate_model(
        cfg, model_dir / "fraud_model.joblib", tmp_path, synthetic=True, n_synthetic=12_000
    )
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["roc_auc"] == metrics["roc_auc"]
    assert metrics["roc_auc"] > 0.9  # synthetic fraud is learnable
    for name in ["pr_curve", "roc_curve", "confusion_matrix", "feature_importance"]:
        assert (tmp_path / "figures" / f"{name}.png").exists()


def test_compute_metrics_known_values():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.3, 0.9, 0.4])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["confusion_matrix"] == {"tn": 3, "fp": 1, "fn": 1, "tp": 1}
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_threshold_strategies():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(950), np.ones(50)].astype(int)
    p = np.r_[rng.beta(2, 8, 950), rng.beta(8, 2, 50)]
    t_f1 = select_threshold(y, p, "max_f1")
    t_prec = select_threshold(y, p, "recall_at_precision", target_precision=0.95)
    assert 0 < t_f1 < 1
    assert t_prec >= t_f1 - 1e-9  # a stricter precision target never lowers the threshold
