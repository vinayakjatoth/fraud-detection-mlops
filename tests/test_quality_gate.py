from fraud_detection.quality_gate import check_gate

GATE = {"min_roc_auc": 0.95, "min_precision": 0.7, "min_recall": 0.7, "max_pr_auc_regression": 0.02}
GOOD = {"roc_auc": 0.98, "precision": 0.9, "recall": 0.8, "pr_auc": 0.85}


def test_passes_without_baseline():
    passed, lines = check_gate(GOOD, GATE)
    assert passed
    assert any("no deployed baseline" in line for line in lines)


def test_fails_when_metric_below_minimum():
    passed, lines = check_gate({**GOOD, "recall": 0.5}, GATE)
    assert not passed
    assert any(line.startswith("FAIL  recall") for line in lines)


def test_fails_on_regression_vs_baseline():
    passed, _ = check_gate({**GOOD, "pr_auc": 0.70}, GATE, baseline={"pr_auc": 0.85})
    assert not passed


def test_small_regression_is_tolerated():
    passed, _ = check_gate({**GOOD, "pr_auc": 0.84}, GATE, baseline={"pr_auc": 0.85})
    assert passed
