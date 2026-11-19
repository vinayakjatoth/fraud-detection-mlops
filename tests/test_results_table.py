import json

import pytest
from scripts_shim import build_table, update_readme

REAL = {
    "data_source": "real",
    "precision": 0.9,
    "recall": 0.8,
    "f1": 0.847,
    "roc_auc": 0.97,
    "pr_auc": 0.85,
    "n_samples": 100,
    "n_fraud": 10,
    "confusion_matrix": {"tn": 88, "fp": 2, "fn": 2, "tp": 8},
}


def test_only_real_metrics_are_published(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a/metrics.json").write_text(json.dumps(REAL))
    (tmp_path / "b/metrics.json").write_text(json.dumps({**REAL, "data_source": "synthetic"}))
    runs = [("real run", "a/metrics.json"), ("synthetic run", "b/metrics.json")]
    table = build_table(runs, root=tmp_path)
    assert "| real run | 0.900 |" in table
    assert "| synthetic run | - | - |" in table  # synthetic numbers never published


def test_update_readme_replaces_between_markers():
    text = "intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\noutro"
    out = update_readme(text, "NEW TABLE")
    assert (
        "NEW TABLE" in out
        and "old" not in out
        and out.startswith("intro")
        and out.endswith("outro")
    )
    with pytest.raises(SystemExit):
        update_readme("no markers", "x")
