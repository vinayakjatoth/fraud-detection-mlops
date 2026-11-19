"""Quality gate: decides whether a freshly trained model may be deployed.

Exit code 0 = pass, 1 = fail. Used by the retraining workflow before the deploy job.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .config import load_config


def check_gate(metrics: dict, gate: dict, baseline: dict | None = None) -> tuple[bool, list[str]]:
    """Return (passed, human-readable lines)."""
    lines: list[str] = []
    passed = True

    def check(name: str, value: float, minimum: float) -> None:
        nonlocal passed
        ok = value >= minimum
        passed &= ok
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name} = {value:.4f} (min {minimum:.4f})")

    check("roc_auc", metrics["roc_auc"], gate["min_roc_auc"])
    check("precision", metrics["precision"], gate["min_precision"])
    check("recall", metrics["recall"], gate["min_recall"])
    if "min_f1" in gate:
        check("f1", metrics["f1"], gate["min_f1"])

    if baseline is not None:
        tol = gate.get("max_pr_auc_regression", 0.02)
        drop = baseline["pr_auc"] - metrics["pr_auc"]
        ok = drop <= tol
        passed &= ok
        lines.append(
            f"{'PASS' if ok else 'FAIL'}  pr_auc regression vs deployed model = {drop:+.4f} "
            f"(new {metrics['pr_auc']:.4f}, deployed {baseline['pr_auc']:.4f}, max drop {tol})"
        )
    else:
        lines.append("INFO  no deployed baseline found - skipping regression check")
    return passed, lines


def _write_step_summary(passed: bool, metrics: dict, lines: list[str]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    cm = metrics["confusion_matrix"]
    md = [
        f"## Model quality gate: {'✅ passed' if passed else '❌ failed'}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
        f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
        f"| PR-AUC | {metrics['pr_auc']:.4f} |",
        f"| Confusion (TN/FP/FN/TP) | {cm['tn']} / {cm['fp']} / {cm['fn']} / {cm['tp']} |",
        "",
        "```",
        *lines,
        "```",
    ]
    with open(summary, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check evaluation metrics against the quality gate."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--metrics", default="reports/metrics.json")
    parser.add_argument(
        "--baseline", default=None, help="metrics.json of the currently deployed model"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    baseline = None
    if args.baseline and Path(args.baseline).exists():
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))

    passed, lines = check_gate(metrics, cfg["quality_gate"], baseline)
    print("\n".join(lines))
    _write_step_summary(passed, metrics, lines)
    print("\nQuality gate:", "PASSED" if passed else "FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
