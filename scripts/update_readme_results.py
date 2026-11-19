"""Fill the README results table from the metrics.json files written by the pipeline.

    make baselines pipeline results

Only metrics computed on the REAL dataset are published; anything produced with --synthetic
is skipped so smoke-test numbers can never end up in the README.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"

RUNS = [
    ("Logistic Regression + SMOTE (notebook baseline)", "reports/baseline_logreg/metrics.json"),
    ("XGBoost + SMOTE (notebook baseline)", "reports/baseline_xgb_smote/metrics.json"),
    ("**This pipeline**: feature engineering + XGBoost + tuned threshold", "reports/metrics.json"),
]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    metrics = json.loads(path.read_text(encoding="utf-8"))
    return metrics if metrics.get("data_source") == "real" else None


def build_table(runs=RUNS, root: Path = Path(".")) -> str:
    rows = [
        "| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Frauds caught | False alarms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    info = ""
    for name, rel in runs:
        m = _load(root / rel)
        if m is None:
            rows.append(f"| {name} | - | - | - | - | - | - | - |")
            continue
        cm = m["confusion_matrix"]
        caught = f"{cm['tp']} / {cm['tp'] + cm['fn']}"
        rows.append(
            f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | "
            f"{m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {caught} | {cm['fp']} |"
        )
        info = f"Test set: {m['n_samples']:,} transactions, {m['n_fraud']} frauds."
    return "\n".join(rows) + (f"\n\n{info}" if info else "")


def update_readme(text: str, table: str) -> str:
    if START not in text or END not in text:
        raise SystemExit(f"README is missing the {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{table}\n{END}{tail}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readme", default="README.md")
    args = parser.parse_args()

    if not any(_load(Path(rel)) for _, rel in RUNS):
        raise SystemExit("No real-data metrics found. Run `make baselines pipeline` first.")
    table = build_table()
    readme = Path(args.readme)
    readme.write_text(update_readme(readme.read_text(encoding="utf-8"), table), encoding="utf-8")
    print(table)
    print(f"\nUpdated {readme}")


if __name__ == "__main__":
    main()
