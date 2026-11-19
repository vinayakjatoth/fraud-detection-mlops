# Online Fraud Detection & Automated ML Pipeline

[![CI](https://github.com/YOUR_USERNAME/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/fraud-detection-mlops/actions/workflows/ci.yml)
[![Retrain](https://github.com/YOUR_USERNAME/fraud-detection-mlops/actions/workflows/retrain.yml/badge.svg)](https://github.com/YOUR_USERNAME/fraud-detection-mlops/actions/workflows/retrain.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end machine learning pipeline for **credit card fraud detection** on a highly imbalanced
dataset. It turns the exploration in [`notebooks/credit-card-fraud-detection.ipynb`](notebooks/credit-card-fraud-detection.ipynb)
(EDA, Logistic Regression + SMOTE, XGBoost + SMOTE) into a reproducible pipeline with feature
engineering, leakage-safe evaluation, a model **quality gate**, a FastAPI service and
**GitHub Actions CI/CD** for automated retraining, evaluation and deployment.

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB / Worldline):
284,807 transactions over two days, 492 frauds. Columns: `Time`, `V1`-`V28` (anonymised PCA
components), `Amount`, `Class`. After removing 1,081 exact duplicate rows (as in the notebook),
**283,726 transactions remain, 473 of them fraud (0.167%)**.

The pipeline downloads the data from OpenML on first run. To use your own copy, put
`creditcard.csv` in `data/raw/` or set `FRAUD_DATA_PATH`.

## From notebook to pipeline

| Step | Notebook | This repo |
|---|---|---|
| Cleaning | drop duplicates | same (`data.drop_duplicates`) |
| Split | 80/20 stratified, `random_state=42` | same test rows; plus a validation split carved from the training part |
| Imbalance | SMOTE on the training set | SMOTE kept as a baseline; default pipeline uses class weighting + threshold tuning |
| Models | Logistic Regression, XGBoost | both reproduced as baselines (`configs/baseline_*.yaml`) |
| Features | raw columns | + temporal, amount-behaviour and PCA-aggregate features (leakage-safe) |
| Metrics | accuracy, precision, recall, confusion matrix | + F1, ROC-AUC, PR-AUC, curves, feature importance |
| Threshold | fixed 0.5 | tuned on the validation split |
| Automation | - | tests, quality gate, CI, scheduled retraining, Docker, API |

Every model in the results table is scored on the **same held-out test set** (56,746 transactions,
95 frauds), so the rows are directly comparable.

## Results

<!-- RESULTS:START -->
| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC | Frauds caught | False alarms |
|---|---|---|---|---|---|---|---|
| Logistic Regression + SMOTE (notebook baseline) | 0.073 | 0.884 | 0.135 | 0.958 | 0.665 | 84 / 95 | 1066 |
| XGBoost + SMOTE (notebook baseline) | 0.717 | 0.800 | 0.756 | 0.967 | 0.811 | 76 / 95 | 30 |
| **This pipeline**: feature engineering + XGBoost + tuned threshold | 0.986 | 0.726 | 0.836 | 0.969 | 0.816 | 69 / 95 | 1 |

Test set: 56,746 transactions, 95 frauds.
<!-- RESULTS:END -->

The table is regenerated from `reports/**/metrics.json` and contains verified real-dataset results:
Logistic Regression + SMOTE prioritizes recall (0.884) but produces 1,066 false alarms; XGBoost +
SMOTE provides a stronger precision-recall balance; and the default engineered XGBoost pipeline
uses validation-tuned threshold `0.9851` to achieve 0.986 precision and 69 detected frauds with
only one false alarm on this test set.

## Quick start

```powershell
git clone https://github.com/YOUR_USERNAME/fraud-detection-mlops.git
Set-Location fraud-detection-mlops
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

python scripts\download_data.py
python -m fraud_detection.train --config configs\baseline_logreg.yaml
python -m fraud_detection.evaluate --config configs\baseline_logreg.yaml
python -m fraud_detection.train --config configs\baseline_xgb_smote.yaml
python -m fraud_detection.evaluate --config configs\baseline_xgb_smote.yaml
python -m fraud_detection.train
python -m fraud_detection.evaluate
python -m fraud_detection.quality_gate
python scripts\update_readme_results.py
```

Individual steps: `make baselines`, `make pipeline` (train -> evaluate -> quality gate), `make results`.
Outputs land in `models/`, `reports/metrics.json` and `reports/figures/` (PR curve, ROC curve,
confusion matrix, feature importance); baselines write to `models/baseline_*` and `reports/baseline_*`.

`make smoke` runs the whole pipeline on synthetic data in about a minute (no download). Synthetic
metrics are for smoke-testing only and are never written to the results table.

**On Kaggle:** enable Internet, then

```python
!git clone https://github.com/YOUR_USERNAME/fraud-detection-mlops.git
%cd fraud-detection-mlops
!pip install -q -e .
%env FRAUD_DATA_PATH=/kaggle/input/creditcardfraud/creditcard.csv
!make baselines pipeline results
```

## Pipeline

```
 data (CSV / OpenML) -> drop duplicates -> stratified train / (val) / test split
        |
        v
 FraudFeatureEngineer (fit on train only) -> [SMOTE | class weights] -> XGBoost / LogReg
        |                                            |
        |                       threshold tuned on validation set
        v                                            v
              FraudDetector bundle (models/fraud_model.joblib)
                       |
      +----------------+-----------------+
      v                                  v
 evaluate on held-out test set     FastAPI service / Docker image
 -> reports/metrics.json + figures
      |
      v
 quality gate (absolute floors + no regression vs deployed model)
      |
      v  pass
 GitHub Release (model + metrics) + image on GHCR
```

**Features** (`src/fraud_detection/features.py`): cyclical hour-of-day and a night flag; log and
z-scored amount, "micro" amounts (card-testing probes) and round amounts; L2 norm of the PCA vector
and composite signals built from the V-components most correlated with fraud. All statistics are
learned on the training split only. `Time` is seconds since the first transaction, so `is_night`
is a proxy that assumes the data starts near midnight.

## Serving the model

```bash
make serve
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict \
  -H "Content-Type: application/json" -d @examples/sample_transaction.json
```

Interactive docs at `http://localhost:8000/docs`. With Docker (after training):
`make docker && docker run -p 8000:8000 fraud-detection-api`.
Batch scoring from a CSV: `python -m fraud_detection.predict --input transactions.csv`.

## CI/CD

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push to `main`, pull requests | ruff lint + format check, pytest on Python 3.10-3.12, end-to-end smoke test on synthetic data |
| `retrain.yml` | weekly (Mon 03:00 UTC), manual, or push to `main` touching `src/`, `configs/`, `requirements.txt` | download (cached) data -> train -> evaluate on the held-out test set -> **quality gate** -> upload artifacts -> on success: GitHub Release (model + metrics) and a smoke-tested Docker image pushed to GHCR |

The **quality gate** (`configs/config.yaml -> quality_gate`) fails the run, and so blocks
deployment, if ROC-AUC / precision / recall fall below the configured floors or if PR-AUC regresses
by more than 0.02 versus the model in the latest release. Metrics are also written to the job
summary. Manual run: **Actions -> Retrain, Evaluate & Deploy -> Run workflow**.

No secrets are needed (built-in `GITHUB_TOKEN`). If publishing is denied, enable *Settings ->
Actions -> General -> Workflow permissions -> Read and write*.

## Project structure

```
.
├── .github/workflows/   ci.yml, retrain.yml
├── configs/             config.yaml (pipeline), baseline_logreg.yaml, baseline_xgb_smote.yaml
├── notebooks/           credit-card-fraud-detection.ipynb (EDA + baselines)
├── scripts/             download_data.py, update_readme_results.py
├── src/fraud_detection/ data, features, model, train, evaluate, quality_gate, predict, api, synthetic
├── tests/               pytest suite (incl. notebook-parity tests)
├── examples/            sample API payload
└── Dockerfile, Makefile, pyproject.toml, requirements.txt
```

## Design notes

- **Metrics.** Accuracy is meaningless at 0.17% fraud (predicting "legit" every time scores 99.8%),
  so the pipeline reports precision, recall, F1, ROC-AUC and PR-AUC. PR-AUC drives early stopping
  and the regression check because it is far more sensitive to minority-class quality.
- **Imbalance.** SMOTE is applied to the training split only, never to validation or test data.
  The default pipeline uses class weighting and threshold tuning instead; set
  `imbalance.strategy: smote` in `configs/config.yaml` to compare.
- **Threshold.** Chosen on the validation split (`max_f1`, or `recall_at_precision` for a business
  precision target). The test set is used once, for the final report.
- **Reproducibility.** Fixed seed and deterministic splits; library versions are stored in the
  model metadata. The notebook baselines land close to, but not bit-identical with, the notebook
  numbers because library versions and random seeds differ between environments.

## Limitations & next steps

- Retraining here re-uses a static public dataset. In production, point `data.raw_path` at a
  warehouse export so scheduled retraining sees fresh labelled transactions.
- The public data has no per-card history (velocity, merchant, geography); those are the
  highest-value additions for a real system.
- Ideas: probability calibration, SHAP explanations, drift monitoring, cost-sensitive thresholds.
- Model files are pickled with joblib: only load models you trained or trust.

## License

MIT - see [LICENSE](LICENSE).
