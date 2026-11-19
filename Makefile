.PHONY: install data train evaluate gate pipeline baselines results all smoke test lint serve docker

install:        ## install package + dev tools
	pip install -e ".[dev]"

data:           ## download the real dataset (284,807 transactions)
	python scripts/download_data.py

train:          ## train on the real dataset
	python -m fraud_detection.train

evaluate:       ## evaluate saved model on the held-out test split
	python -m fraud_detection.evaluate

gate:           ## check metrics against the quality gate
	python -m fraud_detection.quality_gate

pipeline: train evaluate gate   ## full pipeline on real data

baselines:      ## reproduce the notebook's LogReg+SMOTE and XGBoost+SMOTE models
	python -m fraud_detection.train --config configs/baseline_logreg.yaml
	python -m fraud_detection.evaluate --config configs/baseline_logreg.yaml
	python -m fraud_detection.train --config configs/baseline_xgb_smote.yaml
	python -m fraud_detection.evaluate --config configs/baseline_xgb_smote.yaml

results:        ## write the real-data metrics into the README results table
	python scripts/update_readme_results.py

all: data baselines pipeline results   ## everything, on the real dataset

smoke:          ## full pipeline on synthetic data (no download, ~1 min)
	python -m fraud_detection.train --synthetic --model-dir /tmp/fraud_smoke/models
	python -m fraud_detection.evaluate --synthetic --model /tmp/fraud_smoke/models/fraud_model.joblib --reports-dir /tmp/fraud_smoke/reports
	python -m fraud_detection.quality_gate --metrics /tmp/fraud_smoke/reports/metrics.json

test:
	pytest

lint:
	ruff check .

serve:          ## run the API locally
	uvicorn fraud_detection.api:app --reload --port 8000

docker:         ## build the serving image (needs models/fraud_model.joblib)
	docker build -t fraud-detection-api .
