"""Batch scoring from the command line.

python -m fraud_detection.predict --input transactions.csv --output scored.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from .data import FEATURE_COLUMNS
from .model import FraudDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Score transactions with a trained model.")
    parser.add_argument("--model", default="models/fraud_model.joblib")
    parser.add_argument("--input", required=True, help="CSV with Time, V1..V28, Amount")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    detector = FraudDetector.load(args.model)
    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Input is missing columns: {missing}")

    proba = detector.predict_proba(df[FEATURE_COLUMNS])
    df["fraud_probability"] = proba
    df["is_fraud"] = (proba >= detector.threshold).astype(int)
    df.to_csv(args.output, index=False)
    print(
        f"Scored {len(df):,} rows -> {args.output} ({int(df['is_fraud'].sum())} flagged as fraud)"
    )


if __name__ == "__main__":
    main()
