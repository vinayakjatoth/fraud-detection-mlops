"""FastAPI service that serves the trained fraud model.

MODEL_PATH=models/fraud_model.joblib uvicorn fraud_detection.api:app --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, create_model

from . import __version__
from .data import FEATURE_COLUMNS
from .model import FraudDetector

logger = logging.getLogger("fraud_detection.api")

# One field per model input: Time, V1..V28, Amount.
Transaction = create_model(
    "Transaction",
    Time=(
        float,
        Field(
            ...,
            ge=0,
            allow_inf_nan=False,
            description="Seconds elapsed since the first transaction",
        ),
    ),
    **{
        f"V{i}": (
            float,
            Field(..., allow_inf_nan=False, description=f"PCA component V{i}"),
        )
        for i in range(1, 29)
    },
    Amount=(float, Field(..., ge=0, allow_inf_nan=False, description="Transaction amount")),
)


class Prediction(BaseModel):
    fraud_probability: float
    is_fraud: bool
    threshold: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(os.getenv("MODEL_PATH", "models/fraud_model.joblib"))
    if path.exists():
        app.state.detector = FraudDetector.load(path)
        logger.info("Loaded model from %s", path)
    else:
        app.state.detector = None
        logger.warning("Model file %s not found - /predict will return 503", path)
    yield


app = FastAPI(title="Fraud Detection API", version=__version__, lifespan=lifespan)


def _detector(request: Request) -> FraudDetector:
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return detector


def _score(detector: FraudDetector, rows: list[dict]) -> list[Prediction]:
    frame = pd.DataFrame(rows)[FEATURE_COLUMNS]
    proba = detector.predict_proba(frame)
    return [
        Prediction(
            fraud_probability=float(p),
            is_fraud=bool(p >= detector.threshold),
            threshold=detector.threshold,
        )
        for p in proba
    ]


@app.get("/health")
def health(request: Request) -> dict:
    detector = request.app.state.detector
    return {
        "status": "ok",
        "model_loaded": detector is not None,
        "threshold": detector.threshold if detector else None,
        "trained_at": detector.metadata.get("trained_at") if detector else None,
    }


@app.post("/predict", response_model=Prediction)
def predict(transaction: Transaction, request: Request) -> Prediction:  # type: ignore[valid-type]
    return _score(_detector(request), [transaction.model_dump()])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(transactions: list[Transaction], request: Request) -> list[Prediction]:  # type: ignore[valid-type]
    if not transactions:
        raise HTTPException(status_code=422, detail="Provide at least one transaction")
    if len(transactions) > 10_000:
        raise HTTPException(status_code=413, detail="Batch too large (max 10,000)")
    return _score(_detector(request), [t.model_dump() for t in transactions])
