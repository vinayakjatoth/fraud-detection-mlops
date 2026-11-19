import pytest
from fastapi.testclient import TestClient

from fraud_detection.api import app
from fraud_detection.data import FEATURE_COLUMNS
from fraud_detection.synthetic import make_synthetic_transactions


@pytest.fixture()
def client(trained, monkeypatch):
    _, model_dir = trained
    monkeypatch.setenv("MODEL_PATH", str(model_dir / "fraud_model.joblib"))
    with TestClient(app) as c:
        yield c


def _rows(n):
    return make_synthetic_transactions(2_000, fraud_rate=0.05, seed=11)[FEATURE_COLUMNS].head(n)


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_single(client):
    payload = _rows(1).iloc[0].to_dict()
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud"], bool)


def test_predict_batch(client):
    payload = _rows(5).to_dict(orient="records")
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200 and len(r.json()) == 5


def test_validation_error_on_missing_field(client):
    payload = _rows(1).iloc[0].to_dict()
    payload.pop("V5")
    assert client.post("/predict", json=payload).status_code == 422


def test_503_when_model_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.joblib"))
    with TestClient(app) as c:
        assert c.get("/health").json()["model_loaded"] is False
        assert c.post("/predict", json=_rows(1).iloc[0].to_dict()).status_code == 503
