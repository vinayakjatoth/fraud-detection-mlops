import pytest

from fraud_detection.config import load_config
from fraud_detection.train import train_model


@pytest.fixture(scope="session")
def cfg():
    cfg = load_config("configs/config.yaml")
    cfg["model"]["xgboost"]["n_estimators"] = 60  # keep tests fast
    cfg["model"]["xgboost"]["early_stopping_rounds"] = 10
    return cfg


@pytest.fixture(scope="session")
def trained(cfg, tmp_path_factory):
    """Train once on small synthetic data and reuse across tests."""
    model_dir = tmp_path_factory.mktemp("models")
    detector = train_model(cfg, synthetic=True, n_synthetic=12_000, model_dir=model_dir)
    return detector, model_dir
