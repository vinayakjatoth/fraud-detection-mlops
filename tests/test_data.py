import pytest

from fraud_detection.data import FEATURE_COLUMNS, split_data, validate_schema
from fraud_detection.synthetic import make_synthetic_transactions


def test_synthetic_schema_and_imbalance():
    df = make_synthetic_transactions(n_samples=5_000, fraud_rate=0.01, seed=1)
    assert list(df.columns) == [*FEATURE_COLUMNS, "Class"]
    assert df["Class"].isin([0, 1]).all()
    assert 0.005 < df["Class"].mean() < 0.02
    assert df["Time"].is_monotonic_increasing


def test_synthetic_is_deterministic():
    a = make_synthetic_transactions(2_000, seed=7)
    b = make_synthetic_transactions(2_000, seed=7)
    assert a.equals(b)


def test_split_is_stratified_and_disjoint(cfg):
    df = make_synthetic_transactions(10_000, fraud_rate=0.01, seed=3)
    s = split_data(df, cfg)
    assert len(s.X_train) + len(s.X_val) + len(s.X_test) == len(df)
    assert set(s.X_train.index).isdisjoint(s.X_test.index)
    assert set(s.X_val.index).isdisjoint(s.X_test.index)
    rates = [s.y_train.mean(), s.y_val.mean(), s.y_test.mean()]
    assert max(rates) - min(rates) < 0.003


def test_validate_schema_rejects_bad_data():
    df = make_synthetic_transactions(1_000, seed=1)
    with pytest.raises(ValueError):
        validate_schema(df.drop(columns=["V3"]), "Class")
    bad = df.copy()
    bad["Class"] = 0
    with pytest.raises(ValueError):
        validate_schema(bad, "Class")


def test_validate_schema_accepts_feature_missing_values_for_training_imputation():
    df = make_synthetic_transactions(1_000, seed=1)
    df.loc[df.index[0], "Amount"] = float("nan")
    validate_schema(df, "Class")
