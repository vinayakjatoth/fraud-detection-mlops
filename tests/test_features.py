import numpy as np

from fraud_detection.data import split_data
from fraud_detection.features import FraudFeatureEngineer
from fraud_detection.synthetic import make_synthetic_transactions


def _fit(cfg):
    df = make_synthetic_transactions(8_000, fraud_rate=0.02, seed=5)
    s = split_data(df, cfg)
    fe = FraudFeatureEngineer().fit(s.X_train, s.y_train)
    return fe, s


def test_engineered_columns_present_and_finite(cfg):
    fe, s = _fit(cfg)
    out = fe.transform(s.X_test)
    for col in [
        "hour_sin",
        "hour_cos",
        "is_night",
        "log_amount",
        "amount_zscore",
        "is_micro_amount",
        "is_round_amount",
        "v_l2_norm",
        "signal_gap",
    ]:
        assert col in out.columns
    assert "Time" not in out.columns  # raw elapsed-seconds is replaced by hour features
    assert np.isfinite(out.to_numpy(dtype=float)).all()
    assert list(out.columns) == fe.feature_names_


def test_flags_behave_as_documented(cfg):
    fe, s = _fit(cfg)
    row = s.X_test.iloc[:3].copy()
    row["Amount"] = [1.5, 100.0, 57.25]
    row["Time"] = [3 * 3600, 14 * 3600, 26 * 3600]  # 03:00, 14:00, 02:00 next day
    out = fe.transform(row)
    assert out["is_micro_amount"].tolist() == [1, 0, 0]
    assert out["is_round_amount"].tolist() == [0, 1, 0]
    assert out["is_night"].tolist() == [1, 0, 1]


def test_fit_uses_training_statistics_only(cfg):
    fe, s = _fit(cfg)
    before = (fe.log_amount_mean_, list(fe.negative_signal_cols_))
    fe.transform(s.X_test)  # transforming other data must not change learned state
    assert before == (fe.log_amount_mean_, list(fe.negative_signal_cols_))


def test_missing_features_are_imputed_from_training_medians(cfg):
    df = make_synthetic_transactions(8_000, fraud_rate=0.02, seed=5)
    s = split_data(df, cfg)
    fe = FraudFeatureEngineer().fit(s.X_train, s.y_train)
    row = s.X_test.iloc[:1].copy()
    row["Amount"] = np.nan
    out = fe.transform(row)
    assert np.isfinite(out.to_numpy(dtype=float)).all()
