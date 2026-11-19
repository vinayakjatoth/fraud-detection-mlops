"""Synthetic transactions that mimic the schema of the public credit-card fraud dataset.

Used for unit tests, CI smoke runs and quick demos when the real 284K-row dataset is not
available. It is NOT a substitute for the real data: never report its metrics as results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

V_COLUMNS = [f"V{i}" for i in range(1, 29)]

# zero-based index -> mean shift for fraud (loosely mirrors the real dataset, where e.g.
# V4 / V11 trend positive and V10 / V12 / V14 / V16 / V17 trend negative for fraud).
_FRAUD_SHIFTS = {3: 2.0, 10: 2.0, 9: -2.5, 11: -3.0, 13: -3.5, 15: -2.0, 16: -2.5}


def make_synthetic_transactions(
    n_samples: int = 50_000, fraud_rate: float = 0.005, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n_samples * fraud_rate)), 20)

    y = np.zeros(n_samples, dtype=int)
    y[rng.choice(n_samples, size=n_fraud, replace=False)] = 1
    fraud = y == 1

    # PCA-like anonymised components; fraud rows get shifted, with varying strength so that a
    # share of them is "hard" (weak signal) and the task is not trivially separable.
    v = rng.normal(0.0, 1.0, size=(n_samples, 28))
    for idx, shift in _FRAUD_SHIFTS.items():
        v[fraud, idx] += shift * rng.uniform(0.2, 1.2, size=n_fraud)

    # Time: seconds since first transaction over two days. Legit traffic follows a daily cycle,
    # fraud is flatter with extra weight in the small hours.
    hours = np.arange(24)
    p_legit = 1.0 + 0.8 * np.sin(2 * np.pi * (hours - 9) / 24)
    p_fraud = 1.0 + 1.5 * (hours < 6)
    day = rng.integers(0, 2, size=n_samples)
    hour = np.where(
        fraud,
        rng.choice(hours, size=n_samples, p=p_fraud / p_fraud.sum()),
        rng.choice(hours, size=n_samples, p=p_legit / p_legit.sum()),
    )
    time = day * 86_400 + hour * 3_600 + rng.uniform(0, 3_600, size=n_samples)
    time = np.minimum(time, 172_792.0)

    # Amount: legit is log-normal; fraud mixes micro "card testing" amounts with larger ones,
    # and a slice of fraud uses suspiciously round values.
    amount = np.exp(rng.normal(3.2, 1.4, size=n_samples))
    micro = fraud & (rng.random(n_samples) < 0.4)
    big = fraud & ~micro
    amount[micro] = rng.uniform(0.5, 2.0, size=micro.sum())
    amount[big] = np.exp(rng.normal(4.2, 1.5, size=big.sum()))
    round_fraud = big & (rng.random(n_samples) < 0.2)
    amount[round_fraud] = np.round(amount[round_fraud])
    amount = np.round(np.clip(amount, 0.0, 25_000.0), 2)

    df = pd.DataFrame(v, columns=V_COLUMNS)
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = y
    return df.sort_values("Time").reset_index(drop=True)
