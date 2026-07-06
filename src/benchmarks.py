"""Benchmark forecasts: naive (random walk), drift, and seasonal naive."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import evaluate


def naive_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Random walk: repeat the last observed value."""
    return np.full(h, train.iloc[-1])


def drift_forecast(train: pd.Series, h: int) -> np.ndarray:
    """Extrapolate the average linear trend over the training set."""
    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    return train.iloc[-1] + slope * np.arange(1, h + 1)


def seasonal_naive_forecast(train: pd.Series, h: int, season: int = 52) -> np.ndarray:
    """Use the value `season` periods ago."""
    return train.iloc[-season:-season + h].to_numpy() if season > h \
        else train.iloc[-season:].to_numpy()[:h]


def run_benchmarks(train: pd.Series, test: pd.Series,
                   season: int = 52) -> dict[str, dict]:
    """Evaluate all three benchmarks on the test window (fixed protocol)."""
    h = len(test)
    y = test.to_numpy()
    results = {}
    for name, fc in [
        ("Naive", naive_forecast(train, h)),
        ("Drift", drift_forecast(train, h)),
        ("Seasonal Naive", seasonal_naive_forecast(train, h, season)),
    ]:
        m = evaluate(y, fc)
        results[name] = {"forecast": fc, **m}
    return results
