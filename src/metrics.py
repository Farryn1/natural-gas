"""Forecast evaluation metrics: MAPE, PM, and prediction-interval coverage."""
from __future__ import annotations

import numpy as np


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (scale-free, in [0, inf))."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))


def pm(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Prediction measure: SSE(forecast) / SSE(in-sample mean). PM>1 worse than mean."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(np.sum((y_true - y_pred) ** 2) / denom)


def coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of observations falling inside the [lower, upper] interval."""
    y_true = np.asarray(y_true, dtype=float)
    inside = (y_true >= np.asarray(lower)) & (y_true <= np.asarray(upper))
    return float(np.mean(inside))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             lower: np.ndarray | None = None,
             upper: np.ndarray | None = None) -> dict[str, float]:
    """Bundle MAPE, PM, and (optionally) interval coverage into one dict."""
    out = {"mape": mape(y_true, y_pred), "pm": pm(y_true, y_pred)}
    if lower is not None and upper is not None:
        out["coverage"] = coverage(y_true, lower, upper)
    return out
