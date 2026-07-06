"""Classical time-series models: ARIMA, SARIMA, ARIMAX, SARIMAX.

Orders are fixed to the AICc-selected values reported in the paper so the pipeline
runs quickly; set `grid_search=True` in fit_best_order to reproduce the selection.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .metrics import evaluate

warnings.filterwarnings("ignore")

# Paper-selected orders
ORDERS = {
    "weekly": {
        "ARIMA": ((2, 0, 2), (0, 0, 0, 0)),
        "SARIMA": ((1, 1, 1), (0, 0, 1, 52)),
        "ARIMAX": ((0, 1, 2), (0, 0, 0, 0)),
        "SARIMAX": ((1, 0, 2), (0, 1, 1, 52)),
    },
    "daily": {
        "ARIMA": ((4, 0, 4), (0, 0, 0, 0)),
        "SARIMA": ((1, 1, 1), (0, 0, 1, 5)),
    },
}


def _fit_predict(endog: pd.Series, order, seasonal_order, h: int,
                 exog=None, exog_future=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a single SARIMAX model and return (mean, lower, upper) forecasts."""
    model = SARIMAX(endog, order=order, seasonal_order=seasonal_order,
                    exog=exog, enforce_stationarity=False,
                    enforce_invertibility=False)
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=h, exog=exog_future)
    mean = fc.predicted_mean.to_numpy()
    ci = fc.conf_int(alpha=0.05)
    return mean, ci.iloc[:, 0].to_numpy(), ci.iloc[:, 1].to_numpy()


def fixed_forecast(train: pd.Series, test: pd.Series, order, seasonal_order,
                   exog_train=None, exog_test=None) -> dict:
    """Fixed multi-step: fit once on train, forecast the entire test window."""
    h = len(test)
    mean, lo, hi = _fit_predict(train, order, seasonal_order, h,
                                exog=exog_train, exog_future=exog_test)
    return {"forecast": mean, "lower": lo, "upper": hi,
            **evaluate(test.to_numpy(), mean, lo, hi)}


def rolling_forecast(series: pd.Series, test_size: int, order, seasonal_order,
                     exog=None) -> dict:
    """Rolling 1-step-ahead: refit at each step on all data strictly before it."""
    n = len(series)
    preds, lowers, uppers, actuals = [], [], [], []
    for i in range(n - test_size, n):
        endog = series.iloc[:i]
        ex_tr = exog.iloc[:i] if exog is not None else None
        ex_fut = exog.iloc[i:i + 1] if exog is not None else None
        mean, lo, hi = _fit_predict(endog, order, seasonal_order, 1,
                                    exog=ex_tr, exog_future=ex_fut)
        preds.append(mean[0]); lowers.append(lo[0]); uppers.append(hi[0])
        actuals.append(series.iloc[i])
    preds = np.array(preds); lowers = np.array(lowers)
    uppers = np.array(uppers); actuals = np.array(actuals)
    return {"forecast": preds, "lower": lowers, "upper": uppers,
            **evaluate(actuals, preds, lowers, uppers)}
