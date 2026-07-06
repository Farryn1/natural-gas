"""Two-step ARMA-GARCH(1,1) with Student-t innovations.

Step 1: fit ARMA(p,q) on log returns for the conditional mean.
Step 2: fit GARCH(1,1) with Student-t innovations on the ARMA residuals.
Price-level forecasts: exponentiate (last log price + ARMA mean return);
prediction intervals widened by the GARCH conditional std.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .metrics import evaluate

warnings.filterwarnings("ignore")

# Paper-selected ARMA orders on log returns
ARMA_ORDERS = {"weekly": (1, 0, 2), "daily": (3, 0, 3)}
_RESCALE = 100.0  # arch prefers returns on a 1-1000 scale


def _one_step(log_price: pd.Series, order) -> tuple[float, float]:
    """Fit ARMA+GARCH on the history, return (mean_log_return, sigma) for next step."""
    log_ret = log_price.diff().dropna() * _RESCALE
    arma = SARIMAX(log_ret, order=order, trend="c",
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)
    mean_ret = float(arma.get_forecast(1).predicted_mean.iloc[0])
    resid = pd.Series(arma.resid).dropna()
    garch = arch_model(resid, mean="Zero", vol="GARCH", p=1, q=1,
                       dist="t").fit(disp="off")
    sigma = float(np.sqrt(garch.forecast(horizon=1, reindex=False)
                          .variance.iloc[-1, 0]))
    return mean_ret / _RESCALE, sigma / _RESCALE


def fixed_forecast(train: pd.Series, test: pd.Series, order) -> dict:
    """Fixed multi-step: iterate the one-step mean forward from the last train price."""
    h = len(test)
    log_price = np.log(train)
    mean_ret, sigma = _one_step(log_price, order)
    last = log_price.iloc[-1]
    preds, lowers, uppers = [], [], []
    for step in range(1, h + 1):
        lp = last + mean_ret * step
        band = 1.96 * sigma * np.sqrt(step)
        preds.append(np.exp(lp))
        lowers.append(np.exp(lp - band))
        uppers.append(np.exp(lp + band))
    preds = np.array(preds); lowers = np.array(lowers); uppers = np.array(uppers)
    return {"forecast": preds, "lower": lowers, "upper": uppers,
            **evaluate(test.to_numpy(), preds, lowers, uppers)}


def rolling_forecast(series: pd.Series, test_size: int, order) -> dict:
    """Rolling 1-step-ahead: refit ARMA-GARCH at every test date."""
    n = len(series)
    preds, lowers, uppers, actuals = [], [], [], []
    for i in range(n - test_size, n):
        log_price = np.log(series.iloc[:i])
        mean_ret, sigma = _one_step(log_price, order)
        last = log_price.iloc[-1]
        lp = last + mean_ret
        band = 1.96 * sigma
        preds.append(np.exp(lp))
        lowers.append(np.exp(lp - band))
        uppers.append(np.exp(lp + band))
        actuals.append(series.iloc[i])
    preds = np.array(preds); lowers = np.array(lowers)
    uppers = np.array(uppers); actuals = np.array(actuals)
    return {"forecast": preds, "lower": lowers, "upper": uppers,
            **evaluate(actuals, preds, lowers, uppers)}
