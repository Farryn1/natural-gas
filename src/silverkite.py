"""Silverkite forecaster (LinkedIn Greykite). Optional — skipped if not installed.

Greykite has heavy, version-pinned dependencies. If it is unavailable the rest of
the pipeline runs without it; install with `pip install greykite` to enable.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data_loader import Dataset
from .metrics import evaluate

warnings.filterwarnings("ignore")

try:
    from greykite.framework.templates.autogen.forecast_config import (
        ForecastConfig, MetadataParam, ModelComponentsParam)
    from greykite.framework.templates.forecaster import Forecaster
    HAS_GREYKITE = True
except Exception:  # noqa: BLE001
    HAS_GREYKITE = False


def available() -> bool:
    """Whether greykite is importable in this environment."""
    return HAS_GREYKITE


def run_silverkite(dataset: Dataset, rolling: bool = False) -> dict | None:
    """Fit Silverkite with changepoint detection + Fourier seasonality. Weekly only."""
    if not HAS_GREYKITE:
        return None
    price = dataset.price
    ts = dataset.test_size
    df = pd.DataFrame({"ts": price.index, "y": price.values})

    def _forecast(train_df: pd.DataFrame, steps: int) -> np.ndarray:
        metadata = MetadataParam(time_col="ts", value_col="y", freq="W-FRI")
        components = ModelComponentsParam(
            changepoints={"changepoints_dict": {
                "method": "uniform", "n_changepoints": 25,
                "regularization_strength": 0.5}},
            seasonality={"yearly_seasonality": 8, "quarterly_seasonality": 4,
                         "monthly_seasonality": 2})
        forecaster = Forecaster()
        config = ForecastConfig(model_template="SILVERKITE",
                                forecast_horizon=steps,
                                coverage=0.95,
                                model_components_param=components,
                                metadata_param=metadata)
        result = forecaster.run_forecast_config(df=train_df, config=config)
        fc = result.forecast.df.tail(steps)
        return fc["forecast"].to_numpy()

    actuals = price.iloc[-ts:].to_numpy()
    if not rolling:
        preds = _forecast(df.iloc[:-ts], ts)
    else:
        preds = []
        for i in range(len(df) - ts, len(df)):
            preds.append(_forecast(df.iloc[:i], 1)[0])
        preds = np.array(preds)
    return {"forecast": preds, **evaluate(actuals, preds)}
