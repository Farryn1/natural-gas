"""Load and assemble the Henry Hub datasets, build features, define train/test splits."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Held-out test windows (from the paper)
TEST_WEEKS = 6     # weekly: 2026-03-06 to 2026-04-10
TEST_DAYS = 30     # daily:  2026-03-02 to 2026-04-13
EXOG_START = "2010-01-01"  # exogenous storage series begins in 2010

PRICE_LAGS = [1, 2, 3, 4, 8, 13, 52]


@dataclass
class Dataset:
    """Container for a single-frequency price series + optional exogenous frame."""
    freq: str                      # "weekly" or "daily"
    price: pd.Series               # indexed by date
    exog: pd.DataFrame | None      # standardized exogenous regressors, aligned to price
    test_size: int


def _read_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / name, parse_dates=["date"])
    return df.set_index("date").sort_index()


def ensure_data() -> None:
    """Generate synthetic data if the CSVs are missing."""
    if not (DATA_DIR / "henry_hub_weekly.csv").exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, str(DATA_DIR / "generate_data.py")], check=True)


def load_weekly() -> Dataset:
    """Weekly price + standardized lag-1 exogenous (WTI, storage, HDD) from 2010."""
    ensure_data()
    price = _read_csv("henry_hub_weekly.csv")["price"]
    storage = _read_csv("storage_weekly.csv")["storage"]
    wti = _read_csv("wti_weekly.csv")["wti"]
    hdd = _read_csv("hdd_weekly.csv")["hdd"]

    exog = pd.concat([wti, storage, hdd], axis=1)
    exog.columns = ["wti", "storage", "hdd"]
    exog["storage_change"] = exog["storage"].diff()
    exog["wti_return"] = np.log(exog["wti"]).diff()
    exog = exog.loc[EXOG_START:].dropna()
    # standardize
    exog_std = (exog - exog.mean()) / exog.std()
    return Dataset("weekly", price, exog_std, TEST_WEEKS)


def load_daily() -> Dataset:
    """Daily price, univariate (exogenous storage is weekly-only)."""
    ensure_data()
    price = _read_csv("henry_hub_daily.csv")["price"]
    return Dataset("daily", price, None, TEST_DAYS)


def train_test_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    """Time-ordered split: last `test_size` obs are the held-out test window."""
    return series.iloc[:-test_size], series.iloc[-test_size:]


def make_features(price: pd.Series, exog: pd.DataFrame | None) -> pd.DataFrame:
    """Build the 23-dim feature matrix used by XGBoost and LSTM.

    Features: price lags, log-return lags, lag-1 exogenous, cyclic calendar.
    """
    df = pd.DataFrame(index=price.index)
    df["price"] = price
    log_ret = np.log(price).diff()
    for lag in PRICE_LAGS:
        df[f"price_lag{lag}"] = price.shift(lag)
        df[f"logret_lag{lag}"] = log_ret.shift(lag)
    if exog is not None:
        ex = exog.reindex(price.index).ffill()
        for col in ["storage", "storage_change", "wti", "wti_return", "hdd"]:
            df[f"{col}_lag1"] = ex[col].shift(1)
    # cyclic calendar features
    month = df.index.month
    iso_week = df.index.isocalendar().week.to_numpy()
    df["sin_month"] = np.sin(2 * np.pi * month / 12)
    df["cos_month"] = np.cos(2 * np.pi * month / 12)
    df["sin_week"] = np.sin(2 * np.pi * iso_week / 52)
    df["cos_week"] = np.cos(2 * np.pi * iso_week / 52)
    return df.dropna()
