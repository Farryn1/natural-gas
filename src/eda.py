"""Exploratory data analysis: overview plots, descriptive stats, stationarity tests."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss

from .data_loader import Dataset

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def descriptive_stats(price: pd.Series) -> pd.DataFrame:
    """Summary statistics for the price level, first difference, and log return."""
    log_ret = np.log(price).diff().dropna()
    rows = {
        "Spot price": price,
        "Log price": np.log(price),
        "First difference": price.diff().dropna(),
        "Log return": log_ret,
    }
    return pd.DataFrame({
        name: {"N": len(s), "Mean": s.mean(), "SD": s.std(),
               "Min": s.min(), "Median": s.median(), "Max": s.max()}
        for name, s in rows.items()
    }).T


def stationarity_tests(price: pd.Series) -> pd.DataFrame:
    """ADF (unit-root null) and KPSS (level-stationary null) on level and differences."""
    series = {"Price": price, "Log price": np.log(price),
              "First difference": price.diff().dropna(),
              "Log return": np.log(price).diff().dropna()}
    rows = []
    for name, s in series.items():
        adf_stat, adf_p = adfuller(s, autolag="AIC")[:2]
        kpss_stat, kpss_p = kpss(s, regression="c", nlags="auto")[:2]
        rows.append({"Series": name, "ADF stat": adf_stat, "ADF p": adf_p,
                     "KPSS stat": kpss_stat, "KPSS p": kpss_p})
    return pd.DataFrame(rows).set_index("Series")


def plot_overview(price: pd.Series, filename: str = "fig1_eda.png") -> Path:
    """Figure 1: price history, log returns, return distribution, monthly boxplot."""
    log_ret = np.log(price).diff().dropna()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].plot(price.index, price.values, lw=0.8, color="#1f4e79")
    for label, span in {"Uri 2021": ("2021-01-01", "2021-04-01"),
                        "RU war 2022": ("2022-06-01", "2022-12-01")}.items():
        axes[0, 0].axvspan(pd.Timestamp(span[0]), pd.Timestamp(span[1]),
                           alpha=0.2, color="crimson", label=label)
    axes[0, 0].set_title("Henry Hub spot price")
    axes[0, 0].set_ylabel("$/MMBtu"); axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(log_ret.index, log_ret.values, lw=0.6, color="#2e7d32")
    axes[0, 1].set_title("Log returns"); axes[0, 1].set_ylabel("log return")

    axes[1, 0].hist(log_ret.values, bins=80, color="#6a1b9a", alpha=0.8)
    axes[1, 0].set_title(f"Log-return distribution (kurtosis={log_ret.kurt():.1f})")
    axes[1, 0].set_xlabel("log return")

    by_month = pd.DataFrame({"price": price.values, "month": price.index.month})
    sns.boxplot(data=by_month, x="month", y="price", ax=axes[1, 1],
                color="#ef6c00", fliersize=1)
    axes[1, 1].set_title("Monthly price distribution")

    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=130); plt.close(fig)
    return out


def plot_correlation(dataset: Dataset, filename: str = "fig2_correlation.png") -> Path:
    """Figure 2: Pearson correlation heatmap of price, returns, and exogenous vars."""
    price = dataset.price.reindex(dataset.exog.index).ffill()
    df = pd.DataFrame({
        "price": price,
        "log_return": np.log(price).diff(),
        "storage": dataset.exog["storage"],
        "storage_change": dataset.exog["storage_change"],
        "wti": dataset.exog["wti"],
        "wti_return": dataset.exog["wti_return"],
        "hdd": dataset.exog["hdd"],
    }).dropna()
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation matrix (2010-2026)")
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=130); plt.close(fig)
    return out
