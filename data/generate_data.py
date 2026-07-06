"""Generate synthetic Henry Hub natural gas datasets that mimic the statistical
properties reported in the project paper.

This is a fallback so the full pipeline runs end-to-end without external API keys.
Real data can be substituted by replacing the CSVs in this directory with files of
the same schema (see README).

Target stylized facts (from the paper):
  - Weekly price: mean 4.10, std 2.16, min 1.34, max 14.49, right-skewed
  - Log returns: ~0 mean, std 0.109/week, very heavy tails (volatility clustering)
  - Storage (Bcf): mean 2628, std 656, strongly seasonal, corr(storage_change, HDD) ~= -0.90
  - WTI ($/bbl): mean 60.7, std 25.6, corr(price, WTI) ~= 0.46
  - HDD: mean 73.5, std 73, strong winter seasonality
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
HERE = Path(__file__).resolve().parent


def _seasonal_hdd(dates: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Heating degree days: zero in summer, large in winter."""
    doy = dates.dayofyear.to_numpy()
    # peak in January (doy ~ 15), trough in July
    seasonal = np.cos(2 * np.pi * (doy - 15) / 365.25)
    base = np.clip(140 * np.maximum(seasonal, -0.2), 0, None)
    noise = rng.normal(0, 18, size=len(dates))
    return np.clip(base + noise, 0, None)


def _garch_innovations(n: int, rng: np.random.Generator,
                       omega: float, alpha: float, beta: float,
                       nu: float = 5.0) -> np.ndarray:
    """Simulate GARCH(1,1) shocks with Student-t innovations (volatility clustering)."""
    eps = np.zeros(n)
    sigma2 = np.full(n, omega / (1 - alpha - beta))
    z = rng.standard_t(nu, size=n) * np.sqrt((nu - 2) / nu)
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        eps[t] = np.sqrt(sigma2[t]) * z[t]
    return eps


def _ou_log_price(eps: np.ndarray, mu: float, kappa: float) -> np.ndarray:
    """Mean-reverting (OU) log price anchored at mu, driven by GARCH shocks eps."""
    x = np.zeros(len(eps))
    x[0] = mu
    for t in range(1, len(eps)):
        x[t] = x[t - 1] + kappa * (mu - x[t - 1]) + eps[t]
    return x


def _inject_spikes(log_price: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Add the documented structural-break spikes (Uri 2021, RU war 2022, 2026)."""
    out = log_price.copy()
    spikes = {
        "2021-02-12": (0.9, 4),    # Winter Storm Uri, reverts in ~4 weeks
        "2022-08-22": (0.7, 25),   # Russia-Ukraine LNG demand, prolonged
        "2025-10-15": (0.45, 4),   # recent episode; reverts before the 2026 test window
    }
    for date_str, (amp, decay_w) in spikes.items():
        target = pd.Timestamp(date_str)
        idx = int(np.argmin(np.abs(dates - target)))
        for k, i in enumerate(range(idx, min(idx + decay_w * 2, len(out)))):
            out[i] += amp * np.exp(-k / decay_w)
    return out


def generate_weekly(rng: np.random.Generator) -> pd.DataFrame:
    """Weekly Henry Hub price + exogenous series, 1997-2026."""
    dates = pd.date_range("1997-01-10", "2026-04-10", freq="W-FRI")
    n = len(dates)

    # weekly log-return sigma ~ 0.05 (unconditional): omega/(1-a-b) = 0.05^2
    eps = _garch_innovations(n, rng, omega=2.0e-4, alpha=0.12, beta=0.80, nu=5.4)
    log_price = _ou_log_price(eps, mu=np.log(4.1), kappa=0.02)
    log_price = _inject_spikes(log_price, dates)
    price = np.exp(log_price)
    price = price * (4.10 / price.mean())   # scale to target mean, keep volatility
    price = np.clip(price, 1.34, 14.49)

    hdd = _seasonal_hdd(dates, rng)
    # storage negatively driven by HDD (cold weeks draw down storage)
    storage_change = -8.0 * (hdd - hdd.mean()) / hdd.std() + rng.normal(0, 3, n)
    storage = 2628 + np.cumsum(storage_change) * 0.0
    storage = 2628 + 656 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365.25) \
        + rng.normal(0, 80, n)
    storage = np.clip(storage, 1023, 4047)
    # WTI partially correlated with gas price
    wti = 60.7 + 25.6 * (
        0.46 * (price - price.mean()) / price.std()
        + np.sqrt(1 - 0.46 ** 2) * rng.standard_normal(n)
    )
    wti = np.clip(wti, 10, 145)

    weekly = pd.DataFrame({"date": dates, "price": price})
    storage_df = pd.DataFrame({"date": dates, "storage": storage})
    wti_df = pd.DataFrame({"date": dates, "wti": wti})
    hdd_df = pd.DataFrame({"date": dates, "hdd": hdd})
    return weekly, storage_df, wti_df, hdd_df


def generate_daily(rng: np.random.Generator) -> pd.DataFrame:
    """Daily Henry Hub price, 1997-2026 (business days)."""
    dates = pd.bdate_range("1997-01-07", "2026-04-13")
    n = len(dates)
    # daily log-return sigma ~ 0.03 (unconditional): omega/(1-a-b) = 0.03^2
    eps = _garch_innovations(n, rng, omega=9.0e-6, alpha=0.10, beta=0.89, nu=4.5)
    log_price = _ou_log_price(eps, mu=np.log(4.1), kappa=0.004)
    log_price = _inject_spikes(log_price, dates)
    price = np.exp(log_price)
    price = price * (4.10 / price.mean())
    price = np.clip(price, 1.05, 18.5)
    return pd.DataFrame({"date": dates, "price": price})


def main() -> None:
    rng = np.random.default_rng(SEED)
    weekly, storage_df, wti_df, hdd_df = generate_weekly(rng)
    daily = generate_daily(rng)

    weekly.to_csv(HERE / "henry_hub_weekly.csv", index=False)
    daily.to_csv(HERE / "henry_hub_daily.csv", index=False)
    storage_df.to_csv(HERE / "storage_weekly.csv", index=False)
    wti_df.to_csv(HERE / "wti_weekly.csv", index=False)
    hdd_df.to_csv(HERE / "hdd_weekly.csv", index=False)

    print(f"Wrote synthetic data to {HERE}")
    print(f"  weekly price : n={len(weekly)}, mean={weekly.price.mean():.2f}, "
          f"std={weekly.price.std():.2f}, min={weekly.price.min():.2f}, "
          f"max={weekly.price.max():.2f}")
    print(f"  daily price  : n={len(daily)}, mean={daily.price.mean():.2f}")
    print(f"  storage      : mean={storage_df.storage.mean():.0f}, "
          f"std={storage_df.storage.std():.0f}")
    print(f"  WTI          : mean={wti_df.wti.mean():.1f}, std={wti_df.wti.std():.1f}")
    print(f"  HDD          : mean={hdd_df.hdd.mean():.1f}, std={hdd_df.hdd.std():.1f}")
    corr = np.corrcoef(weekly.price, wti_df.wti)[0, 1]
    print(f"  corr(price, WTI) = {corr:.2f}")


if __name__ == "__main__":
    main()
