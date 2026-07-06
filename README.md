# Henry Hub Natural Gas Price Forecasting

A comprehensive forecasting benchmark comparing nine model families on Henry Hub natural gas spot prices, spanning classical time series (ARIMA/SARIMA, ARMA–GARCH), exogenous-regressor models (ARIMAX/SARIMAX), and machine learning (XGBoost, LSTM, Silverkite). Both fixed multi-step and rolling 1-step-ahead evaluation protocols are implemented.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Protocol dominates model choice** | Rolling 1-step-ahead cuts MAPE ~4× vs. fixed multi-step across every model |
| **Best daily model** | Rolling ARMA(3,0,3)–GARCH(1,1) Student-t: **MAPE 3.36%** (beats naïve 3.49%) |
| **Best weekly model** | XGBoost with lag + exogenous features: **MAPE 4.70%** rolling |
| **LSTM underperforms** | Only ~800 training obs after inner-join; sample-starved vs. model size |
| **Silverkite**: | Changepoint regularization too conservative for this series |

---

## Data

All data is publicly available. Download scripts provided in `data/download_data.py`.

| Series | Source | Frequency | Coverage | Obs. |
|--------|--------|-----------|----------|------|
| Henry Hub spot price | EIA | Daily | 1997–2026 | 7,348 |
| Henry Hub spot price | EIA | Weekly | 1997–2026 | 1,526 |
| U.S. Working Gas Storage | EIA | Weekly | 2010–2026 | 849 |
| WTI crude oil | FRED | Weekly | 1986–2026 | 2,154 |
| Heating Degree Days (HDD) | NOAA CPC | Weekly | 1997–2026 | 1,480 |

> Exogenous regressors are available from 2010. Models using exogenous variables (ARIMAX, SARIMAX, XGBoost, LSTM, Silverkite) are restricted to the 2010–2026 inner-joined window.

### Download Instructions

```bash
# Option 1: EIA API (requires free API key from eia.gov)
export EIA_API_KEY="your_key_here"
python data/download_data.py

# Option 2: Manual download
# Henry Hub daily: https://www.eia.gov/dnav/ng/hist/rngwhhdd.htm
# Henry Hub weekly: https://www.eia.gov/dnav/ng/hist/rngwhhdw.htm
# Storage: https://www.eia.gov/dnav/ng/hist/ngt_epg0_sws_dcu_bcfw.htm
# WTI (FRED): https://fred.stlouisfed.org/series/DCOILWTICO
# HDD: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/cdus/degree_days/
```

---

## Models

### Benchmarks
- **Naïve** (random walk): repeat last observed price
- **Drift**: extrapolate linear trend over training set
- **Seasonal Naïve** (S=52): use value from 52 weeks ago

### Classical Time Series (`statsmodels`)
- **ARIMA**: (2,0,2) weekly · (4,0,4) daily
- **SARIMA**: (1,1,1)×(0,0,1)₅₂ weekly · (1,1,1)×(0,0,1)₅ daily
- **ARIMAX**: (0,1,2) with WTI, storage, HDD regressors (lagged 1 week)
- **SARIMAX**: (1,0,2)×(0,1,1)₅₂ with same regressors

### ARMA–GARCH (`arch`)
Two-step approach on log returns:
1. Fit ARMA(1,0,2) weekly / ARMA(3,0,3) daily for the conditional mean
2. Fit GARCH(1,1) with Student-*t* innovations on ARMA residuals
3. Forecast price level by exponentiating ARMA mean + last log price; PI widened by GARCH σ_{t+1|t}

### XGBoost
**Feature matrix (23 dims)**:
- Price lags: {1, 2, 3, 4, 8, 13, 52} weeks
- Log-return lags: same set
- Lag-1: storage level, storage change, WTI price, WTI return, HDD
- Cyclic calendar: sin/cos of month, sin/cos of ISO week

Config: 200 trees · max_depth=3 · lr=0.05 · subsample=0.8 · colsample_bytree=0.8

### LSTM (PyTorch)
Architecture: 2 × LSTM(64, dropout=0.2) → FF 64→32→1  
Input: same 23-dim features, 12-week sliding sequences  
Training: Adam (lr=1e-3), MSE loss, gradient clip norm=1.0, early stopping patience=10

### Silverkite (LinkedIn Greykite)
- Changepoint detection: uniform, 25 candidates, regularization λ=0.5
- Fourier seasonality: annual order 8, quarterly 4, monthly 2
- Ridge-penalized exogenous: lag-1 WTI, storage, HDD

---

## Results

**Table: Forecast MAPE on held-out test window**

| Model | Fixed MAPE | Rolling MAPE |
|-------|:----------:|:------------:|
| Naïve | 3.49% | 3.49% |
| Drift | 3.46% | — |
| Seasonal Naïve | 34.67% | — |
| ARIMA(2,0,2) W | 29.87% | 9.52% |
| SARIMA W | 26.55% | 9.12% |
| ARIMAX W | 40.82% | 13.54% |
| SARIMAX W | 26.23% | 11.50% |
| **ARMA–GARCH W** | 15.54% | 5.15% |
| **XGBoost W** | **5.29%** | **4.70%** |
| LSTM W | 12.90% | 12.90% |
| Silverkite W | 36.46% | 23.78% |
| ARIMA(4,0,4) D | 18.36% | 3.70% |
| SARIMA D | 26.79% | 4.00% |
| **ARMA–GARCH D** | 10.39% | **3.36%** |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download data
python data/download_data.py     # or place CSVs manually in data/

# 3. Run the notebook top-to-bottom
jupyter notebook henry_hub_forecasting.ipynb
```

Outputs (figures + results CSV) are written to `outputs/` automatically.

---

## Project Structure

```
henry-hub-forecasting/
├── data/
│   ├── download_data.py          # Data collection script
│   ├── henry_hub_daily.csv
│   ├── henry_hub_weekly.csv
│   ├── storage_weekly.csv
│   ├── wti_weekly.csv
│   └── hdd_weekly.csv
├── outputs/                      # Auto-generated figures & tables
│   ├── fig1_eda.png
│   ├── fig2_correlation.png
│   ├── fig3_mape_comparison.png
│   ├── fig6_rolling_forecasts.png
│   └── fig_xgb_importance.png
├── henry_hub_forecasting.ipynb   # Main notebook
├── requirements.txt
└── README.md
```

---

## References

- Box, Jenkins, Reinsel & Ljung (2015). *Time Series Analysis: Forecasting and Control*, 5th ed.
- Bollerslev (1986). Generalized ARCH. *Journal of Econometrics*.
- Chen & Guestrin (2016). XGBoost. *KDD '16*.
- Hosseini et al. (2022). Greykite. *KDD '22*.
- Hyndman & Athanasopoulos (2021). *Forecasting: Principles and Practice*, 3rd ed.
- Tsay (2010). *Analysis of Financial Time Series*, 3rd ed.
