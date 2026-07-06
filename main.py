"""End-to-end Henry Hub natural gas forecasting pipeline.

Runs EDA, fits all model families on weekly and daily frequencies under both the
fixed multi-step and rolling 1-step-ahead protocols, then writes a comparison table
and figures to outputs/.

Usage:
    python main.py            # full run
    python main.py --quick    # fewer rolling steps / epochs for a fast smoke test
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import benchmarks, classical, eda, garch
from src import ml_models, silverkite
from src.data_loader import load_daily, load_weekly, train_test_split

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def _fmt(results: dict) -> str:
    parts = [f"MAPE={results['mape'] * 100:5.2f}%", f"PM={results['pm']:.2f}"]
    if "coverage" in results:
        parts.append(f"cov={results['coverage']:.2f}")
    return "  ".join(parts)


def run_weekly(rows: list, forecasts: dict) -> None:
    """Fit and evaluate all weekly models."""
    print("\n" + "=" * 60 + "\nWEEKLY MODELS\n" + "=" * 60)
    ds = load_weekly()
    price = ds.price
    tr, te = train_test_split(price, ds.test_size)

    # exog-aligned price (2010+) for ARIMAX/SARIMAX
    price_x = price.reindex(ds.exog.index).ffill()
    tr_x, te_x = train_test_split(price_x, ds.test_size)
    ex_tr, ex_te = train_test_split(ds.exog, ds.test_size)

    # Benchmarks (fixed only)
    for name, res in benchmarks.run_benchmarks(tr, te).items():
        print(f"{name:22s} fixed   {_fmt(res)}")
        rows.append({"Model": name, "Freq": "W", "Protocol": "fixed", **res})

    O = classical.ORDERS["weekly"]
    weekly_specs = [
        ("ARIMA", O["ARIMA"], None, None, None),
        ("SARIMA", O["SARIMA"], None, None, None),
        ("ARIMAX", O["ARIMAX"], ds.exog, tr_x, te_x),
        ("SARIMAX", O["SARIMAX"], ds.exog, tr_x, te_x),
    ]
    for name, (order, sorder), exog, trn, tst in weekly_specs:
        if exog is None:
            fx = classical.fixed_forecast(tr, te, order, sorder)
            rl = classical.rolling_forecast(price, ds.test_size, order, sorder)
        else:
            fx = classical.fixed_forecast(trn, tst, order, sorder, ex_tr, ex_te)
            rl = classical.rolling_forecast(price_x, ds.test_size, order, sorder, exog)
        print(f"{name:22s} fixed   {_fmt(fx)}")
        print(f"{name:22s} rolling {_fmt(rl)}")
        rows.append({"Model": name, "Freq": "W", "Protocol": "fixed", **fx})
        rows.append({"Model": name, "Freq": "W", "Protocol": "rolling", **rl})
        forecasts[f"{name} (W)"] = rl

    # ARMA-GARCH
    go = garch.ARMA_ORDERS["weekly"]
    fx = garch.fixed_forecast(tr, te, go)
    rl = garch.rolling_forecast(price, ds.test_size, go)
    print(f"{'ARMA-GARCH':22s} fixed   {_fmt(fx)}")
    print(f"{'ARMA-GARCH':22s} rolling {_fmt(rl)}")
    rows.append({"Model": "ARMA-GARCH", "Freq": "W", "Protocol": "fixed", **fx})
    rows.append({"Model": "ARMA-GARCH", "Freq": "W", "Protocol": "rolling", **rl})
    forecasts["ARMA-GARCH (W)"] = rl

    # XGBoost
    fx = ml_models.run_xgboost(ds, rolling=False)
    rl = ml_models.run_xgboost(ds, rolling=True)
    print(f"{'XGBoost':22s} fixed   {_fmt(fx)}")
    print(f"{'XGBoost':22s} rolling {_fmt(rl)}")
    rows.append({"Model": "XGBoost", "Freq": "W", "Protocol": "fixed", **fx})
    rows.append({"Model": "XGBoost", "Freq": "W", "Protocol": "rolling", **rl})
    forecasts["XGBoost (W)"] = rl

    # LSTM (rolling == fixed; not retrained)
    ls = ml_models.run_lstm(ds)
    print(f"{'LSTM':22s} fixed   {_fmt(ls)}")
    rows.append({"Model": "LSTM", "Freq": "W", "Protocol": "fixed", **ls})
    rows.append({"Model": "LSTM", "Freq": "W", "Protocol": "rolling", **ls})
    forecasts["LSTM (W)"] = ls

    # Silverkite (optional)
    if silverkite.available():
        fx = silverkite.run_silverkite(ds, rolling=False)
        rl = silverkite.run_silverkite(ds, rolling=True)
        print(f"{'Silverkite':22s} fixed   {_fmt(fx)}")
        print(f"{'Silverkite':22s} rolling {_fmt(rl)}")
        rows.append({"Model": "Silverkite", "Freq": "W", "Protocol": "fixed", **fx})
        rows.append({"Model": "Silverkite", "Freq": "W", "Protocol": "rolling", **rl})
        forecasts["Silverkite (W)"] = rl
    else:
        print("Silverkite             SKIPPED (greykite not installed)")

    # XGBoost feature importance figure
    imp = ml_models.xgb_feature_importance(ds)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.head(15).iloc[::-1].plot.barh(ax=ax, color="#1f77b4")
    ax.set_title("XGBoost feature importance (top 15)")
    ax.set_xlabel("gain")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig_xgb_importance.png", dpi=130)
    plt.close(fig)


def run_daily(rows: list, forecasts: dict, quick: bool) -> None:
    """Fit and evaluate daily ARIMA / SARIMA / ARMA-GARCH."""
    print("\n" + "=" * 60 + "\nDAILY MODELS\n" + "=" * 60)
    ds = load_daily()
    price = ds.price
    test_size = 10 if quick else ds.test_size
    tr, te = train_test_split(price, test_size)

    O = classical.ORDERS["daily"]
    for name, (order, sorder) in [("ARIMA", O["ARIMA"]), ("SARIMA", O["SARIMA"])]:
        fx = classical.fixed_forecast(tr, te, order, sorder)
        rl = classical.rolling_forecast(price, test_size, order, sorder)
        print(f"{name:22s} fixed   {_fmt(fx)}")
        print(f"{name:22s} rolling {_fmt(rl)}")
        rows.append({"Model": name, "Freq": "D", "Protocol": "fixed", **fx})
        rows.append({"Model": name, "Freq": "D", "Protocol": "rolling", **rl})
        forecasts[f"{name} (D)"] = rl

    go = garch.ARMA_ORDERS["daily"]
    fx = garch.fixed_forecast(tr, te, go)
    rl = garch.rolling_forecast(price, test_size, go)
    print(f"{'ARMA-GARCH':22s} fixed   {_fmt(fx)}")
    print(f"{'ARMA-GARCH':22s} rolling {_fmt(rl)}")
    rows.append({"Model": "ARMA-GARCH", "Freq": "D", "Protocol": "fixed", **fx})
    rows.append({"Model": "ARMA-GARCH", "Freq": "D", "Protocol": "rolling", **rl})
    forecasts["ARMA-GARCH (D)"] = rl


def run_eda() -> None:
    """Generate EDA figures and print stationarity tests."""
    print("\n" + "=" * 60 + "\nEDA & STATIONARITY\n" + "=" * 60)
    ds = load_weekly()
    p1 = eda.plot_overview(ds.price)
    p2 = eda.plot_correlation(ds)
    print(f"saved {p1.name}, {p2.name}")
    print("\nDescriptive statistics:")
    print(eda.descriptive_stats(ds.price).round(3).to_string())
    print("\nStationarity tests (ADF unit-root null / KPSS level-stationary null):")
    print(eda.stationarity_tests(ds.price).round(4).to_string())


def make_comparison_figures(df: pd.DataFrame, forecasts: dict) -> None:
    """Figure 3 (fixed vs rolling MAPE) and Figure 6 (rolling forecasts vs actual)."""
    # Figure 3: fixed vs rolling MAPE bar chart
    pivot = df.pivot_table(index=["Model", "Freq"], columns="Protocol",
                           values="mape").reset_index()
    pivot["label"] = pivot["Model"] + " (" + pivot["Freq"] + ")"
    pivot = pivot.sort_values("rolling" if "rolling" in pivot else "fixed")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot))
    w = 0.4
    ax.bar(x - w / 2, pivot.get("fixed", 0) * 100, w, label="fixed", color="#bbbbbb")
    if "rolling" in pivot:
        ax.bar(x + w / 2, pivot["rolling"].fillna(0) * 100, w, label="rolling",
               color="#1f77b4")
    ax.axhline(3.49, ls="--", color="crimson", lw=1, label="naive floor (paper)")
    ax.set_xticks(x); ax.set_xticklabels(pivot["label"], rotation=45, ha="right")
    ax.set_ylabel("MAPE (%)"); ax.set_title("Fixed vs rolling MAPE by model")
    ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_mape_comparison.png", dpi=130)
    plt.close(fig)

    # Figure 6: rolling 1-step forecasts vs actual (weekly + daily)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, freq, title in [(axes[0], "(W)", "Weekly"), (axes[1], "(D)", "Daily")]:
        plotted = False
        for name, res in forecasts.items():
            if not name.endswith(freq) or "forecast" not in res:
                continue
            ax.plot(res["forecast"], marker="o", ms=3, lw=1, label=name)
            plotted = True
        if plotted:
            ax.set_title(f"{title}: rolling 1-step forecasts")
            ax.set_xlabel("test step"); ax.set_ylabel("price ($/MMBtu)")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig6_rolling_forecasts.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="fewer daily rolling steps for a fast smoke test")
    parser.add_argument("--skip-daily", action="store_true")
    args = parser.parse_args()

    rows: list[dict] = []
    forecasts: dict[str, dict] = {}

    run_eda()
    run_weekly(rows, forecasts)
    if not args.skip_daily:
        run_daily(rows, forecasts, quick=args.quick)

    df = pd.DataFrame(rows)
    make_comparison_figures(df, forecasts)

    # Final comparison table
    table = df.pivot_table(index=["Model", "Freq"], columns="Protocol",
                           values=["mape", "pm"])
    table = table.sort_index(axis=1)
    out_csv = OUTPUT_DIR / "results_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 60 + "\nFINAL COMPARISON (MAPE %)\n" + "=" * 60)
    summary = df.pivot_table(index=["Freq", "Model"], columns="Protocol",
                             values="mape") * 100
    print(summary.round(2).to_string())
    print(f"\nFull results written to {out_csv}")
    print(f"Figures written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
