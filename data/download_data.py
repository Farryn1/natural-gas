"""Download the real Henry Hub datasets from public sources.

This is the production data path. It is all-or-nothing: if any source fails (no API
key, no network), nothing is written and the pipeline falls back to generate_data.py.

Sources:
  - WTI crude (weekly)      : FRED public CSV, no key required
  - Henry Hub daily/weekly  : EIA API v2, requires free EIA_API_KEY
  - Working gas storage     : EIA API v2, requires free EIA_API_KEY
  - Heating degree days     : NOAA CPC weekly archive (manual; see README)

Get a free EIA key at https://www.eia.gov/opendata/register.php and run:
    export EIA_API_KEY="your_key"
    python data/download_data.py
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
EIA_KEY = os.environ.get("EIA_API_KEY", "")
TIMEOUT = 30


def fetch_fred_wti() -> pd.DataFrame:
    """Weekly WTI spot price from FRED (public, no key)."""
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=WCOILWTICO"
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = ["date", "wti"]
    df["date"] = pd.to_datetime(df["date"])
    df["wti"] = pd.to_numeric(df["wti"], errors="coerce")
    return df.dropna()


def fetch_eia_series(series_id: str, value_col: str) -> pd.DataFrame:
    """Generic EIA API v2 series fetch."""
    if not EIA_KEY:
        raise RuntimeError("EIA_API_KEY not set")
    url = (f"https://api.eia.gov/v2/seriesid/{series_id}"
           f"?api_key={EIA_KEY}")
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()["response"]["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["period"])
    df[value_col] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", value_col]].dropna().sort_values("date")


def main() -> None:
    try:
        wti = fetch_fred_wti()
        hh_daily = fetch_eia_series("NG.RNGWHHD.D", "price")     # Henry Hub daily
        hh_weekly = fetch_eia_series("NG.RNGWHHD.W", "price")    # Henry Hub weekly
        storage = fetch_eia_series("NG.NW2_EPG0_SWO_R48_BCF.W", "storage")
    except Exception as exc:  # noqa: BLE001
        print(f"[download_data] real download failed: {exc}")
        print("[download_data] Falling back to synthetic data. "
              "Run: python data/generate_data.py")
        return

    print("[download_data] HDD must be obtained from NOAA CPC manually; see README.")
    print("[download_data] Writing real WTI / Henry Hub / storage CSVs.")
    wti.to_csv(HERE / "wti_weekly.csv", index=False)
    hh_daily.to_csv(HERE / "henry_hub_daily.csv", index=False)
    hh_weekly.to_csv(HERE / "henry_hub_weekly.csv", index=False)
    storage.to_csv(HERE / "storage_weekly.csv", index=False)


if __name__ == "__main__":
    main()
