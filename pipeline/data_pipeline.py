import yfinance as yf
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

# CONFIG
SYMBOL = "AAPL"
INTERVAL = "1m"
PERIOD = "1d"
ROLLING_WINDOW = 14

console = Console()

# Raw Market Data
def fetch_raw_market_data(symbol, interval, period):
    """
    Fetches raw OHLCV data from the API and normalizes column names.
    Handles MultiIndex columns safely.
    """
    data = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False
    )

    # Reset index so timestamp becomes a column
    data = data.reset_index()

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Normalize column names
    data.columns = [
        str(c).lower().replace(" ", "_")
        for c in data.columns
    ]

    return data


# Derived Market Context
def generate_derived_features(df, window):
    """
    Generates rolling statistics, volatility, returns,
    and basic market context variables.
    """

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["simple_return"] = df["close"].pct_change()

    df["rolling_mean"] = df["close"].rolling(window).mean()
    df["rolling_std"] = df["close"].rolling(window).std()
    df["rolling_zscore"] = (
        (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    )

    df["rolling_volatility"] = df["log_return"].rolling(window).std()

    df["price_range"] = df["high"] - df["low"]
    df["rolling_range"] = df["price_range"].rolling(window).mean()

    df["volume_mean"] = df["volume"].rolling(window).mean()
    df["volume_zscore"] = (
        (df["volume"] - df["volume_mean"]) / df["volume"].rolling(window).std()
    )

    df["trend_strength"] = (
        df["rolling_mean"].diff()
    )

    return df

# Database Initialisation
def persist_dataset(df):
    """
    This function represents your storage layer.
    Right now it simply returns the DataFrame,
    but this is where SQLite, DuckDB, or Parquet fits later.
    """
    return df.dropna().reset_index(drop=True)


# TUI Rendering
def render_dataset_table(df, lookback=3):
    """
    Renders a narrow-terminal-friendly vertical table.
    Rows = variables
    Columns = recent timesteps
    """

    recent = df.tail(lookback)

    table = Table(title=f"Market Snapshot | {SYMBOL}")

    # First column: variable names
    table.add_column("Variable", justify="left", no_wrap=True)

    # Add one column per recent timestep
    for ts in recent["datetime"]:
        table.add_column(str(ts.time()), justify="right", no_wrap=True)

    # Select variables to display (you control density here)
    display_vars = [
        "close",
        "log_return",
        "rolling_mean",
        "rolling_std",
        "rolling_zscore",
        "rolling_volatility",
        "volume",
        "volume_zscore",
        "trend_strength"
    ]

    for var in display_vars:
        row = [var]
        for val in recent[var]:
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table.add_row(*row)

    console.clear()
    console.print(table)


# Pipeline
def run_feature_pipeline():
    raw_data = fetch_raw_market_data(SYMBOL, INTERVAL, PERIOD)
    enriched_data = generate_derived_features(raw_data, ROLLING_WINDOW)
    dataset = persist_dataset(enriched_data)
    render_dataset_table(dataset)

if __name__ == "__main__":
    run_feature_pipeline()