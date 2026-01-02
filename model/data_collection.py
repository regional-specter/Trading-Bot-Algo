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

# LAYER 1: RAW MARKET DATA
def fetch_raw_market_data(symbol, interval, period):
    """
    Fetches raw OHLCV data from the API.
    This is the only place external data enters the system.
    """
    data = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False
    )

    data.reset_index(inplace=True)
    data.columns = [c.lower().replace(" ", "_") for c in data.columns]
    return data


# LAYER 1.5: DERIVED MARKET CONTEXT
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

# STORAGE LAYER (DATASET / DB READY)
def persist_dataset(df):
    """
    This function represents your storage layer.
    Right now it simply returns the DataFrame,
    but this is where SQLite, DuckDB, or Parquet fits later.
    """
    return df.dropna().reset_index(drop=True)



# TUI RENDERING LAYER
def render_dataset_table(df, max_rows=10):
    """
    Renders the dataset as a TUI table using rich.
    """

    table = Table(title=f"Market Dataset | {SYMBOL}")

    for col in df.columns:
        table.add_column(col, justify="right", no_wrap=True)

    for _, row in df.tail(max_rows).iterrows():
        table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

    console.clear()
    console.print(table)


# PIPELINE ORCHESTRATOR
def run_pipeline():
    raw_data = fetch_raw_market_data(SYMBOL, INTERVAL, PERIOD)
    enriched_data = generate_derived_features(raw_data, ROLLING_WINDOW)
    dataset = persist_dataset(enriched_data)
    render_dataset_table(dataset)

if __name__ == "__main__":
    run_pipeline()
