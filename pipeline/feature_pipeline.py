import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Main entry point to generate all engineered features from raw OHLCV data
def build_feature_set(df, window):
    """
    Orchestrates all feature computation steps
    """

    # Compute return-based features
    df = compute_returns(df)

    # Compute rolling statistics and dispersion measures
    df = compute_rolling_statistics(df, window)

    # Compute volatility-focused features
    df = compute_volatility_features(df, window)

    # Compute price action and range-based features
    df = compute_price_action_features(df, window)

    # Compute volume-normalized features
    df = compute_volume_features(df, window)

    # Compute simple trend and momentum proxies
    df = compute_trend_features(df)

    return df


# Returns & Normalisation

# Computes both log and simple returns for price normalization
def compute_returns(df):
    # Log returns stabilize variance and are additive over time
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    # Simple returns are intuitive percentage changes
    df["simple_return"] = df["close"].pct_change()

    return df


# Rolling Statistics

# Computes rolling mean, standard deviation and z-score normalization
def compute_rolling_statistics(df, window):
    # Rolling mean represents short-term market equilibrium
    df["rolling_mean"] = df["close"].rolling(window).mean()

    # Rolling standard deviation captures dispersion around the mean
    df["rolling_std"] = df["close"].rolling(window).std()

    # Z-score measures how extreme the current price is vs recent history
    df["rolling_zscore"] = (
        (df["close"] - df["rolling_mean"]) / df["rolling_std"]
    )

    return df


# Volatility Features

# Computes volatility metrics using returns
def compute_volatility_features(df, window):
    # Rolling volatility is the standard deviation of log returns
    df["rolling_volatility"] = df["log_return"].rolling(window).std()

    return df


# Price Action Features

# Extracts candle-based and range-based behavior
def compute_price_action_features(df, window):
    # Intrabar price range reflects immediate uncertainty
    df["price_range"] = df["high"] - df["low"]

    # Smoothed price range highlights regime shifts in volatility
    df["rolling_range"] = df["price_range"].rolling(window).mean()

    return df


# Volume Features

# Normalizes volume to detect abnormal participation
def compute_volume_features(df, window):
    # Rolling mean volume establishes baseline participation
    df["volume_mean"] = df["volume"].rolling(window).mean()

    # Volume z-score flags unusual activity relative to recent history
    df["volume_zscore"] = (
        (df["volume"] - df["volume_mean"]) /
        df["volume"].rolling(window).std()
    )

    return df

# Trend & Momentum Proxies

# Computes simple trend strength from rolling averages
def compute_trend_features(df):
    # First derivative of rolling mean approximates trend direction & strength
    df["trend_strength"] = df["rolling_mean"].diff()

    return df

# Renders recent engineered features in a vertical, terminal-friendly table
def render_feature_table(df, lookback=3, title="Feature Snapshot"):
    """
    Rows = feature names
    Columns = recent timesteps
    """

    # Select only the most recent timesteps
    recent = df.tail(lookback)

    # Create Rich table with contextual title
    table = Table(title=title)

    # First column lists feature names
    table.add_column("Feature", justify="left", no_wrap=True)

    # Add one column per recent timestamp
    for ts in recent["datetime"]:
        table.add_column(str(ts.time()), justify="right", no_wrap=True)

    # Explicitly control which features are visualized
    display_features = [
        "close",
        "log_return",
        "simple_return",
        "rolling_mean",
        "rolling_std",
        "rolling_zscore",
        "rolling_volatility",
        "price_range",
        "rolling_range",
        "volume",
        "volume_zscore",
        "trend_strength"
    ]

    # Render each feature as a row across recent timesteps
    for feature in display_features:
        row = [feature]
        for val in recent[feature]:
            # Format floats for dense terminal readability
            if isinstance(val, float):
                row.append(f"{val:.4f}")
            else:
                row.append(str(val))
        table.add_row(*row)

    # Print table to terminal
    console.print(table)

# Runs feature engineering when this file is executed directly
def run():
    from pipeline.data_pipeline import (
        fetch_raw_market_data,
        SYMBOL,
        INTERVAL,
        PERIOD,
        ROLLING_WINDOW
    )

    # Fetch clean raw OHLCV data
    raw_df = fetch_raw_market_data(SYMBOL, INTERVAL, PERIOD)

    # Generate full feature set from raw data
    feature_df = build_feature_set(raw_df, ROLLING_WINDOW)

    # Remove NaNs introduced by rolling windows
    feature_df = feature_df.dropna().reset_index(drop=True)

    # Render engineered feature snapshot in terminal
    render_feature_table(
        feature_df,
        lookback=3,
        title="Engineered Feature Snapshot"
    )

    return feature_df


if __name__ == "__main__":
    run()