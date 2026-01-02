import yfinance as yf
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def fetch_and_clean(ticker, start, end, interval="1d"):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False
    )

    df = df.dropna()
    df = df[~df.index.duplicated()]
    df.index = pd.to_datetime(df.index)

    return df


def save_data(df, ticker):
    path = DATA_DIR / f"{ticker}.csv"
    df.to_csv(path)
    return path


def load_data(ticker):
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    df = df.dropna()
    return df


def sample_time_window(df, window_size):
    if len(df) <= window_size:
        raise ValueError("Window size larger than dataset")

    start = df.sample(1).index[0]
    idx = df.index.get_loc(start)

    end = idx + window_size
    if end >= len(df):
        idx = len(df) - window_size

    return df.iloc[idx:idx + window_size]


if __name__ == "__main__":
    df = fetch_and_clean("AAPL", "2018-01-01", "2024-01-01")
    save_data(df, "AAPL")
