"""Fetch daily OHLCV from yfinance with parquet caching."""
from __future__ import annotations
import pandas as pd
import yfinance as yf
from pathlib import Path
import numpy as np

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def fetch(tickers: list[str], start: str, end: str, refresh: bool = False) -> pd.DataFrame:
    """Fetch adjusted OHLCV. Returns a DataFrame with MultiIndex columns (field, ticker)."""
    key = f"prices_{'_'.join(sorted(tickers))}_{start}_{end}.parquet"
    cache_path = CACHE_DIR / key

    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    df = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    df = df.dropna(how="all")
    df.to_parquet(cache_path)
    return df


def compute_features(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Per-ticker features: log return, rolling vol, momentum.
    Returns DataFrame with columns like AAPL_logret, AAPL_vol20, AAPL_mom20.
    """
    close = prices["Close"][tickers]
    volume = prices["Volume"][tickers]
    out = pd.DataFrame(index=close.index)

    logret = np.log(close / close.shift(1))
    for t in tickers:
        out[f"{t}_logret"] = logret[t]
        out[f"{t}_vol20"] = logret[t].rolling(20).std()
        out[f"{t}_mom20"] = (close[t] / close[t].shift(20) - 1)
        out[f"{t}_mom60"] = (close[t] / close[t].shift(60) - 1)
        # Volume z-score - signals unusual trading
        v = volume[t]
        out[f"{t}_volz"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)

    return out.dropna()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"]
    prices = fetch(tickers, "2015-01-01", "2026-06-01")
    feats = compute_features(prices, tickers)
    print(f"Prices shape: {prices.shape}")
    print(f"Features shape: {feats.shape}")
    print(f"\nFirst 3 features:\n{feats.iloc[:3, :5]}")
    print(f"\nDate range: {feats.index.min()} ~ {feats.index.max()}")
