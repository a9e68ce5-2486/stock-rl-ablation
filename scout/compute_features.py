"""Compute per-day technical features for each ticker in the universe.

All features are point-in-time: at date D, only data from [start, D] is used.
This prevents look-ahead bias.

Labels: positive (1) if any rally for this ticker has trough_date within
        [D, D + early_window_days]. I.e., a rally is starting "now".
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yfinance as yf
from scout.fetch_universe import list_universe

CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(exist_ok=True, parents=True)


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch Close + Volume, cached."""
    cache_path = CACHE_DIR / f"ohlcv_{ticker}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        # Flatten potential MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Close", "Volume"]].rename(columns={"Close": "close",
                                                      "Volume": "volume"})
        df.to_parquet(cache_path)
        return df
    except Exception as e:
        print(f"    ⚠️ {ticker} fetch error: {e}")
        return None


def compute_features(prices_df: pd.DataFrame, spy_close: pd.Series,
                     sample_every: int = 5) -> pd.DataFrame:
    """Compute features for a single ticker. Returns sampled DataFrame.

    All features only look BACKWARD in time (no look-ahead).
    """
    close = prices_df["close"].astype(float)
    volume = prices_df["volume"].astype(float)

    # Align SPY to ticker's dates
    spy = spy_close.reindex(close.index).ffill()

    # Pre-compute rolling values
    daily_ret = close.pct_change()

    ma200 = close.rolling(200).mean()
    high52w = close.rolling(252).max()

    ret_3m = close / close.shift(63) - 1
    ret_6m = close / close.shift(126) - 1
    ret_12m = close / close.shift(252) - 1

    vol_60d = daily_ret.rolling(60).std() * np.sqrt(252)
    # Rank vol vs its own past distribution (only past info)
    vol_60d_rank = vol_60d.expanding(min_periods=60).rank(pct=True)

    vol_30d_mean = volume.rolling(30).mean()
    vol_90d_mean = volume.rolling(90).mean()
    volume_surge = vol_30d_mean / vol_90d_mean

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi_14 = 100 - 100 / (1 + rs)

    # Relative strength vs SPY (3-month excess return)
    spy_ret_3m = spy / spy.shift(63) - 1
    rs_vs_spy_3m = ret_3m - spy_ret_3m

    # Days since 52w high (clipped at 252)
    rolling_high_idx = close.rolling(252).apply(
        lambda x: 252 - 1 - np.argmax(x.values), raw=False
    )
    days_since_high = rolling_high_idx.astype(float)

    feats = pd.DataFrame({
        "drawdown_52w": close / high52w - 1,
        "price_vs_ma200": close / ma200 - 1,
        "momentum_3m": ret_3m,
        "momentum_6m": ret_6m,
        "momentum_12m": ret_12m,
        "vol_60d": vol_60d,
        "vol_60d_rank": vol_60d_rank,
        "volume_surge_30d": volume_surge,
        "rsi_14": rsi_14,
        "rs_vs_spy_3m": rs_vs_spy_3m,
        "days_since_high": days_since_high,
        "close": close,  # keep for analysis, drop before training
    })

    # Drop warmup rows
    feats = feats.dropna()

    # Sample every N trading days
    feats = feats.iloc[::sample_every].copy()
    feats.index.name = "date"
    return feats.reset_index()


def label_features(df: pd.DataFrame, rallies_df: pd.DataFrame,
                   early_window_days: int = 30) -> pd.DataFrame:
    """Label: positive if any rally trough for this ticker is within
    [date, date + early_window_days].
    """
    # Build per-ticker sorted trough dates
    lookup = {}
    for ticker, group in rallies_df.groupby("ticker"):
        lookup[ticker] = np.sort(group["trough_date"].values)

    # Vectorized labeling per ticker
    labels = np.zeros(len(df), dtype=np.int8)
    for ticker in df["ticker"].unique():
        mask = df["ticker"] == ticker
        if ticker not in lookup:
            continue
        troughs = lookup[ticker]
        dates = df.loc[mask, "date"].values
        # For each date, check if any trough is in [date, date + window]
        for i, d in enumerate(dates):
            window_end = d + np.timedelta64(early_window_days, 'D')
            if np.any((troughs >= d) & (troughs <= window_end)):
                global_idx = df.index[mask][i]
                labels[global_idx] = 1
    df["label"] = labels
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2026-06-01")
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--early_window", type=int, default=30)
    parser.add_argument("--out", default="results/features.parquet")
    args = parser.parse_args()

    # Load rallies
    rallies = pd.read_csv(PROJECT_ROOT / "results" / "rallies.csv",
                         parse_dates=["trough_date", "peak_date"])
    print(f"📋 {len(rallies)} rally labels loaded")

    # Fetch SPY first
    print("📡 Fetching SPY benchmark...")
    spy_df = fetch_ohlcv("SPY", args.start, args.end)
    spy_close = spy_df["close"].astype(float)

    tickers = list_universe()
    print(f"📊 Computing features for {len(tickers)} tickers, sample every {args.sample_every} days...\n")

    all_feats = []
    for i, ticker in enumerate(tickers):
        df = fetch_ohlcv(ticker, args.start, args.end)
        if df is None or len(df) < 252:
            print(f"  [{i+1:3d}/{len(tickers)}] {ticker:6s} — skipped")
            continue
        feats = compute_features(df, spy_close, args.sample_every)
        feats["ticker"] = ticker
        all_feats.append(feats)
        print(f"  [{i+1:3d}/{len(tickers)}] {ticker:6s} — {len(feats)} samples")

    df = pd.concat(all_feats, ignore_index=True)
    print(f"\n📦 Total samples: {len(df):,}")

    print(f"🏷️  Labeling (positive = trough within next {args.early_window} days)...")
    df = label_features(df, rallies, args.early_window)
    pos_rate = df["label"].mean()
    print(f"   Positive: {df['label'].sum():,} ({pos_rate*100:.1f}%)")
    print(f"   Negative: {(df['label']==0).sum():,}")

    # Save
    out_path = PROJECT_ROOT / args.out
    df.to_parquet(out_path)
    print(f"\n💾 Saved → {out_path}")

    # Stats by year
    df["year"] = df["date"].dt.year
    by_year = df.groupby("year").agg(
        total=("label", "count"),
        positive=("label", "sum"),
    )
    by_year["pos_pct"] = by_year["positive"] / by_year["total"] * 100
    print("\nSamples per year:")
    print(by_year.to_string())


if __name__ == "__main__":
    main()
