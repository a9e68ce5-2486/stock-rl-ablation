"""Find historical rallies in the universe.

A "rally" = stock gained ≥ gain_threshold within window_days.
Rally start (trough) = local minimum preceding the peak.
Rally peak = local maximum within window from trough.

This produces the supervised learning labels:
  positive sample = "at date X, ticker T was within `early_window` days of rally start"
  negative sample = everything else
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


def fetch_close(ticker: str, start: str, end: str) -> pd.Series | None:
    """Fetch adjusted close price. Cache per-ticker."""
    cache_path = CACHE_DIR / f"close_{ticker}_{start}_{end}.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)["close"]

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                         progress=False)
        if df.empty or "Close" not in df.columns:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close.name = "close"
        close.to_frame().to_parquet(cache_path)
        return close
    except Exception as e:
        print(f"  ⚠️ {ticker} fetch failed: {e}")
        return None


def find_rallies(
    prices: pd.Series,
    gain_threshold: float = 0.50,
    window_days: int = 180,
    min_gap_days: int = 60,
) -> list[dict]:
    """Find non-overlapping rallies in a price series.

    Algorithm: sliding window. At each position, find local trough within
    `min_gap_days`, look forward `window_days`, check if peak gain ≥ threshold.
    """
    if len(prices) < window_days + min_gap_days:
        return []

    rallies = []
    i = 0
    prices = prices.dropna()
    n = len(prices)

    while i < n - window_days:
        # Local trough in next min_gap_days
        trough_slice = prices.iloc[i : i + min_gap_days]
        trough_price = float(trough_slice.min())
        trough_date = trough_slice.idxmin()

        # Look forward window_days from trough
        trough_pos = prices.index.get_loc(trough_date)
        forward_end = min(trough_pos + window_days, n)
        forward_slice = prices.iloc[trough_pos : forward_end]
        peak_price = float(forward_slice.max())
        peak_date = forward_slice.idxmax()
        peak_pos = prices.index.get_loc(peak_date)

        gain = (peak_price - trough_price) / trough_price

        if gain >= gain_threshold and peak_pos > trough_pos:
            rallies.append({
                "trough_date": trough_date,
                "trough_price": trough_price,
                "peak_date": peak_date,
                "peak_price": peak_price,
                "gain_pct": gain * 100,
                "duration_days": (peak_date - trough_date).days,
            })
            # Skip past this rally
            i = peak_pos + 1
        else:
            i += min_gap_days  # advance ~2 months

    return rallies


def label_universe(
    tickers: list[str],
    start: str,
    end: str,
    gain_threshold: float = 0.50,
    window_days: int = 180,
) -> pd.DataFrame:
    """Build a master label table across the universe."""
    rows = []
    for j, ticker in enumerate(tickers):
        print(f"  [{j+1:3d}/{len(tickers)}] {ticker:6s}", end="")
        prices = fetch_close(ticker, start, end)
        if prices is None or len(prices) < 250:
            print(" — skipped (no data)")
            continue
        rallies = find_rallies(prices, gain_threshold, window_days)
        print(f" — {len(rallies)} rallies (first close {prices.index[0].date()})")
        for r in rallies:
            r["ticker"] = ticker
            rows.append(r)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["ticker", "trough_date", "peak_date", "trough_price",
                 "peak_price", "gain_pct", "duration_days"]]
        df = df.sort_values(["ticker", "trough_date"]).reset_index(drop=True)
    return df


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-01-01")
    parser.add_argument("--gain", type=float, default=0.50,
                        help="rally threshold (0.50 = 50%)")
    parser.add_argument("--window", type=int, default=180,
                        help="window days for rally to complete")
    parser.add_argument("--out", default="results/rallies.csv")
    args = parser.parse_args()

    tickers = list_universe()
    print(f"📡 Scanning {len(tickers)} tickers from {args.start} to {args.end}...")
    print(f"   Rally criterion: ≥{args.gain*100:.0f}% gain within {args.window} days")
    print()

    df = label_universe(tickers, args.start, args.end, args.gain, args.window)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(out_path, index=False)

    print()
    print("=" * 60)
    print(f"💾 Saved {len(df)} rallies → {out_path}")
    print("=" * 60)
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
