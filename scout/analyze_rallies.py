"""Summarize the rally labels to decide if there's enough data to build a classifier."""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def main():
    csv_path = PROJECT_ROOT / "results" / "rallies.csv"
    if not csv_path.exists():
        print(f"❌ Run scout/label_rallies.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path, parse_dates=["trough_date", "peak_date"])
    print(f"📊 Loaded {len(df)} rallies from {csv_path.name}\n")

    print("=" * 60)
    print("Top-level stats")
    print("=" * 60)
    print(f"Tickers covered      : {df['ticker'].nunique()}")
    print(f"Total rallies        : {len(df)}")
    print(f"Avg gain             : {df['gain_pct'].mean():>6.1f}%")
    print(f"Median gain          : {df['gain_pct'].median():>6.1f}%")
    print(f"Max gain             : {df['gain_pct'].max():>6.1f}%")
    print(f"Avg duration (days)  : {df['duration_days'].mean():>6.1f}")
    print(f"Median duration      : {df['duration_days'].median():>6.1f}")

    print()
    print("=" * 60)
    print("Rallies per year (year of trough)")
    print("=" * 60)
    by_year = df.groupby(df["trough_date"].dt.year).size()
    for year, n in by_year.items():
        bar = "█" * min(n, 40)
        print(f"  {year}: {n:>3d}  {bar}")

    print()
    print("=" * 60)
    print("Top 10 tickers by rally count")
    print("=" * 60)
    top = df.groupby("ticker").size().sort_values(ascending=False).head(10)
    for t, n in top.items():
        print(f"  {t:6s}: {n} rallies")

    print()
    print("=" * 60)
    print("⭐ Benchmark cases — did we catch them?")
    print("=" * 60)
    for benchmark in ["MU", "LITE", "ENPH", "FSLR", "AMD", "NVDA"]:
        sub = df[df["ticker"] == benchmark].sort_values("trough_date")
        if sub.empty:
            print(f"  {benchmark:6s}: ❌ no rallies found in our threshold")
            continue
        print(f"\n  {benchmark}:")
        for _, r in sub.iterrows():
            print(f"    {r['trough_date'].date()} → {r['peak_date'].date()}  "
                  f"{r['trough_price']:>7.2f} → {r['peak_price']:>7.2f}  "
                  f"({r['gain_pct']:>5.1f}% in {r['duration_days']:>3} days)")

    print()
    print("=" * 60)
    print("Verdict")
    print("=" * 60)
    if len(df) < 100:
        print("❌ Too few labels (<100). Lower gain_threshold or expand universe.")
    elif len(df) < 300:
        print("⚠️  Adequate (100-300). Classifier doable but with caveats.")
    else:
        print(f"✅ Plenty of labels ({len(df)}). Classifier can be trained.")

    n_2024_plus = len(df[df["trough_date"].dt.year >= 2024])
    n_train = len(df) - n_2024_plus
    print(f"   Training period 2015-2023: {n_train} rallies")
    print(f"   Test period 2024+        : {n_2024_plus} rallies")
    if n_2024_plus < 20:
        print(f"   ⚠️ Test set has only {n_2024_plus} rallies — eval will be noisy.")


if __name__ == "__main__":
    main()
