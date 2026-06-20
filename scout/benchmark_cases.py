"""Benchmark the model on multiple MU-like rally cases.

Goal: verify the model isn't just 'MU lucky' — it should catch other
similar sustained tech/semiconductor/solar rallies early.

For each benchmark case, report:
- When did model score peak in the rally window?
- What was the stock's rally-progress at top-score date?
- Which features were 'hottest' at that date?
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from scout.validate import (
    load_split, train_one, FEATURE_COLS,
    FEATURES_PATH, RALLIES_PATH
)


# Curated benchmark cases — significant rallies of "MU-like" stocks.
# Story field captures the thematic backdrop (for context, not used by model).
BENCHMARK_CASES = [
    # Recent semiconductors (AI / HBM / picks-and-shovels)
    ("MU",   "2023-07-07", "Memory chips / HBM rebound"),
    ("LITE", "2023-10-27", "Optical components / AI datacenter"),
    ("MRVL", None,         "Networking silicon / AI custom"),
    ("AMD",  "2022-09-26", "AI GPU positioning"),
    ("ON",   "2018-10-29", "Auto SiC takeoff"),
    ("LSCC", None,         "FPGA / edge AI"),
    # Solar / renewable boom-bust
    ("FSLR", "2023-10-30", "IRA stimulus + AI datacenter PPA"),
    ("ENPH", "2022-01-27", "Solar boom + energy crisis"),
    ("ENPH", "2020-03-18", "COVID rebound: famous +485% rally"),
    # Other big winners worth studying
    ("AMD",  "2018-10-29", "x86 server comeback"),
    ("AMD",  "2018-12-24", "Christmas Eve bottom"),
    ("MU",   "2020-03-16", "COVID memory rebound"),
    ("MU",   "2022-09-26", "Pre-AI bottom"),
    ("LITE", "2019-05-31", "100G transition"),
]


def explain_features(row: pd.Series) -> list[str]:
    """Turn feature values into human-readable bullet points."""
    bullets = []
    rsi = row["rsi_14"]
    if rsi < 30:
        bullets.append(f"RSI {rsi:.0f} — **深度超賣**")
    elif rsi < 40:
        bullets.append(f"RSI {rsi:.0f} — 接近超賣")
    elif rsi > 60:
        bullets.append(f"RSI {rsi:.0f} — 已偏熱")

    dd = row["drawdown_52w"] * 100
    if dd < -40:
        bullets.append(f"距 52w 高 {dd:.0f}% — **嚴重回檔**，反彈空間大")
    elif dd < -25:
        bullets.append(f"距 52w 高 {dd:.0f}% — 顯著回檔")

    ma = row["price_vs_ma200"] * 100
    if ma < -15:
        bullets.append(f"低於 MA200 {ma:.0f}% — 弱勢，但容易超跌反彈")
    elif ma > 5:
        bullets.append(f"高於 MA200 {ma:.0f}% — 仍在上升趨勢")

    vol_surge = row["volume_surge_30d"]
    if vol_surge > 1.5:
        bullets.append(f"30d/90d 量比 {vol_surge:.2f}× — **量能放大**，可能在 accumulate")
    elif vol_surge > 1.2:
        bullets.append(f"30d/90d 量比 {vol_surge:.2f}×")

    m3 = row["momentum_3m"] * 100
    m6 = row["momentum_6m"] * 100
    if m3 < -10 and m6 < -10:
        bullets.append(f"3m {m3:+.0f}% / 6m {m6:+.0f}% — 中期 capitulation")
    elif m3 > 5 and m6 < 0:
        bullets.append(f"3m {m3:+.0f}% / 6m {m6:+.0f}% — **轉向中**")

    rs = row["rs_vs_spy_3m"] * 100
    if rs > 10:
        bullets.append(f"3m 對 SPY 超額 {rs:+.0f}% — 強勢")
    elif rs < -10:
        bullets.append(f"3m 對 SPY 落後 {rs:+.0f}% — 落後大盤")

    days = row["days_since_high"]
    if days > 180:
        bullets.append(f"距上次新高 {days:.0f} 天 — 已熬很久")

    return bullets


def find_rally(rallies_df, ticker, trough_date_str=None):
    """Find the rally for this ticker. If trough_date_str given, find closest."""
    sub = rallies_df[rallies_df["ticker"] == ticker]
    if trough_date_str is None:
        return sub
    target = pd.Timestamp(trough_date_str)
    sub = sub.copy()
    sub["dist"] = (sub["trough_date"] - target).abs()
    return sub.sort_values("dist").head(1)


def study_case(scored_df: pd.DataFrame, rallies_df: pd.DataFrame,
               ticker: str, trough_date_str: str | None, story: str):
    rally = find_rally(rallies_df, ticker, trough_date_str)
    if rally.empty:
        print(f"\n  {ticker} ({trough_date_str}): no matching rally in labels")
        return None
    r = rally.iloc[0]
    trough = r["trough_date"]
    peak = r["peak_date"]
    trough_price = r["trough_price"]
    peak_price = r["peak_price"]
    gain_pct = r["gain_pct"]

    # Window: 90 days before trough → 60 days after
    window_start = trough - pd.Timedelta(days=90)
    window_end = trough + pd.Timedelta(days=60)
    sub = scored_df[
        (scored_df["ticker"] == ticker)
        & (scored_df["date"] >= window_start)
        & (scored_df["date"] <= window_end)
    ].sort_values("date")

    if sub.empty:
        print(f"\n  {ticker} ({trough.date()}): no samples in window")
        return None

    top = sub.loc[sub["score"].idxmax()]
    catch_price = top.get("close", np.nan)
    if pd.notna(catch_price):
        progress = (catch_price - trough_price) / (peak_price - trough_price) * 100
        progress = float(np.clip(progress, -100, 200))
    else:
        progress = np.nan

    grade = (
        "🏆 perfect" if (pd.notna(progress) and progress < 10) else
        "✅ early"   if (pd.notna(progress) and progress < 50) else
        "🟡 mid"     if (pd.notna(progress) and progress < 80) else
        "❌ late"
    )

    print(f"\n{'─'*60}")
    print(f"  {ticker}  ({story})")
    print(f"  Rally: {trough.date()} (${trough_price:>7.2f}) → "
          f"{peak.date()} (${peak_price:>7.2f})  +{gain_pct:.1f}%")
    print(f"  Best model score date: {top['date'].date()}  "
          f"score={top['score']:.3f}  price=${catch_price:.2f}")
    print(f"  Progress at top score: {progress:+.1f}%   {grade}")
    print(f"\n  Why model fired:")
    for bullet in explain_features(top):
        print(f"    • {bullet}")

    return {
        "ticker": ticker,
        "trough_date": trough.date(),
        "top_score": float(top["score"]),
        "top_score_date": top["date"].date(),
        "catch_price": float(catch_price) if pd.notna(catch_price) else None,
        "progress_pct": float(progress) if pd.notna(progress) else None,
        "gain_pct": float(gain_pct),
        "grade": grade,
    }


def main():
    print("📊 Loading data + training model...")
    df, train, val, test = load_split()
    # Train on all train+val data — for benchmark we want max-fit model
    train_combined = pd.concat([train, val], ignore_index=True)
    model = train_one(train_combined, seed=1)

    # Score everything (so we can study val/test rallies too)
    df_scored = df.copy()
    df_scored["score"] = model.predict_proba(df[FEATURE_COLS].values)[:, 1]

    rallies = pd.read_csv(RALLIES_PATH, parse_dates=["trough_date", "peak_date"])

    print(f"\n{'='*60}")
    print(f"🦞 Benchmark cases: {len(BENCHMARK_CASES)} historical rallies")
    print(f"{'='*60}")

    summary = []
    for ticker, trough, story in BENCHMARK_CASES:
        res = study_case(df_scored, rallies, ticker, trough, story)
        if res:
            summary.append(res)

    if not summary:
        return
    sdf = pd.DataFrame(summary)
    print(f"\n{'='*60}")
    print("📊 Summary across benchmarks")
    print(f"{'='*60}")
    n = len(sdf)
    pct_perfect = (sdf["progress_pct"] < 10).sum() / n * 100
    pct_early = (sdf["progress_pct"] < 50).sum() / n * 100
    pct_late = (sdf["progress_pct"] >= 80).sum() / n * 100
    print(f"   Perfect (<10% progress): {pct_perfect:.0f}%  🏆")
    print(f"   Early   (<50% progress): {pct_early:.0f}%  ✅")
    print(f"   Late    (≥80% progress): {pct_late:.0f}%  ❌")
    print(f"   Median progress at catch: {sdf['progress_pct'].median():+.1f}%")
    print(f"   Median top score: {sdf['top_score'].median():.3f}")


if __name__ == "__main__":
    main()
