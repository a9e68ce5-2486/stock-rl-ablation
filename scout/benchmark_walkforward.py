"""Walk-forward benchmark: TRULY out-of-sample case studies.

For each benchmark rally, train the model on data BEFORE that rally's trough,
then score the model on the rally window. This is the only honest way to test
"did the model discover this stock, or did it memorize it?".

Contrast with benchmark_cases.py which had data leak (training data included
the rallies being studied).
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from scout.validate import FEATURES_PATH, RALLIES_PATH, FEATURE_COLS
from scout.benchmark_cases import explain_features, BENCHMARK_CASES, find_rally


def train_with_cutoff(features_df: pd.DataFrame, cutoff_date: pd.Timestamp,
                      seed: int = 1) -> GradientBoostingClassifier:
    """Train using only features dated BEFORE cutoff_date.

    This is the key: model has never seen any sample from on/after cutoff_date,
    so any rally starting at or after cutoff is truly out-of-sample.
    """
    train = features_df[features_df["date"] < cutoff_date]
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=seed,
    )
    model.fit(train[FEATURE_COLS].values, train["label"].values)
    return model, len(train)


def study_case_clean(features_df: pd.DataFrame, rallies_df: pd.DataFrame,
                     ticker: str, trough_date_str: str | None,
                     story: str, cutoff_buffer_days: int = 60):
    rally = find_rally(rallies_df, ticker, trough_date_str)
    if rally.empty:
        return None
    r = rally.iloc[0]
    trough = pd.Timestamp(r["trough_date"])
    peak = pd.Timestamp(r["peak_date"])
    trough_price = r["trough_price"]
    peak_price = r["peak_price"]
    gain_pct = r["gain_pct"]

    # Train cutoff = trough_date - buffer_days
    # Why buffer? Because if model sees data RIGHT before trough, labels at
    # that point already include "rally coming in 30 days" which is leak.
    cutoff = trough - pd.Timedelta(days=cutoff_buffer_days)

    model, n_train = train_with_cutoff(features_df, cutoff)

    # Score the window around the trough (90 days before to 60 after)
    window_start = trough - pd.Timedelta(days=90)
    window_end = trough + pd.Timedelta(days=60)
    sub = features_df[
        (features_df["ticker"] == ticker)
        & (features_df["date"] >= window_start)
        & (features_df["date"] <= window_end)
    ].copy()
    if sub.empty:
        return None

    sub["score"] = model.predict_proba(sub[FEATURE_COLS].values)[:, 1]
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
    print(f"  🔒 Train cutoff: {cutoff.date()}  (model trained on {n_train:,} samples,")
    print(f"     all from BEFORE {cutoff.date()})")
    print(f"  Best score date: {top['date'].date()}  score={top['score']:.3f}  "
          f"price=${catch_price:.2f}")
    print(f"  Progress at top score: {progress:+.1f}%   {grade}")
    print(f"  Why model fired:")
    for bullet in explain_features(top):
        print(f"    • {bullet}")

    return {
        "ticker": ticker, "trough_date": trough.date(),
        "top_score": float(top["score"]),
        "top_score_date": top["date"].date(),
        "progress_pct": float(progress) if pd.notna(progress) else None,
        "gain_pct": float(gain_pct),
        "grade": grade,
    }


def main():
    print("📊 Loading data...")
    features_df = pd.read_parquet(FEATURES_PATH)
    features_df["date"] = pd.to_datetime(features_df["date"])
    rallies = pd.read_csv(RALLIES_PATH, parse_dates=["trough_date", "peak_date"])

    print(f"\n{'='*60}")
    print(f"🔬 Walk-forward benchmark: {len(BENCHMARK_CASES)} cases")
    print(f"   For each rally, train on data BEFORE that rally's trough.")
    print(f"   Model has NEVER seen the rally's features or labels.")
    print(f"{'='*60}")

    summary = []
    for ticker, trough, story in BENCHMARK_CASES:
        try:
            res = study_case_clean(features_df, rallies, ticker, trough, story)
            if res:
                summary.append(res)
        except Exception as e:
            print(f"\n  ⚠️ {ticker} failed: {e}")

    if not summary:
        return
    sdf = pd.DataFrame(summary)
    n = len(sdf)
    pct_perfect = (sdf["progress_pct"] < 10).sum() / n * 100
    pct_early = (sdf["progress_pct"] < 50).sum() / n * 100
    pct_mid = ((sdf["progress_pct"] >= 50) & (sdf["progress_pct"] < 80)).sum() / n * 100
    pct_late = (sdf["progress_pct"] >= 80).sum() / n * 100

    print(f"\n{'='*60}")
    print(f"📊 WALK-FORWARD Summary (truly out-of-sample, n={n})")
    print(f"{'='*60}")
    print(f"   Perfect (<10% progress): {pct_perfect:.0f}%  🏆")
    print(f"   Early   (<50% progress): {pct_early:.0f}%  ✅")
    print(f"   Mid     (50-80%):        {pct_mid:.0f}%  🟡")
    print(f"   Late    (≥80% progress): {pct_late:.0f}%  ❌")
    print(f"   Median progress at catch: {sdf['progress_pct'].median():+.1f}%")
    print(f"   Median top score        : {sdf['top_score'].median():.3f}")
    print()
    print("   📝 Compare with leaky benchmark_cases.py:")
    print("      Leaky version was 85% perfect, 100% early.")
    print("      If walk-forward is significantly worse → leak was real.")
    print("      If walk-forward is similar → model genuinely discovered these.")


if __name__ == "__main__":
    main()
