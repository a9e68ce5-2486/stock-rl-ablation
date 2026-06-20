"""Validation pipeline for the rally classifier.

Two pieces:
  1. **Multi-seed AUC** — train N models with different random_state.
     Reports mean ± std of test AUC to know if 0.69 is real or lucky.

  2. **Top-K Precision** — the actual screening use case.
     Each week: rank stocks by score, take top K, check how many
     had a rally start in the next 30 days. This is what you'd
     actually do as a user.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES_PATH = PROJECT_ROOT / "results" / "features.parquet"
RALLIES_PATH = PROJECT_ROOT / "results" / "rallies.csv"

FEATURE_COLS = [
    "drawdown_52w", "price_vs_ma200",
    "momentum_3m", "momentum_6m", "momentum_12m",
    "vol_60d", "vol_60d_rank", "volume_surge_30d",
    "rsi_14", "rs_vs_spy_3m", "days_since_high",
]


def load_split():
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    train = df[df["year"] <= 2022].copy()
    val = df[df["year"] == 2023].copy()
    test = df[df["year"] >= 2024].copy()
    return df, train, val, test


def train_one(train, seed):
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=seed,
    )
    model.fit(train[FEATURE_COLS].values, train["label"].values)
    return model


def multi_seed(train, val, test, n_seeds: int = 5):
    print(f"\n{'='*60}")
    print(f"🌱 Multi-seed AUC stability (n={n_seeds})")
    print(f"{'='*60}")
    rows = []
    for seed in range(n_seeds):
        print(f"  Seed {seed}... ", end="", flush=True)
        model = train_one(train, seed)
        val_auc = roc_auc_score(val["label"], model.predict_proba(val[FEATURE_COLS])[:, 1])
        test_auc = roc_auc_score(test["label"], model.predict_proba(test[FEATURE_COLS])[:, 1])
        test_ap = average_precision_score(test["label"],
                                          model.predict_proba(test[FEATURE_COLS])[:, 1])
        print(f"val AUC = {val_auc:.3f}  test AUC = {test_auc:.3f}  test AP = {test_ap:.3f}")
        rows.append({"seed": seed, "val_auc": val_auc,
                    "test_auc": test_auc, "test_ap": test_ap})
    df = pd.DataFrame(rows)
    print(f"\n  📐 Test AUC  : {df['test_auc'].mean():.3f} ± {df['test_auc'].std():.3f}")
    print(f"  📐 Test AP   : {df['test_ap'].mean():.3f} ± {df['test_ap'].std():.3f}")
    print(f"  📐 Val AUC   : {df['val_auc'].mean():.3f} ± {df['val_auc'].std():.3f}")

    # Significance: is mean test AUC > 0.5?
    mean = df['test_auc'].mean()
    std = df['test_auc'].std() if len(df) > 1 else 1.0
    t_stat = (mean - 0.5) / (std / np.sqrt(len(df)) + 1e-9)
    print(f"\n  📐 Significance (AUC > 0.5): t = {t_stat:.2f}")
    if abs(t_stat) > 2:
        verdict = "✅ Real signal"
    elif abs(t_stat) > 1:
        verdict = "⚠️  Probably real, need more seeds"
    else:
        verdict = "❌ Could be noise"
    print(f"  Verdict: {verdict}")
    return df


def topk_eval(model, scored_test, rallies, ks=(5, 10, 20), eval_freq_days: int = 7):
    """Each `eval_freq_days`, rank stocks by score, take top K, check
    how many had a rally trough within the next 30 days.
    """
    # Build rally lookup
    rally_lookup = {}
    for _, r in rallies.iterrows():
        rally_lookup.setdefault(r["ticker"], []).append(pd.Timestamp(r["trough_date"]))

    # Score test set
    scored = scored_test.copy()
    scored = scored.sort_values("date")

    # Get evaluation dates (every Nth day)
    dates = scored["date"].drop_duplicates().sort_values().values
    eval_dates = dates[::eval_freq_days]

    # Universe size (constant across dates)
    n_tickers = scored["ticker"].nunique()

    results = {k: [] for k in ks}
    baseline_rates = []

    for d in eval_dates:
        d = pd.Timestamp(d)
        window_end = d + pd.Timedelta(days=30)

        # Take latest score per ticker up to d
        recent = scored[(scored["date"] >= d - pd.Timedelta(days=10))
                        & (scored["date"] <= d)]
        if recent.empty:
            continue
        latest = recent.sort_values("date").groupby("ticker").tail(1)
        latest = latest.sort_values("score", ascending=False)

        # Baseline: random pick — what's the expected rate?
        # Count how many tickers have rally in window
        rally_tickers_in_window = set()
        for ticker, troughs in rally_lookup.items():
            if any(d <= t <= window_end for t in troughs):
                rally_tickers_in_window.add(ticker)
        baseline_rate = len(rally_tickers_in_window) / max(n_tickers, 1)
        baseline_rates.append(baseline_rate)

        for k in ks:
            top_k = latest.head(k)
            hits = sum(1 for tk in top_k["ticker"]
                       if tk in rally_tickers_in_window)
            results[k].append({"date": d, "precision": hits / k, "hits": hits})

    summary = []
    avg_baseline = np.mean(baseline_rates) if baseline_rates else 0
    for k in ks:
        if not results[k]:
            continue
        precisions = [x["precision"] for x in results[k]]
        hits = [x["hits"] for x in results[k]]
        lift = (np.mean(precisions) / avg_baseline) if avg_baseline > 0 else np.nan
        summary.append({
            "k": k,
            "n_evals": len(results[k]),
            "precision_mean": np.mean(precisions),
            "precision_median": np.median(precisions),
            "expected_hits": np.mean(hits),
            "baseline_rate": avg_baseline,
            "lift_over_random": lift,
        })

    return pd.DataFrame(summary), results


def main():
    df, train, val, test = load_split()
    print(f"Loaded: {len(df):,} samples, {train['label'].sum()} train positives, "
          f"{test['label'].sum()} test positives")

    # 1. Multi-seed stability
    seed_df = multi_seed(train, val, test, n_seeds=5)

    # 2. Train one model (best seed by val AUC) for top-K analysis
    best_seed = int(seed_df.sort_values("val_auc", ascending=False).iloc[0]["seed"])
    print(f"\n🎯 Using seed {best_seed} (best by val AUC) for top-K analysis")
    model = train_one(train, best_seed)

    rallies = pd.read_csv(RALLIES_PATH, parse_dates=["trough_date", "peak_date"])

    scored_test = test.copy()
    scored_test["score"] = model.predict_proba(test[FEATURE_COLS].values)[:, 1]

    print(f"\n{'='*60}")
    print("🎯 Top-K Precision (your actual screening use case)")
    print(f"{'='*60}")
    print("Every 7 days, rank stocks by score, take top K, check % with rally in next 30d.\n")

    summary_df, _ = topk_eval(model, scored_test, rallies, ks=(5, 10, 20, 50))

    for _, r in summary_df.iterrows():
        print(f"  Top-{int(r['k']):>3d}: precision = {r['precision_mean']*100:>5.1f}%  "
              f"(~{r['expected_hits']:.1f} of {int(r['k'])} pick(s) rally)  "
              f"vs random {r['baseline_rate']*100:.1f}%  "
              f"→ lift {r['lift_over_random']:.2f}×")

    print(f"\n{'='*60}")
    print("💡 How to read")
    print(f"{'='*60}")
    print("• Top-5 precision 20% = 1 of 5 top picks rally → useful")
    print("• Top-20 precision 8% vs random 4% = 2× lift = clear edge")
    print("• 'Random baseline' = if you picked randomly, this is your % hit rate")
    print("• Lift > 1.5× = model adds real value over random screening")


if __name__ == "__main__":
    main()
