"""Train a rally-detection classifier.

Train  : 2015-2022 rallies (warmup period; smaller universe)
Val    : 2023 rallies     (single year for tuning threshold)
Test   : 2024+ rallies    (~40 rallies — the real test)

Reports:
- AUC-ROC, Average Precision
- Per-rally recall: did model flag the stock during early window?
- Early Catch Rate: when caught, what % of rally was already done?
  (Lower = caught earlier = better)
- Top-K precision: when model picks K candidates, how many were real rallies?
- MU 2023-2024 case study: was MU flagged?
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
    "drawdown_52w",
    "price_vs_ma200",
    "momentum_3m",
    "momentum_6m",
    "momentum_12m",
    "vol_60d",
    "vol_60d_rank",
    "volume_surge_30d",
    "rsi_14",
    "rs_vs_spy_3m",
    "days_since_high",
]


def load_data():
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def split_train_val_test(df):
    train = df[df["year"] <= 2022].copy()
    val = df[df["year"] == 2023].copy()
    test = df[df["year"] >= 2024].copy()
    return train, val, test


def train_model(train: pd.DataFrame) -> GradientBoostingClassifier:
    X = train[FEATURE_COLS].values
    y = train["label"].values
    print(f"  X shape: {X.shape}")
    print(f"  Positive class: {y.sum()} ({y.mean()*100:.1f}%)")

    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    model.fit(X, y)
    return model


def score(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    df["score"] = model.predict_proba(df[FEATURE_COLS].values)[:, 1]
    return df


def evaluate_basic(label_name: str, scored: pd.DataFrame):
    y = scored["label"].values
    p = scored["score"].values
    if y.sum() == 0:
        print(f"  ({label_name} has no positive labels — skipping)")
        return
    auc = roc_auc_score(y, p)
    ap = average_precision_score(y, p)
    print(f"  {label_name:6s} | AUC = {auc:.3f} | AvgPrecision = {ap:.3f} | "
          f"baseline AP = {y.mean():.3f} | n={len(scored):,} | pos={y.sum()}")


def evaluate_rally_capture(scored: pd.DataFrame, rallies: pd.DataFrame,
                           score_threshold: float = 0.5,
                           lookback_days: int = 60):
    """For each rally in test period, check if model scored above threshold
    at any point in [trough - 60d, trough + 30d]. If yes, find the EARLIEST
    such date and compute rally_progress at that date.
    """
    captured = []
    missed = []

    for _, r in rallies.iterrows():
        ticker = r["ticker"]
        trough = pd.Timestamp(r["trough_date"])
        peak = pd.Timestamp(r["peak_date"])
        trough_price = r["trough_price"]
        peak_price = r["peak_price"]

        # Window: 60 days before trough → 30 days after
        window_start = trough - pd.Timedelta(days=lookback_days)
        window_end = trough + pd.Timedelta(days=30)

        ticker_scores = scored[
            (scored["ticker"] == ticker)
            & (scored["date"] >= window_start)
            & (scored["date"] <= window_end)
        ].sort_values("date")

        if ticker_scores.empty:
            continue  # no feature samples in window (skipped per sample_every)

        flagged = ticker_scores[ticker_scores["score"] >= score_threshold]
        if flagged.empty:
            missed.append({"ticker": ticker, "trough": trough,
                          "peak": peak, "gain_pct": r["gain_pct"]})
            continue

        catch_date = flagged.iloc[0]["date"]
        catch_price = flagged.iloc[0].get("close", np.nan)

        # Rally progress at catch (0% = trough, 100% = peak)
        if pd.notna(catch_price) and peak_price > trough_price:
            progress = (catch_price - trough_price) / (peak_price - trough_price) * 100
            progress = float(np.clip(progress, -50, 150))
        else:
            progress = np.nan

        # If caught before trough → progress < 0 (good)
        captured.append({
            "ticker": ticker,
            "trough": trough,
            "peak": peak,
            "catch_date": catch_date,
            "catch_price": catch_price,
            "progress_at_catch": progress,
            "gain_pct": r["gain_pct"],
        })

    cap_df = pd.DataFrame(captured)
    miss_df = pd.DataFrame(missed)
    total = len(captured) + len(missed)
    if total == 0:
        return cap_df, miss_df
    print(f"\n  📡 Rally capture (threshold={score_threshold}, lookback=-{lookback_days}d):")
    print(f"     Caught : {len(captured):>3d}/{total} ({len(captured)/total*100:.0f}%)")
    print(f"     Missed : {len(missed):>3d}/{total}")
    if captured:
        prog = cap_df["progress_at_catch"].dropna()
        print(f"     Median progress at catch: {prog.median():>+5.1f}%  "
              f"(<0 = caught BEFORE rally start; <50% = early)")
        print(f"     25th-75th percentile    : {prog.quantile(0.25):>+5.1f}% "
              f"to {prog.quantile(0.75):>+5.1f}%")
    return cap_df, miss_df


def mu_case_study(scored: pd.DataFrame, rallies: pd.DataFrame):
    mu_test_rallies = rallies[(rallies["ticker"] == "MU")
                              & (rallies["trough_date"] >= "2024-01-01")]
    mu_train_rallies = rallies[(rallies["ticker"] == "MU")
                                & (rallies["trough_date"] < "2024-01-01")
                                & (rallies["trough_date"] >= "2023-01-01")]

    print("\n  🎯 MU case study (the $100 MU criterion):")
    for source_name, source in [("2023 rally (technically in val set)", mu_train_rallies),
                                 ("2024+ rallies (test set)", mu_test_rallies)]:
        if source.empty:
            continue
        print(f"\n  --- {source_name} ---")
        for _, r in source.iterrows():
            trough = pd.Timestamp(r["trough_date"])
            peak = pd.Timestamp(r["peak_date"])
            print(f"\n  MU rally: {trough.date()} (\${r['trough_price']:.2f}) → "
                  f"{peak.date()} (\${r['peak_price']:.2f}), +{r['gain_pct']:.1f}%")

            window_start = trough - pd.Timedelta(days=90)
            window_end = trough + pd.Timedelta(days=180)
            mu_window = scored[(scored["ticker"] == "MU")
                              & (scored["date"] >= window_start)
                              & (scored["date"] <= window_end)].sort_values("date")
            if mu_window.empty:
                print("    (no samples in window)")
                continue
            top_scores = mu_window.nlargest(5, "score")
            print(f"    Top 5 score dates in [{window_start.date()}, {window_end.date()}]:")
            for _, row in top_scores.iterrows():
                price = row.get("close", np.nan)
                progress = ((price - r["trough_price"])
                            / (r["peak_price"] - r["trough_price"]) * 100
                            if pd.notna(price) else np.nan)
                print(f"      {row['date'].date()}  score={row['score']:.3f}  "
                      f"price=${price:>7.2f}  progress={progress:>+6.1f}%")


def feature_importance(model):
    print("\n  📈 Feature importance:")
    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    imp = imp.sort_values(ascending=False)
    for f, v in imp.items():
        bar = "█" * int(v * 50)
        print(f"     {f:<22} {v:.4f} {bar}")


def main():
    print("📊 Loading data...")
    df = load_data()
    print(f"  Total samples: {len(df):,}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Tickers: {df['ticker'].nunique()}")
    print(f"  Positive rate: {df['label'].mean()*100:.1f}%")

    rallies = pd.read_csv(RALLIES_PATH,
                          parse_dates=["trough_date", "peak_date"])

    print("\n📊 Train/Val/Test split...")
    train, val, test = split_train_val_test(df)
    print(f"  Train ({train['date'].min().date()}–{train['date'].max().date()}): "
          f"{len(train):,} samples, {train['label'].sum()} positive")
    print(f"  Val   ({val['date'].min().date()}–{val['date'].max().date()}): "
          f"{len(val):,} samples, {val['label'].sum()} positive")
    print(f"  Test  ({test['date'].min().date()}–{test['date'].max().date()}): "
          f"{len(test):,} samples, {test['label'].sum()} positive")

    print("\n🏋️  Training GradientBoosting...")
    model = train_model(train)

    print("\n📊 Basic metrics:")
    for label, dset in [("Train", train), ("Val", val), ("Test", test)]:
        scored = score(dset, model)
        evaluate_basic(label, scored)

    # Detailed rally capture on test set
    test_scored = score(test, model)
    test_rallies = rallies[rallies["trough_date"] >= "2024-01-01"]
    print(f"\n📡 Test-set rally capture ({len(test_rallies)} rallies in 2024+):")
    for threshold in [0.3, 0.5, 0.7]:
        evaluate_rally_capture(test_scored, test_rallies,
                              score_threshold=threshold)

    # MU case study (also include val)
    val_scored = score(val, model)
    combined = pd.concat([val_scored, test_scored], ignore_index=True)
    mu_case_study(combined, rallies)

    feature_importance(model)

    print("\n" + "=" * 60)
    print("💡 Reading the results")
    print("=" * 60)
    print("• AUC > 0.6 = model has SOME signal (random = 0.5)")
    print("• Test rally capture % > 50% = decent recall")
    print("• Median progress at catch < 50% = catching early")
    print("• MU 2023+ rally caught at progress < 50% = your $100 MU goal met")


if __name__ == "__main__":
    main()
