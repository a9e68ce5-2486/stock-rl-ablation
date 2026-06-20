"""Generate today's screening output: top picks with reasoning.

Trains the model on ALL available training data (2015-2023), then ranks
every ticker by its current feature score. Outputs top K picks with
feature-based reasoning bullets.

This is the actual deliverable — your morning briefing.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scout.validate import (
    load_split, train_one, FEATURE_COLS,
)
from scout.benchmark_cases import explain_features


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=10,
                        help="how many picks to show")
    parser.add_argument("--as_of", default=None,
                        help="date to evaluate (default: latest feature date)")
    parser.add_argument("--out", default="results/picks_latest.md",
                        help="markdown output path")
    args = parser.parse_args()

    print("📊 Loading data...")
    df, train, val, test = load_split()

    # Train on train+val (everything < 2024)
    train_combined = pd.concat([train, val], ignore_index=True)
    print(f"🏋️  Training production model on {len(train_combined):,} samples...")
    model = train_one(train_combined, seed=1)

    # Choose evaluation date
    if args.as_of:
        as_of = pd.Timestamp(args.as_of)
    else:
        as_of = df["date"].max()
    print(f"📅 Evaluating as of {as_of.date()}")

    # For each ticker, get latest feature row up to as_of
    recent = df[df["date"] <= as_of].copy()
    latest = recent.sort_values("date").groupby("ticker").tail(1)

    # Predict
    X = latest[FEATURE_COLS].values
    latest["score"] = model.predict_proba(X)[:, 1]

    # Rank
    top = latest.sort_values("score", ascending=False).head(args.top_k)

    # Write markdown
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True, parents=True)
    lines = []
    lines.append(f"# 🦞 Scout Picks — {as_of.date()}")
    lines.append("")
    lines.append(f"Top {args.top_k} candidates ranked by rally-start probability.")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"Model: GradientBoostingClassifier, trained 2016–2023, AUC≈0.69")
    lines.append("")
    lines.append("**Reading**: higher score = more rally-start-like features. ")
    lines.append("This is a *candidate list*, not a buy signal — research each pick.")
    lines.append("")

    print(f"\n{'='*60}")
    print(f"🎯 Top {args.top_k} picks as of {as_of.date()}")
    print(f"{'='*60}")

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ticker = row["ticker"]
        score = row["score"]
        price = row["close"]
        bullets = explain_features(row)
        bullet_text = "\n".join(f"- {b}" for b in bullets) if bullets else "- (中性訊號組合)"

        # Console
        print(f"\n  #{rank}  {ticker:6s}  score={score:.4f}  price=${price:>7.2f}")
        for b in bullets:
            print(f"     • {b}")

        # Markdown
        lines.append(f"## #{rank}  {ticker} — score {score:.4f}")
        lines.append(f"**Price**: ${price:.2f}  ")
        lines.append("")
        lines.append("### 信號")
        lines.append(bullet_text)
        lines.append("")
        lines.append("### Raw feature snapshot")
        lines.append("```")
        for f in FEATURE_COLS:
            v = row[f]
            if f in ("drawdown_52w", "price_vs_ma200",
                     "momentum_3m", "momentum_6m", "momentum_12m",
                     "rs_vs_spy_3m"):
                lines.append(f"  {f:<22} = {v*100:+6.1f}%")
            elif f in ("vol_60d_rank",):
                lines.append(f"  {f:<22} = {v:.2f}")
            elif f == "rsi_14":
                lines.append(f"  {f:<22} = {v:>6.1f}")
            elif f == "vol_60d":
                lines.append(f"  {f:<22} = {v*100:>5.1f}% ann")
            elif f == "volume_surge_30d":
                lines.append(f"  {f:<22} = {v:>5.2f}x")
            elif f == "days_since_high":
                lines.append(f"  {f:<22} = {int(v)}")
            else:
                lines.append(f"  {f:<22} = {v}")
        lines.append("```")
        lines.append("")
        lines.append("### TODO before buying")
        lines.append("- [ ] Check fundamentals (revenue growth, FCF)")
        lines.append("- [ ] Recent news / catalyst")
        lines.append("- [ ] Sector momentum")
        lines.append("- [ ] Earnings date proximity")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Score distribution for context
    p95 = latest["score"].quantile(0.95)
    median = latest["score"].quantile(0.5)
    lines.append(f"\n## Score distribution context")
    lines.append(f"- 95th percentile: {p95:.4f}")
    lines.append(f"- Median: {median:.4f}")
    lines.append(f"- Min top pick: {top['score'].min():.4f}")
    print(f"\n  📊 Universe score percentiles: median={median:.4f} p95={p95:.4f}")

    out_path.write_text("\n".join(lines))
    print(f"\n💾 Markdown report saved → {out_path}")


if __name__ == "__main__":
    main()
