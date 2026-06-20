"""Generate top picks + LLM-written investment thesis for each.

This is the "agent" deliverable: 1-5 stocks ranked by rally probability,
each with a full reasoning written by Groq Llama 3.3 70B.

Usage:
    ./run.sh scout-agent                  # default top 5
    ./run.sh scout-agent --top_k 3
    ./run.sh scout-agent --as_of 2024-01-15
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scout.validate import load_split, train_one, FEATURE_COLS
from scout.benchmark_cases import explain_features
from scout.thesis_writer import write_thesis
from scout.macro_context import get_macro_brief


def filter_depressed(latest):
    """Keep only picks with multiple 'depressed bottom' signals.

    A truly undiscovered MU/LITE-style candidate should have at least
    2 of these 5 conditions:
      - Drawdown from 52w high  < -20%
      - RSI < 45 (not yet rallying hard)
      - Price below MA200 by ≥ 5%
      - Underperforming SPY in last 3 months
      - Negative 3m or 6m momentum (capitulation)

    This filters out 'already-rallying continuation' picks (like CIEN
    above MA200 by 106%) that aren't actually 'undiscovered'.
    """
    latest = latest.copy()
    deep_dd = latest["drawdown_52w"] < -0.20
    cool_rsi = latest["rsi_14"] < 45
    below_ma = latest["price_vs_ma200"] < -0.05
    underperf = latest["rs_vs_spy_3m"] < 0.0
    capitulation = (latest["momentum_3m"] < -0.10) | (latest["momentum_6m"] < -0.15)

    n_signals = (deep_dd.astype(int) + cool_rsi.astype(int) + below_ma.astype(int)
                 + underperf.astype(int) + capitulation.astype(int))
    latest["n_depressed_signals"] = n_signals
    return latest[n_signals >= 2].copy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--as_of", default=None)
    parser.add_argument("--out", default="results/agent_report.md")
    parser.add_argument("--mode", default="discover",
                        choices=["discover", "raw"],
                        help="discover = only depressed-bottom candidates (default), "
                             "raw = no filter (top scores regardless of mode)")
    parser.add_argument("--no_thesis", action="store_true",
                        help="skip LLM thesis (debug)")
    args = parser.parse_args()

    print("📊 Loading data + training production model...")
    df, train, val, test = load_split()
    train_combined = pd.concat([train, val], ignore_index=True)
    model = train_one(train_combined, seed=1)

    as_of = pd.Timestamp(args.as_of) if args.as_of else df["date"].max()
    print(f"📅 Evaluating as of {as_of.date()}")

    recent = df[df["date"] <= as_of].copy()
    latest = recent.sort_values("date").groupby("ticker").tail(1)
    latest["score"] = model.predict_proba(latest[FEATURE_COLS].values)[:, 1]

    universe_median = float(latest["score"].median())
    universe_p95 = float(latest["score"].quantile(0.95))

    if args.mode == "discover":
        filtered = filter_depressed(latest)
        n_filtered = len(filtered)
        top = filtered.sort_values("score", ascending=False).head(args.top_k)
        print(f"\n🔍 Mode: discover (depressed-bottom candidates only)")
        print(f"   Universe = {len(latest)} → after filter = {n_filtered} candidates")
    else:
        top = latest.sort_values("score", ascending=False).head(args.top_k)
        print(f"\n🎯 Mode: raw (no filter)")

    print(f"\n🎯 Top {args.top_k} picks:")
    for _, r in top.iterrows():
        n_sigs = r.get("n_depressed_signals", "-")
        print(f"  {r['ticker']:6s}  score={r['score']:.4f}  "
              f"${r['close']:>7.2f}  signals={n_sigs}")
    print(f"\n  Universe score: median={universe_median:.4f}  p95={universe_p95:.4f}")

    # Fetch macro brief once (cached for 1h)
    if args.no_thesis:
        macro_data = {"brief": "", "sources_used": []}
    else:
        print("\n🌍 Fetching macro context...")
        macro_data = get_macro_brief()
        print(f"   ({len(macro_data.get('brief', ''))} chars from "
              f"{', '.join(macro_data.get('sources_used', []))})")
    macro_brief = macro_data.get("brief", "")

    # Markdown report
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(exist_ok=True, parents=True)
    lines = []
    lines.append(f"# 🦞 Scout Agent Report — {as_of.date()}")
    lines.append("")
    lines.append(f"Top {args.top_k} undiscovered rally candidates ranked by model score.")
    lines.append(f"Generated: {datetime.now():%Y-%m-%d %H:%M}  ")
    lines.append(f"Model: GradientBoostingClassifier (Test AUC 0.697 ± 0.003)  ")
    lines.append(f"Thesis: Groq Llama 3.3 70B with yfinance ticker + macro news context")
    lines.append("")
    lines.append("**📊 Score distribution context**")
    lines.append(f"- Median (universe): {universe_median:.4f}")
    lines.append(f"- 95th percentile: {universe_p95:.4f}")
    lines.append("")
    if macro_brief and not macro_brief.startswith("("):
        lines.append("## 🌍 當前總體環境")
        lines.append(f"_(來源: SPY / QQQ / SMH / TLT 新聞 → Groq summary, "
                     f"cached 1h)_")
        lines.append("")
        lines.append(macro_brief)
        lines.append("")
    lines.append("---")
    lines.append("")

    for rank, (_, row) in enumerate(top.iterrows(), start=1):
        ticker = row["ticker"]
        score = float(row["score"])
        price = float(row["close"])

        print(f"\n📝 [{rank}/{args.top_k}] Writing thesis for {ticker}...")

        bullets = explain_features(row)
        bullet_text = "\n".join(f"- {b}" for b in bullets) if bullets else "- 中性訊號組合"

        if args.no_thesis:
            thesis = "_(LLM thesis skipped — use without --no_thesis to generate)_"
        else:
            t0 = time.time()
            features_dict = {
                "date_str": str(row["date"].date()),
                "close": price,
                **{c: float(row[c]) for c in FEATURE_COLS},
            }
            thesis = write_thesis(ticker, features_dict, score,
                                  macro_brief=macro_brief)
            elapsed = time.time() - t0
            print(f"   ✅ done in {elapsed:.1f}s")

        lines.append(f"## #{rank}  {ticker} — score {score:.4f}")
        lines.append(f"**Price**: ${price:.2f}  ")
        lines.append("")
        lines.append("### 技術信號")
        lines.append(bullet_text)
        lines.append("")
        lines.append("### 投資論點 (LLM-generated)")
        lines.append(thesis)
        lines.append("")
        lines.append("### TODO before buying")
        lines.append("- [ ] 開官網 / Yahoo Finance 確認最新財報")
        lines.append("- [ ] 看 sector 同業表現")
        lines.append("- [ ] 看下一次 earnings 是哪天")
        lines.append("- [ ] 過去 12 個月最大跌幅是否能接受")
        lines.append("- [ ] 部位大小（建議單檔不超過 portfolio 5%）")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.write_text("\n".join(lines))
    print(f"\n💾 Agent report saved → {out_path}")
    print(f"   ({len(lines)} lines, ~{sum(len(l) for l in lines)} chars)")


if __name__ == "__main__":
    main()
