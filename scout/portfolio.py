"""Monitor user's actual portfolio positions.

Reads positions from results/positions.json (broker-agnostic schema).
For each position, computes:
- Current price + 1d/5d/30d change
- Unrealized P/L in $ and %
- Days held
- Cost basis vs current value
- The full chart / signal / recommendation analysis from the watchlist module

Then generates a markdown report with portfolio summary at the top.

Schema (results/positions.json):
{
  "positions": [
    {
      "ticker": "NVDA",
      "shares": 10,
      "avg_cost": 180.50,
      "purchase_date": "2024-03-15",
      "notes": "long-term AI bet"
    },
    ...
  ],
  "_meta": {
    "last_updated": "2026-06-21",
    "source": "manual"
  }
}

This works for any broker — just edit the JSON when you buy/sell. For
Firstrade specifically there's an unofficial sync option (see
scout/firstrade_sync.py) but it's higher risk.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime, date

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scout.chart_analysis import fetch_history, analyze_chart, signals_to_bullets, signals_to_prompt_block
from scout.sector_news import fetch_sector_news
from scout.earnings_calendar import get_next_earnings, format_earnings_summary
from scout.thesis_writer import fetch_news
from scout.watchlist import compute_live_features, make_recommendation, write_recommendation_thesis
from scout.macro_context import get_macro_brief


POSITIONS_PATH = PROJECT_ROOT / "results" / "positions.json"
DEFAULT_OUT = PROJECT_ROOT / "results" / "daily" / f"portfolio_{date.today()}.md"

POSITIONS_TEMPLATE = {
    "positions": [
        {
            "ticker": "NVDA",
            "shares": 10,
            "avg_cost": 180.0,
            "purchase_date": "2024-03-15",
            "notes": "(填入你的買進原因 / 想法)",
        }
    ],
    "_meta": {
        "last_updated": str(date.today()),
        "source": "manual",
        "_help": (
            "Edit this file when you buy or sell. Shares can be fractional. "
            "avg_cost should be your average cost basis per share including fees. "
            "purchase_date is for days-held tracking. notes is free text."
        ),
    },
}


def load_positions() -> list[dict]:
    if not POSITIONS_PATH.exists():
        return []
    try:
        data = json.loads(POSITIONS_PATH.read_text())
        return data.get("positions", [])
    except Exception as e:
        print(f"⚠️  positions.json parse error: {e}")
        return []


def init_positions_file():
    """Create a sample positions.json if it doesn't exist yet."""
    if POSITIONS_PATH.exists():
        print(f"✓ Already exists: {POSITIONS_PATH}")
        return
    POSITIONS_PATH.parent.mkdir(exist_ok=True, parents=True)
    POSITIONS_PATH.write_text(
        json.dumps(POSITIONS_TEMPLATE, indent=2, ensure_ascii=False))
    print(f"✅ Created template: {POSITIONS_PATH}")
    print("   Edit it with your real holdings and run again.")


def compute_position_stats(pos: dict, history: pd.DataFrame) -> dict:
    """All the numeric stats for one position."""
    current_price = float(history["Close"].iloc[-1])
    shares = float(pos["shares"])
    avg_cost = float(pos["avg_cost"])
    current_value = current_price * shares
    cost_basis = avg_cost * shares
    pl_dollar = current_value - cost_basis
    pl_pct = (current_price / avg_cost - 1) * 100 if avg_cost else 0

    # 1d/5d/30d change
    close = history["Close"]
    try:
        change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
        change_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        change_30d = (close.iloc[-1] / close.iloc[-22] - 1) * 100
    except IndexError:
        change_1d = change_5d = change_30d = 0

    days_held = None
    try:
        pd_date = pd.Timestamp(pos.get("purchase_date", ""))
        if pd.notna(pd_date):
            days_held = (pd.Timestamp.now() - pd_date).days
    except Exception:
        pass

    return {
        "ticker": pos["ticker"],
        "shares": shares,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "current_value": current_value,
        "cost_basis": cost_basis,
        "pl_dollar": pl_dollar,
        "pl_pct": pl_pct,
        "days_held": days_held,
        "change_1d": change_1d,
        "change_5d": change_5d,
        "change_30d": change_30d,
        "notes": pos.get("notes", ""),
        "purchase_date": pos.get("purchase_date", ""),
    }


def render_portfolio_summary(stats_list: list[dict]) -> str:
    """Total portfolio value, total P/L, biggest winners/losers."""
    if not stats_list:
        return "_(no positions)_"

    total_value = sum(s["current_value"] for s in stats_list)
    total_cost = sum(s["cost_basis"] for s in stats_list)
    total_pl = total_value - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost else 0

    sorted_pl = sorted(stats_list, key=lambda x: x["pl_pct"], reverse=True)
    best = sorted_pl[0] if sorted_pl else None
    worst = sorted_pl[-1] if len(sorted_pl) > 1 else None

    lines = []
    lines.append(f"**總部位**: {len(stats_list)} 檔")
    lines.append(f"**現值**: ${total_value:,.0f}  ")
    lines.append(f"**成本**: ${total_cost:,.0f}  ")
    pl_sign = "+" if total_pl >= 0 else ""
    lines.append(f"**未實現損益**: {pl_sign}${total_pl:,.0f}  "
                 f"({pl_sign}{total_pl_pct:.1f}%)")
    if best:
        lines.append(f"\n**最佳**: {best['ticker']} {best['pl_pct']:+.1f}% "
                     f"({best['pl_dollar']:+,.0f}$)")
    if worst and worst != best:
        lines.append(f"**最差**: {worst['ticker']} {worst['pl_pct']:+.1f}% "
                     f"({worst['pl_dollar']:+,.0f}$)")
    return "\n".join(lines)


def render_position_card(stats: dict, rec: dict, chart_bullets: list[str],
                        rationale: str, ticker_news: list[str],
                        sector_etf: str, sector_news: list[str],
                        earnings_summary: str, chart_block: str) -> list[str]:
    """Markdown for a single position."""
    pl_emoji = "🟢" if stats["pl_pct"] >= 0 else "🔴"
    lines = []
    lines.append(f"## {stats['ticker']}  {rec['emoji']} **{rec['label']}**  "
                 f"({pl_emoji} {stats['pl_pct']:+.1f}%)")

    # Position facts
    lines.append(f"**現價**: ${stats['current_price']:.2f}  "
                 f"|  **持股**: {stats['shares']:g}  "
                 f"|  **成本**: ${stats['avg_cost']:.2f}  ")
    lines.append(f"**現值**: ${stats['current_value']:,.0f}  "
                 f"|  **損益**: {stats['pl_dollar']:+,.0f}$  ")
    held = f"{stats['days_held']} 天" if stats["days_held"] is not None else "未填"
    lines.append(f"**買進日**: {stats['purchase_date']} ({held}前)  ")
    lines.append(f"**1d**: {stats['change_1d']:+.2f}%  "
                 f"|  **5d**: {stats['change_5d']:+.2f}%  "
                 f"|  **30d**: {stats['change_30d']:+.2f}%  ")
    if stats["notes"]:
        lines.append(f"_買進原因_: {stats['notes']}  ")
    lines.append(f"**Net signal score**: {rec['net_score']:+d}")
    lines.append("")

    # Signals
    lines.append("### 多頭訊號")
    if rec["bullish_signals"]:
        for s, w in rec["bullish_signals"]:
            lines.append(f"- 🟢 {s} _(weight {w})_")
    else:
        lines.append("- (無)")
    lines.append("")
    lines.append("### 空頭訊號")
    if rec["bearish_signals"]:
        for s, w in rec["bearish_signals"]:
            lines.append(f"- 🔴 {s} _(weight {w})_")
    else:
        lines.append("- (無)")
    lines.append("")
    lines.append("### 圖表型態信號")
    if chart_bullets:
        for b in chart_bullets:
            lines.append(f"- {b}")
    else:
        lines.append("- (無顯著訊號)")
    lines.append("")

    if rationale:
        lines.append("### 🤖 LLM 分析師建議")
        lines.append(rationale)
        lines.append("")

    if ticker_news:
        lines.append("### 個股新聞")
        for h in ticker_news:
            lines.append(f"- {h}")
        lines.append("")
    if sector_news:
        lines.append(f"### 產業 {sector_etf} 新聞")
        for h in sector_news:
            lines.append(f"- {h}")
        lines.append("")
    if earnings_summary and not earnings_summary.startswith("("):
        lines.append("### 財報日期")
        lines.append(earnings_summary)
        lines.append("")
    lines.append("### 圖表型態原始數據")
    lines.append("```")
    lines.append(chart_block)
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def analyze_portfolio(out_path: Path, use_llm: bool = True):
    positions = load_positions()
    if not positions:
        print(f"⚠️  No positions found. Run with --init to create template.")
        return

    print(f"📡 Fetching SPY benchmark...")
    spy_hist = fetch_history("SPY", days=400)

    macro_data = {"brief": "", "indicators_formatted": ""}
    if use_llm:
        print("🌍 Fetching macro context...")
        macro_data = get_macro_brief()

    all_stats = []
    all_blocks = []

    for i, pos in enumerate(positions, start=1):
        ticker = pos["ticker"]
        print(f"\n📊 [{i}/{len(positions)}] Analyzing position: {ticker}...")
        hist = fetch_history(ticker, days=400)
        if hist.empty:
            print(f"  ⚠️  no data for {ticker}, skipping")
            continue

        stats = compute_position_stats(pos, hist)
        all_stats.append(stats)

        features = compute_live_features(ticker, hist, spy_hist)
        chart = analyze_chart(ticker, hist)
        rec = make_recommendation(features, chart)
        chart_bullets = signals_to_bullets(chart)
        chart_block = signals_to_prompt_block(chart)

        ticker_news = fetch_news(ticker, max_items=5)
        sector_etf, sector_news = fetch_sector_news(ticker, max_items=4)
        earnings_info = get_next_earnings(ticker)
        earnings_summary = format_earnings_summary(earnings_info)

        rationale = ""
        if use_llm:
            print(f"   Writing recommendation rationale ({rec['label']})...")
            rationale = write_recommendation_thesis(
                ticker, features, chart_block, rec, ticker_news,
                sector_etf, sector_news, earnings_summary,
                macro_data.get("brief", ""),
                macro_data.get("indicators_formatted", ""),
            )

        print(f"   ${stats['current_price']:.2f}  "
              f"P/L {stats['pl_pct']:+.1f}%  "
              f"{rec['emoji']} {rec['label']}")

        all_blocks.append(render_position_card(
            stats, rec, chart_bullets, rationale, ticker_news,
            sector_etf, sector_news, earnings_summary, chart_block,
        ))

    # Compose markdown
    lines = []
    lines.append(f"# 📊 Portfolio Monitor — {date.today()}")
    lines.append("")
    lines.append("⚠️ **以下為基於公開資料的技術分析，非投資建議。任何進出皆為使用者自行決定。**")
    lines.append("")
    lines.append("## 投組總覽")
    lines.append(render_portfolio_summary(all_stats))
    lines.append("")
    if macro_data.get("brief") and not macro_data["brief"].startswith("("):
        lines.append("## 🌍 當前總體環境")
        if macro_data.get("indicators_formatted"):
            lines.append("```")
            lines.append(macro_data["indicators_formatted"])
            lines.append("```")
        lines.append(macro_data["brief"])
        lines.append("")
    lines.append("---")
    lines.append("")
    for block in all_blocks:
        lines.extend(block)

    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text("\n".join(lines))
    print(f"\n💾 Portfolio report saved → {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", action="store_true",
                        help="Create a template positions.json and exit")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    if args.init:
        init_positions_file()
        return

    out_path = Path(args.out) if args.out else DEFAULT_OUT
    analyze_portfolio(out_path, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
