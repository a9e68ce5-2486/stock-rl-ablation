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
from scout.dividend_info import dividend_snapshot
from scout.events import get_events, events_to_bullets
from scout.analyst_targets import get_targets_snapshot, render_targets_bullets
from scout.sector_news import get_sector_etf
from scout.portfolio_log import append_daily_snapshot
from scout.indices import get_all_indices, render_indices_table_md


def get_twd_to_usd_rate() -> float:
    """yfinance TWD=X returns TWD per USD (e.g. 30.58). Return inverse for TWD→USD."""
    try:
        import yfinance as yf
        hist = yf.Ticker("TWD=X").history(period="5d")
        twd_per_usd = float(hist["Close"].iloc[-1])
        return 1 / twd_per_usd if twd_per_usd else 0.0327
    except Exception:
        return 0.0327  # fallback ≈ 1 USD = 30.58 TWD


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


def load_positions() -> tuple[list[dict], float, float]:
    """Returns (positions, cash_usd, cash_twd)."""
    if not POSITIONS_PATH.exists():
        return [], 0.0, 0.0
    try:
        data = json.loads(POSITIONS_PATH.read_text())
        return (data.get("positions", []),
                float(data.get("cash_usd", 0)),
                float(data.get("cash_twd", 0)))
    except Exception as e:
        print(f"⚠️  positions.json parse error: {e}")
        return [], 0.0, 0.0


def aggregate_by_ticker(positions: list[dict]) -> list[dict]:
    """Combine multiple lots of same ticker. Preserves currency."""
    by_t = {}
    for p in positions:
        t = p["ticker"]
        if t not in by_t:
            by_t[t] = {
                "ticker": t,
                "shares": 0.0,
                "total_cost": 0.0,
                "lots": [],
                "currency": p.get("currency", "USD"),
            }
        agg = by_t[t]
        s = float(p.get("shares", 0))
        c = float(p.get("avg_cost", 0))
        agg["shares"] += s
        agg["total_cost"] += s * c
        agg["lots"].append(p)

    out = []
    for agg in by_t.values():
        agg["avg_cost"] = (agg["total_cost"] / agg["shares"]) if agg["shares"] else 0
        notes = [p.get("notes", "") for p in agg["lots"] if p.get("notes")]
        agg["notes"] = " | ".join(notes) if notes else ""
        out.append(agg)
    return out


def _dividends_for_lot(divs_series: pd.Series, purchase_date: str | None,
                       shares: float) -> tuple[float, int]:
    """For one lot, compute dividends received since its purchase_date."""
    if not purchase_date or shares <= 0 or divs_series is None or divs_series.empty:
        return 0.0, 0
    start = pd.Timestamp(purchase_date)
    if divs_series.index.tz is not None:
        start = start.tz_localize(divs_series.index.tz)
    relevant = divs_series[divs_series.index >= start]
    if relevant.empty:
        return 0.0, 0
    return float(relevant.sum()) * shares, len(relevant)


def compute_dividends_per_lots(ticker: str, lots: list[dict]) -> tuple[float, int]:
    """Sum dividends per individual lot (correct when lots have different
    purchase_dates). Fetches the dividend series once per ticker.

    Returns (total_dividends_dollar, total_payment_count).
    """
    try:
        import yfinance as yf
        divs = yf.Ticker(ticker).dividends
    except Exception:
        divs = None
    total = 0.0
    count = 0
    for lot in lots:
        s = float(lot.get("shares", 0))
        d = lot.get("purchase_date")
        amt, n = _dividends_for_lot(divs, d, s)
        total += amt
        count += n
    return total, count


def compute_dividends(ticker: str, purchase_date: str | None,
                      shares: float) -> tuple[float, int]:
    """Legacy single-lot helper (kept for backward compat).

    Prefer compute_dividends_per_lots() when you have multiple lots.
    """
    try:
        import yfinance as yf
        divs = yf.Ticker(ticker).dividends
    except Exception:
        return 0.0, 0
    return _dividends_for_lot(divs, purchase_date, shares)


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

    # Latest dividend snapshot (forward-looking — doesn't need purchase_date)
    div_snap = dividend_snapshot(pos["ticker"], shares=shares,
                                  current_price=current_price)

    # Analyst targets + fair value snapshot
    sector_etf = get_sector_etf(pos["ticker"]) or ""
    targets_snap = get_targets_snapshot(pos["ticker"], current_price, sector_etf)

    # Estimated annual dividend total (latest per-share × implied annual freq × shares)
    latest = div_snap.get("latest", {})
    freq = div_snap.get("frequency", {})
    est_annual_per_share = (latest.get("per_share", 0)
                            * freq.get("implied_per_year", 0))
    est_annual_total = est_annual_per_share * shares

    return {
        "ticker": pos["ticker"],
        "shares": shares,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "current_value": current_value,
        "cost_basis": cost_basis,
        "pl_dollar": pl_dollar,
        "pl_pct": pl_pct,
        "est_annual_dividend": est_annual_total,
        "est_annual_per_share": est_annual_per_share,
        "change_1d": change_1d,
        "change_5d": change_5d,
        "change_30d": change_30d,
        "notes": pos.get("notes", ""),
        "lots": pos.get("lots", []),
        "div_snapshot": div_snap,
        "targets": targets_snap,
        "currency": pos.get("currency", "USD"),
    }


def render_portfolio_summary(stats_list: list[dict], cash_usd: float = 0,
                              cash_twd: float = 0,
                              twd_to_usd: float = 0.0327) -> str:
    """Total portfolio value grouped by currency, combined in USD."""
    if not stats_list and cash_usd <= 0 and cash_twd <= 0:
        return "_(no positions)_"

    # Group by currency
    by_ccy = {}
    for s in stats_list:
        ccy = s.get("currency", "USD")
        if ccy not in by_ccy:
            by_ccy[ccy] = {"positions": [], "equity": 0, "cost": 0, "annual_div": 0}
        b = by_ccy[ccy]
        b["positions"].append(s)
        b["equity"] += s["current_value"]
        b["cost"] += s["cost_basis"]
        b["annual_div"] += s["est_annual_dividend"]

    # Add cash buckets
    if cash_usd > 0:
        by_ccy.setdefault("USD", {"positions": [], "equity": 0, "cost": 0, "annual_div": 0})
    if cash_twd > 0:
        by_ccy.setdefault("TWD", {"positions": [], "equity": 0, "cost": 0, "annual_div": 0})

    lines = []
    lines.append(f"**總部位**: {len(stats_list)} 檔 "
                 f"(美股 {sum(1 for s in stats_list if s.get('currency','USD')=='USD')} + "
                 f"台股 {sum(1 for s in stats_list if s.get('currency')=='TWD')})")
    lines.append(f"**匯率**: 1 USD = {1/twd_to_usd:.2f} TWD")
    lines.append("")

    grand_total_usd = 0
    grand_cost_usd = 0
    for ccy in ("USD", "TWD"):
        if ccy not in by_ccy:
            continue
        b = by_ccy[ccy]
        if ccy == "USD":
            cash = cash_usd
            sym = "$"
            equity_usd = b["equity"]
            cost_usd = b["cost"]
            cash_usd_eq = cash_usd
        else:
            cash = cash_twd
            sym = "NT$"
            equity_usd = b["equity"] * twd_to_usd
            cost_usd = b["cost"] * twd_to_usd
            cash_usd_eq = cash_twd * twd_to_usd
        total_value = b["equity"] + cash
        pl = b["equity"] - b["cost"]
        pl_pct = (pl / b["cost"] * 100) if b["cost"] else 0
        pl_s = "+" if pl >= 0 else ""
        lines.append(f"### {('🇺🇸 美股' if ccy == 'USD' else '🇹🇼 台股')} ({ccy})")
        lines.append(f"- **部位數**: {len(b['positions'])} 檔 + 現金 {sym}{cash:,.2f}")
        lines.append(f"- **現值**: {sym}{total_value:,.2f}  "
                     f"(股票 {sym}{b['equity']:,.2f} + 現金 {sym}{cash:,.2f})")
        lines.append(f"- **成本**: {sym}{b['cost']:,.2f}")
        lines.append(f"- **未實現損益**: {pl_s}{sym}{pl:,.2f} ({pl_s}{pl_pct:.2f}%)")
        if ccy == "TWD":
            lines.append(f"- **USD 等值**: ${equity_usd + cash_usd_eq:,.2f}  "
                         f"(成本 ${cost_usd:,.2f})")
        if b["annual_div"] > 0:
            lines.append(f"- **預估年配息**: {sym}{b['annual_div']:,.2f}")
        lines.append("")
        grand_total_usd += equity_usd + cash_usd_eq
        grand_cost_usd += cost_usd

    grand_pl = grand_total_usd - grand_cost_usd
    grand_pl_pct = (grand_pl / grand_cost_usd * 100) if grand_cost_usd else 0
    pl_s = "+" if grand_pl >= 0 else ""
    lines.append(f"### 💰 總計（USD）")
    lines.append(f"- **總現值**: ${grand_total_usd:,.2f}")
    lines.append(f"- **總成本**: ${grand_cost_usd:,.2f}")
    lines.append(f"- **總損益**: {pl_s}${grand_pl:,.2f} ({pl_s}{grand_pl_pct:.2f}%)")

    return "\n".join(lines)


def render_position_card(stats: dict, rec: dict, chart_bullets: list[str],
                        rationale: str, ticker_news: list[str],
                        sector_etf: str, sector_news: list[str],
                        earnings_summary: str, chart_block: str) -> list[str]:
    """Markdown for a single position."""
    pl_emoji = "🟢" if stats["pl_pct"] >= 0 else "🔴"
    ccy = stats.get("currency", "USD")
    sym = "NT$" if ccy == "TWD" else "$"
    flag = "🇹🇼" if ccy == "TWD" else "🇺🇸"
    lines = []
    lines.append(f"## {flag} {stats['ticker']}  {rec['emoji']} **{rec['label']}**  "
                 f"({pl_emoji} {stats['pl_pct']:+.1f}%)")

    # Position facts
    lines.append(f"**現價**: {sym}{stats['current_price']:.2f}  "
                 f"|  **持股**: {stats['shares']:g}  "
                 f"|  **加權均價**: {sym}{stats['avg_cost']:.2f}  ")
    lines.append(f"**現值**: {sym}{stats['current_value']:,.2f}  "
                 f"|  **價差損益**: {stats['pl_dollar']:+,.2f} {sym} "
                 f"({stats['pl_pct']:+.1f}%)  ")
    # 配息資訊（最近一次 + 預估年配息）
    snap = stats.get("div_snapshot", {})
    latest = snap.get("latest", {})
    freq = snap.get("frequency", {})
    yld = snap.get("yield", {})
    if latest:
        lines.append("")
        lines.append("### 💰 配息資訊")
        lines.append(f"- **最近配息日**: {latest['date']} ({latest['days_ago']} 天前)")
        lines.append(f"- **每股配息（最近）**: ${latest['per_share']:.4f}")
        lines.append(f"- **本次入帳**: ${snap.get('latest_total_dollar', 0):.2f}")
        lines.append(f"- **配息頻率**: {freq.get('label', '未知')} "
                     f"(中位間隔 {freq.get('median_days') or '—'} 天, "
                     f"年約 {freq.get('implied_per_year', 0)} 次)")
        lines.append(f"- **預估年配息（總）**: "
                     f"${stats.get('est_annual_dividend', 0):,.2f} "
                     f"(每股 ${stats.get('est_annual_per_share', 0):.4f})")
        if yld:
            lines.append(f"- **年化殖利率**: {yld['current_yield_pct']:.2f}%")
        changes = snap.get("policy_changes", [])
        if changes:
            recent_changes = changes[:3]
            lines.append(f"- **配息政策變化**（近 {len(recent_changes)} 次）:")
            for c in recent_changes:
                emoji = "📈" if c["change_pct"] > 0 else "📉"
                lines.append(f"  - {emoji} {c['date']}: ${c['from_per_share']:.4f} → "
                             f"${c['to_per_share']:.4f} ({c['change_pct']:+.1f}%)")
    lines.append(f"**1d**: {stats['change_1d']:+.2f}%  "
                 f"|  **5d**: {stats['change_5d']:+.2f}%  "
                 f"|  **30d**: {stats['change_30d']:+.2f}%  ")

    # Lots breakdown
    lots = stats.get("lots", [])
    if len(lots) > 1:
        lines.append("")
        lines.append("**持股分布**:")
        lines.append("")
        lines.append("| Broker | 股數 | 均價 | Cost basis |")
        lines.append("|--------|------|------|------------|")
        for lot in lots:
            shares = float(lot.get("shares", 0))
            cost = float(lot.get("avg_cost", 0))
            broker = lot.get("broker", "—")
            basis = shares * cost
            lines.append(f"| {broker} | {shares:g} | ${cost:.2f} | ${basis:,.2f} |")
        lines.append(f"| **合計** | **{stats['shares']:g}** | "
                     f"**${stats['avg_cost']:.2f}** | "
                     f"**${stats['cost_basis']:,.2f}** |")

    if stats["notes"]:
        lines.append(f"\n_買進原因_: {stats['notes']}  ")
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

    # NEW: Analyst targets + Fair Value section
    targets = stats.get("targets", {})
    if targets:
        target_bullets = render_targets_bullets(targets)
        if target_bullets:
            lines.append("")
            lines.append("### 🎯 目標價 & Fair Value (分析師 + naive 估值)")
            for b in target_bullets:
                lines.append(f"- {b}")
            lines.append("")

    # Material events timeline
    events = get_events(stats["ticker"])
    if events:
        bullets = events_to_bullets(events)
        if bullets:
            lines.append("")
            lines.append("### 📅 大事件 (近期)")
            for b in bullets:
                lines.append(f"- {b}")
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
    raw_positions, cash_usd, cash_twd = load_positions()
    if not raw_positions and cash_usd <= 0 and cash_twd <= 0:
        print(f"⚠️  No positions found. Run with --init to create template.")
        return

    aggregated = aggregate_by_ticker(raw_positions)
    n_us = sum(1 for p in aggregated if p.get("currency", "USD") == "USD")
    n_tw = sum(1 for p in aggregated if p.get("currency") == "TWD")
    print(f"📂 Loaded {len(raw_positions)} lots → {len(aggregated)} unique tickers "
          f"({n_us} US + {n_tw} TW), cash ${cash_usd:.2f} USD + NT${cash_twd:.2f}")

    twd_to_usd = get_twd_to_usd_rate()
    print(f"💱 TWD→USD rate: {twd_to_usd:.6f} (1 USD = {1/twd_to_usd:.2f} TWD)")

    print(f"📡 Fetching SPY benchmark + indices...")
    spy_hist = fetch_history("SPY", days=400)
    indices = get_all_indices()

    macro_data = {"brief": "", "indicators_formatted": ""}
    if use_llm:
        print("🌍 Fetching macro context...")
        macro_data = get_macro_brief()

    all_stats = []
    all_blocks = []

    for i, pos in enumerate(aggregated, start=1):
        ticker = pos["ticker"]
        print(f"\n📊 [{i}/{len(aggregated)}] Analyzing position: {ticker} "
              f"({len(pos['lots'])} lot{'s' if len(pos['lots']) > 1 else ''})...")
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

        annual_div_str = (f"  est. annual div ${stats['est_annual_dividend']:.2f}"
                          if stats['est_annual_dividend'] > 0 else "")
        print(f"   ${stats['current_price']:.2f}  "
              f"P/L {stats['pl_pct']:+.1f}%{annual_div_str}  "
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
    lines.append(render_portfolio_summary(all_stats, cash_usd, cash_twd, twd_to_usd))
    lines.append("")
    if indices:
        lines.append("## 📈 主要指數")
        lines.append(render_indices_table_md(indices))
        lines.append("")
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

    # Append today's snapshot to Excel log (only on trading days, called from
    # run_daily.sh which already gates by is_trading_day.py)
    try:
        log_info = append_daily_snapshot(all_stats)
        print(f"📊 Excel log appended  → {log_info['log_path']}  "
              f"({log_info['rows_appended']} 列新增, 累計 {log_info['total_rows']} 列)")
    except Exception as e:
        print(f"⚠️  Excel log error: {e}")


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
