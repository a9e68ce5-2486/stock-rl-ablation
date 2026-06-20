"""Watchlist monitor: chart analysis + buy/sell/hold recommendation for known tickers.

Different from `picks_with_thesis.py` which DISCOVERS new candidates.
This module tracks tickers you ALREADY care about (e.g. NVDA, AVGO, TSM)
and gives a structured rule-based recommendation + LLM-written rationale.

Recommendation tiers (rule-based, based on net technical signals):
   BUY        — strong bullish convergence (golden cross + oversold + uptrend …)
   ACCUMULATE — mildly bullish (some buy signals, no major red flags)
   HOLD       — mixed / no clear edge
   TRIM       — mildly bearish (some sell signals, position larger than warranted)
   SELL       — strong bearish convergence (death cross + overbought + downtrend …)

⚠️ NOT financial advice. This is educational. Always do your own research.

Usage:
    ./run.sh scout-watch NVDA AVGO TSM
    ./run.sh scout-watch NVDA               # just one
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scout.chart_analysis import (
    fetch_history, analyze_chart, signals_to_bullets, signals_to_prompt_block,
)
from scout.sector_news import fetch_sector_news, get_sector_etf
from scout.earnings_calendar import get_next_earnings, format_earnings_summary
from scout.thesis_writer import fetch_news, _get_groq_key, GROQ_URL, DEFAULT_MODEL
from scout.macro_context import get_macro_brief


# ============================================================
# Live feature computation (no universe needed — works for any ticker)
# ============================================================

def compute_live_features(ticker: str, history: pd.DataFrame,
                          spy_history: pd.DataFrame) -> dict:
    """Snapshot of the same features the rally classifier uses, on a single ticker."""
    if history.empty or "Close" not in history.columns:
        return {}
    close = history["Close"].astype(float)
    volume = history.get("Volume")
    if len(close) < 252:
        return {}

    spy = spy_history["Close"].astype(float).reindex(close.index).ffill()

    last_close = float(close.iloc[-1])
    high_52w = float(close.iloc[-252:].max())
    ma200 = float(close.iloc[-200:].mean())
    ret_3m = last_close / float(close.iloc[-63]) - 1
    ret_6m = last_close / float(close.iloc[-126]) - 1
    ret_12m = last_close / float(close.iloc[-252]) - 1

    daily_ret = close.pct_change()
    vol_60d = float(daily_ret.iloc[-60:].std() * np.sqrt(252))

    spy_ret_3m = float(spy.iloc[-1]) / float(spy.iloc[-63]) - 1

    if volume is not None:
        v30 = float(volume.iloc[-30:].mean())
        v90 = float(volume.iloc[-90:].mean())
        volume_surge = v30 / (v90 + 1e-9)
    else:
        volume_surge = 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    rsi_14 = float((100 - 100 / (1 + rs)).iloc[-1])

    rolling_high_idx = close.rolling(252).apply(
        lambda x: 252 - 1 - np.argmax(x.values), raw=False)
    days_since_high = float(rolling_high_idx.iloc[-1])

    return {
        "date_str": str(close.index[-1].date()),
        "close": last_close,
        "drawdown_52w": last_close / high_52w - 1,
        "price_vs_ma200": last_close / ma200 - 1,
        "momentum_3m": ret_3m,
        "momentum_6m": ret_6m,
        "momentum_12m": ret_12m,
        "vol_60d": vol_60d,
        "volume_surge_30d": volume_surge,
        "rsi_14": rsi_14,
        "rs_vs_spy_3m": ret_3m - spy_ret_3m,
        "days_since_high": days_since_high,
    }


# ============================================================
# Rule-based recommendation
# ============================================================

def make_recommendation(features: dict, chart: dict) -> dict:
    """Score bullish vs bearish signals and emit a label.

    Each signal has a weight (1 = mild, 2 = strong).
    Net score thresholds:
       net ≥ +4 → BUY
       net ≥ +2 → ACCUMULATE
       net in [-1, +1] → HOLD
       net ≤ -2 → TRIM
       net ≤ -4 → SELL
    """
    bullish = []
    bearish = []

    # ── MACD ──
    macd = chart.get("macd", {})
    if macd:
        cross = macd.get("cross_5d", "none")
        if cross == "bullish":
            bullish.append(("MACD 5 日內黃金交叉", 2))
        elif cross == "bearish":
            bearish.append(("MACD 5 日內死亡交叉", 2))
        if macd.get("histogram_dir") == "rising" and not macd.get("above_zero"):
            bullish.append(("MACD histogram 從負轉正中", 1))

    # ── Bollinger ──
    bb = chart.get("bollinger", {})
    if bb:
        state = bb.get("state", "")
        if state == "below_lower":
            bullish.append(("跌破 BB 下緣（統計超賣）", 1))
        elif state == "above_upper":
            bearish.append(("突破 BB 上緣（統計超買）", 1))

    # ── Trend slope ──
    trend = chart.get("trend", {})
    if trend:
        t = trend.get("trend", "")
        if t == "strong uptrend":
            bullish.append((f"60d 強勁上升趨勢", 2))
        elif t == "mild uptrend":
            bullish.append((f"60d 溫和上升", 1))
        elif t == "mild downtrend":
            bearish.append((f"60d 溫和下跌", 1))
        elif t == "strong downtrend":
            bearish.append((f"60d 強勁下跌", 2))

    # ── RSI ──
    rsi = features.get("rsi_14", 50)
    if rsi < 30:
        bullish.append((f"RSI {rsi:.0f} 深度超賣", 2))
    elif rsi < 40:
        bullish.append((f"RSI {rsi:.0f} 偏低", 1))
    elif rsi > 75:
        bearish.append((f"RSI {rsi:.0f} 嚴重過熱", 2))
    elif rsi > 65:
        bearish.append((f"RSI {rsi:.0f} 偏熱", 1))

    # ── Support / resistance proximity ──
    sr = chart.get("support_resistance", {})
    if sr:
        to_60_low = sr.get("to_60d_low_pct", 0)
        to_60_high = sr.get("to_60d_high_pct", 0)
        if abs(to_60_low) < 5:
            bullish.append((f"接近 60d 低點（剩 {abs(to_60_low):.1f}%）", 1))
        if to_60_high > 0 and to_60_high < 5:
            bearish.append((f"接近 60d 高點（剩 {to_60_high:.1f}%）", 1))

    # ── Candlestick patterns ──
    candles = chart.get("candlesticks", [])
    for c in candles:
        if "hammer" in c:
            bullish.append((f"K 棒: {c}", 2))
        elif "bullish engulfing" in c:
            bullish.append((f"K 棒: {c}", 2))
        elif "bearish engulfing" in c:
            bearish.append((f"K 棒: {c}", 2))

    # ── Volume confirmation ──
    vs = features.get("volume_surge_30d", 1)
    if vs > 1.5:
        bullish.append((f"成交量 surge {vs:.2f}× (accumulation)", 1))

    # ── Net score ──
    bullish_score = sum(w for _, w in bullish)
    bearish_score = sum(w for _, w in bearish)
    net = bullish_score - bearish_score

    if net >= 4:
        label = "BUY"
        emoji = "🟢🟢"
    elif net >= 2:
        label = "ACCUMULATE"
        emoji = "🟢"
    elif net >= -1:
        label = "HOLD"
        emoji = "⚪"
    elif net >= -3:
        label = "TRIM"
        emoji = "🟡"
    else:
        label = "SELL"
        emoji = "🔴"

    return {
        "label": label,
        "emoji": emoji,
        "net_score": net,
        "bullish_signals": bullish,
        "bearish_signals": bearish,
    }


# ============================================================
# LLM rationale
# ============================================================

def write_recommendation_thesis(ticker: str, features: dict, chart_block: str,
                                rec: dict, ticker_news: list[str],
                                sector_etf: str, sector_news: list[str],
                                earnings_summary: str,
                                macro_brief: str,
                                macro_indicators: str) -> str:
    api_key = _get_groq_key()
    if not api_key:
        return "(LLM rationale unavailable — Groq key missing)"

    import requests, json

    bullish_text = "\n".join(f"  + {s} (weight {w})" for s, w in rec["bullish_signals"]) or "  (none)"
    bearish_text = "\n".join(f"  - {s} (weight {w})" for s, w in rec["bearish_signals"]) or "  (none)"

    sector_block = (f"\n**{ticker} 產業 ETF ({sector_etf}) 新聞**：\n"
                    + "\n".join(f"- {h}" for h in sector_news) + "\n"
                    if sector_news else "")
    ticker_news_block = ("\n**個股新聞**：\n" + "\n".join(f"- {h}" for h in ticker_news) + "\n"
                        if ticker_news else "")

    prompt = f"""你是嚴謹的量化技術分析師。針對 **{ticker}**（價格 ${features.get('close', 0):.2f}，截至 {features.get('date_str', 'N/A')}），請給投資建議。

**規則式系統的初步建議**：{rec['emoji']} **{rec['label']}**（net = {rec['net_score']}）

多頭訊號：
{bullish_text}

空頭訊號：
{bearish_text}

**圖表型態原始數據**：
{chart_block}

**Features**:
- 距 52w 高: {features.get('drawdown_52w', 0)*100:+.1f}%
- 距 MA200: {features.get('price_vs_ma200', 0)*100:+.1f}%
- 3m / 6m / 12m return: {features.get('momentum_3m', 0)*100:+.1f}% / {features.get('momentum_6m', 0)*100:+.1f}% / {features.get('momentum_12m', 0)*100:+.1f}%
- 60d 波動率: {features.get('vol_60d', 0)*100:.0f}% (年化)
- 對 SPY 3m 超額: {features.get('rs_vs_spy_3m', 0)*100:+.1f}%

**Macro context**：
{macro_indicators}

{macro_brief}
{sector_block}{ticker_news_block}
**財報日期**: {earnings_summary}

請以繁體中文輸出**結構化建議**：

### 建議: {rec['label']}  ({rec['emoji']})
**信心程度**: 高 / 中高 / 中 / 低（依據訊號 convergence + 新聞 / macro 一致性）

### 主要理由
（**3-5 條 bullet**，每條一句話。優先引用最強的訊號、最重要的 macro / 產業 narrative、最關鍵的個股新聞）

### 風險與反向訊號
（**2-3 條**，誠實列出可能讓建議轉向的因素）

### 具體執行建議
- **進場 / 加碼價位區間**（如果 BUY/ACCUMULATE）
- **停損價位**（任何 BUY/HOLD 都要設）
- **減碼 / 出場 trigger**（任何 HOLD/TRIM 都要設）
- **觀察期**：下次重新評估的時機（例：FOMC 後、Q3 財報前）

### 一句話總結
（直接告訴使用者該怎麼動：例「現價分批進，跌破 $X 停損」）

⚠️ 最後加一行：「以上為基於公開資料的技術分析，非投資建議。」

注意：
- BUY 不等於「全押」。實際部位由使用者根據自己的 portfolio size 決定。
- 設停損是必須的紀律，不是選項。
- 不要保證任何事。"""

    try:
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system",
                     "content": "你是嚴謹的技術分析師。誠實、結構化、永遠提醒非投資建議。"},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 1000,
                "temperature": 0.3,
            },
            timeout=60,
        )
        if response.status_code != 200:
            return f"❌ Groq API error {response.status_code}: {response.text[:200]}"
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq API error: {e}"


# ============================================================
# Main
# ============================================================

def analyze_watchlist(tickers: list[str], out_path: Path,
                      use_llm: bool = True) -> None:
    # SPY once for relative-strength calculation
    print("📡 Fetching SPY benchmark...")
    spy_hist = fetch_history("SPY", days=400)

    # Macro brief once
    macro_data = {"brief": "", "indicators_formatted": ""}
    if use_llm:
        print("🌍 Fetching macro context...")
        macro_data = get_macro_brief()

    lines = []
    lines.append(f"# 📡 Scout Watchlist — {datetime.now():%Y-%m-%d %H:%M}")
    lines.append("")
    lines.append(f"監看 tickers: {', '.join(tickers)}")
    lines.append("")
    lines.append("⚠️ **以下為基於公開資料的技術分析，非投資建議。任何進出皆為使用者自行決定。**")
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

    for i, ticker in enumerate(tickers, start=1):
        print(f"\n📊 [{i}/{len(tickers)}] Analyzing {ticker}...")

        hist = fetch_history(ticker, days=400)
        if hist.empty:
            print(f"  ⚠️ {ticker}: no data")
            lines.append(f"## {ticker}\n\n_(no data)_\n\n---\n")
            continue

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
                macro_data.get("brief", ""), macro_data.get("indicators_formatted", ""),
            )

        # 1d, 5d, 30d change
        close = hist["Close"]
        try:
            change_1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100
            change_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            change_30d = (close.iloc[-1] / close.iloc[-22] - 1) * 100
        except IndexError:
            change_1d = change_5d = change_30d = 0

        # Console summary
        print(f"   Price ${features['close']:.2f}  "
              f"1d {change_1d:+.1f}%  5d {change_5d:+.1f}%  30d {change_30d:+.1f}%")
        print(f"   Recommendation: {rec['emoji']} {rec['label']}  "
              f"(net score {rec['net_score']:+d})")

        # Markdown
        lines.append(f"## {ticker}  {rec['emoji']} **{rec['label']}**")
        lines.append(f"**Price**: ${features['close']:.2f}  ")
        lines.append(f"**1d**: {change_1d:+.2f}%  |  "
                     f"**5d**: {change_5d:+.2f}%  |  "
                     f"**30d**: {change_30d:+.2f}%  ")
        lines.append(f"**Net signal score**: {rec['net_score']:+d}  "
                     f"(bullish {sum(w for _, w in rec['bullish_signals'])}, "
                     f"bearish {sum(w for _, w in rec['bearish_signals'])})")
        lines.append("")

        lines.append("### 多頭訊號")
        for s, w in rec["bullish_signals"]:
            lines.append(f"- 🟢 {s} _(weight {w})_")
        if not rec["bullish_signals"]:
            lines.append("- (無)")
        lines.append("")
        lines.append("### 空頭訊號")
        for s, w in rec["bearish_signals"]:
            lines.append(f"- 🔴 {s} _(weight {w})_")
        if not rec["bearish_signals"]:
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

        # Evidence
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
            lines.append(f"### 財報日期")
            lines.append(earnings_summary)
            lines.append("")

        lines.append("### 圖表型態原始數據")
        lines.append("```")
        lines.append(chart_block)
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text("\n".join(lines))
    print(f"\n💾 Watchlist report saved → {out_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tickers", nargs="+",
                        help="Tickers to watch, e.g. NVDA AVGO TSM")
    parser.add_argument("--out", default="results/watchlist_report.md")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM rationale (rules-only)")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    out_path = PROJECT_ROOT / args.out
    analyze_watchlist(tickers, out_path, use_llm=not args.no_llm)


if __name__ == "__main__":
    main()
