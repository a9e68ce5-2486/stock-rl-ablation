"""LLM-based investment thesis writer.

Takes a stock pick's features + recent news → asks Groq Llama 3.3 70B
to write a short Chinese investment thesis explaining why this stock
might rally.

Uses Groq's free tier (~$0 / month for daily picks).
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


def _get_groq_key() -> str:
    """Try env var first, then OpenClaw auth-profiles.json fallback."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key
    # OpenClaw stores per-agent profiles
    candidates = [
        Path.home() / ".openclaw" / "agents" / "main" / "agent" / "auth-profiles.json",
        Path.home() / ".openclaw" / "auth-profiles.json",
    ]
    for profile_path in candidates:
        if not profile_path.exists():
            continue
        try:
            with open(profile_path) as f:
                data = json.load(f)
        except Exception:
            continue
        # Try several shapes
        candidates_obj = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict):
                    candidates_obj.append(v)
                elif isinstance(v, list):
                    candidates_obj.extend(x for x in v if isinstance(x, dict))
        elif isinstance(data, list):
            candidates_obj.extend(x for x in data if isinstance(x, dict))
        for prof in candidates_obj:
            blob = json.dumps(prof, default=str)
            if "groq" not in blob.lower():
                continue
            # Probe common key names
            for kn in ("apiKey", "token", "api_key", "key", "secret"):
                v = prof.get(kn)
                if isinstance(v, str) and v.startswith("gsk_"):
                    return v
            # Walk nested dicts one level
            for v in prof.values():
                if isinstance(v, dict):
                    for kn in ("apiKey", "token", "api_key", "key", "secret"):
                        vv = v.get(kn)
                        if isinstance(vv, str) and vv.startswith("gsk_"):
                            return vv
                if isinstance(v, str) and v.startswith("gsk_"):
                    return v
    return ""


def fetch_news(ticker: str, max_items: int = 5) -> list[str]:
    """Best-effort news fetch via yfinance. Returns short headline list."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw = getattr(t, "news", None) or []
        headlines = []
        for item in raw[:max_items]:
            # Newer yfinance wraps in .content
            content = item.get("content") if isinstance(item, dict) else None
            if content:
                title = content.get("title") or content.get("summary")
                publisher = (content.get("provider") or {}).get("displayName", "")
            else:
                title = item.get("title", "") if isinstance(item, dict) else ""
                publisher = item.get("publisher", "") if isinstance(item, dict) else ""
            if title:
                headlines.append(f"{title}" + (f" ({publisher})" if publisher else ""))
        return headlines
    except Exception:
        return []


def build_prompt(ticker: str, features: dict, score: float,
                 news_headlines: list[str],
                 macro_brief: str = "",
                 macro_indicators: str = "",
                 sector_etf: str = "",
                 sector_news: list[str] | None = None,
                 earnings_summary: str = "") -> str:
    news_block = ("**個股新聞**：\n" + "\n".join(f"- {h}" for h in news_headlines)
                  if news_headlines else "(無法取得最近新聞)")

    macro_block = (f"**當前總體環境**（請納入考量）：\n{macro_brief}\n\n"
                   if macro_brief and not macro_brief.startswith("(") else "")

    indicators_block = (f"**量化指標即時值**：\n{macro_indicators}\n\n"
                        if macro_indicators else "")

    sector_block = ""
    if sector_etf and sector_news:
        sector_block = (f"**{ticker} 的產業 ETF ({sector_etf}) 新聞**：\n"
                        + "\n".join(f"- {h}" for h in sector_news) + "\n\n")

    earnings_block = (f"**財報日期**: {earnings_summary}\n\n"
                      if earnings_summary and not earnings_summary.startswith("(") else "")

    return f"""你是嚴謹的量化分析師。針對美股 **{ticker}**（價格 ${features.get('close', 0):.2f}，截至 {features.get('date_str', 'N/A')}），根據下方資料寫一份**簡短的投資論點**。

{macro_block}{indicators_block}{sector_block}{earnings_block}{news_block}

技術面 features：
- 模型 rally 機率分數: {score:.3f}（universe 中位數 ≈ 0.05，p95 ≈ 0.14）
- 距 52 週高點: {features.get('drawdown_52w', 0)*100:+.1f}%
- 距 200 日 MA: {features.get('price_vs_ma200', 0)*100:+.1f}%
- 3 個月報酬: {features.get('momentum_3m', 0)*100:+.1f}%
- 6 個月報酬: {features.get('momentum_6m', 0)*100:+.1f}%
- 12 個月報酬: {features.get('momentum_12m', 0)*100:+.1f}%
- 60 日年化波動率: {features.get('vol_60d', 0)*100:.0f}%
- 60 日波動率歷史百分位: {features.get('vol_60d_rank', 0):.2f}
- 30 / 90 日成交量比: {features.get('volume_surge_30d', 1):.2f}×
- RSI(14): {features.get('rsi_14', 50):.0f}
- 對 SPY 3 個月超額報酬: {features.get('rs_vs_spy_3m', 0)*100:+.1f}%
- 距上次 52w 新高: {int(features.get('days_since_high', 0))} 天

**請用繁體中文輸出，依照下方格式（嚴格遵守）：**

### 公司
（**1 句話**敘述 {ticker} 主要業務、所在產業、規模）

### 技術面解讀
（**1 段**，4-6 句。把上面 features 翻譯成「目前股價處於什麼狀態」。例如：是否深度超賣？資金是否在 accumulate？相對大盤強弱如何？）

### 為什麼可能 rally
（**1 段**，5-7 句。**必須結合上面的「當前總體環境」與本檔個股新聞**，具體說明：
- 當前總體環境中**哪幾條** narrative 跟 {ticker} 直接相關？（例如：升息對科技股、AI capex 對半導體、中美關係對 EV、能源價格對太陽能）
- 個股本身的催化劑：財報、產品上市、合作案、政策受益
- **必須誠實提及主要風險**（基本面不確定、產業逆風、流動性、地緣風險等）

不要只說「產業趨勢正面」這種空話，要具體說「**因為 X 總體 narrative + Y 個股新聞 = Z 可能催化股價**」。
如果總體環境跟個股無關，要明說「目前總體環境跟 {ticker} 無直接關聯」。）

### 一句話結論
（**1 句**結論 + 信心程度 + 評分理由）

**信心程度評定規則（嚴格按此標準）**：
- 「**高**」：技術面同時符合 4+ 個深度 bottom 訊號（RSI<35、距高點<-30%、低於 MA200>15%、中期 capitulation、量能放大），且新聞有具體可預期催化劑（財報、產業政策、合作案、產品上市）→ 給「高」
- 「**中高**」：技術 pattern 強（3+ 深度訊號）但催化劑曖昧 / 沒明確催化劑 → 給「中高」
- 「**中**」：技術 pattern 部分滿足（2-3 訊號）或新聞偏中性 → 給「中」
- 「**低**」：訊號薄弱、或新聞明顯偏負面（如：被降評、合約丟失、現金流問題、產業逆風）→ 給「低」

**不要因為「不確定未來」就一律給中。當 pattern 強且新聞中性以上，請敢給高或中高。**
（市場上 RSI<25 + DD -50% 的 setup 歷史上 70% 機率反彈，這就是「高」信心 setup）

不要保證任何事，但給出**校準後的合理機率判斷**。"""


def write_thesis(ticker: str, features: dict, score: float,
                 model: str = DEFAULT_MODEL,
                 include_news: bool = True,
                 macro_brief: str = "",
                 macro_indicators: str = "",
                 include_sector_news: bool = True,
                 include_earnings: bool = True) -> str:
    """Generate a Chinese investment thesis using Groq."""
    api_key = _get_groq_key()
    if not api_key:
        return ("❌ Groq API key not found. Set GROQ_API_KEY env var:\n"
                "  export GROQ_API_KEY='gsk_...'\n"
                "Or log in via `openclaw models auth paste-api-key --provider groq`")

    headlines = fetch_news(ticker, max_items=5) if include_news else []

    sector_etf = ""
    sector_news: list[str] = []
    if include_sector_news:
        from scout.sector_news import fetch_sector_news
        sector_etf, sector_news = fetch_sector_news(ticker, max_items=4)

    earnings_summary = ""
    if include_earnings:
        from scout.earnings_calendar import get_next_earnings, format_earnings_summary
        earnings_info = get_next_earnings(ticker)
        earnings_summary = format_earnings_summary(earnings_info)

    prompt = build_prompt(ticker, features, score, headlines, macro_brief,
                          macro_indicators, sector_etf, sector_news,
                          earnings_summary)

    try:
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": "你是嚴謹的量化分析師，誠實提及不確定性與風險。"},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 700,
                "temperature": 0.4,
            },
            timeout=45,
        )
        if response.status_code != 200:
            return f"❌ Groq API error {response.status_code}: {response.text[:200]}"
        data = response.json()
        thesis = data["choices"][0]["message"]["content"]
        # Append evidence at end (ticker + sector news + earnings)
        evidence_blocks = []
        if headlines:
            evidence_blocks.append(
                "### 個股新聞 (yfinance)\n" + "\n".join(f"- {h}" for h in headlines))
        if sector_news:
            evidence_blocks.append(
                f"### 產業 {sector_etf} 新聞\n" + "\n".join(f"- {h}" for h in sector_news))
        if earnings_summary and not earnings_summary.startswith("("):
            evidence_blocks.append(f"### 財報日期\n{earnings_summary}")
        if evidence_blocks:
            thesis = thesis + "\n\n" + "\n\n".join(evidence_blocks)
        return thesis
    except requests.Timeout:
        return "❌ Groq API timeout (>45s)"
    except Exception as e:
        return f"❌ Groq API error: {e}"


def main():
    """Quick smoke test: write thesis for one ticker with mock features."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MU")
    args = parser.parse_args()

    # Mock features that look like a deep-value bottom
    mock = {
        "date_str": "2026-05-29",
        "close": 95.5,
        "drawdown_52w": -0.32,
        "price_vs_ma200": -0.12,
        "momentum_3m": -0.18,
        "momentum_6m": -0.05,
        "momentum_12m": 0.10,
        "vol_60d": 0.45,
        "vol_60d_rank": 0.65,
        "volume_surge_30d": 1.4,
        "rsi_14": 28,
        "rs_vs_spy_3m": -0.15,
        "days_since_high": 180,
    }

    print(f"🦞 Writing thesis for {args.ticker} (mock features)...\n")
    thesis = write_thesis(args.ticker, mock, score=0.42)
    print(thesis)


if __name__ == "__main__":
    main()
