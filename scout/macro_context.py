"""Fetch and summarize the current macro / international / tech environment.

Pulls general market news from broad ETFs (SPY, QQQ, SMH, TLT) then asks Groq
LLM to summarize the dominant themes. The summary is cached for 1 hour to
avoid re-summarizing across multiple thesis calls within the same run.

This macro brief is injected into every per-ticker thesis prompt so the
LLM grounds its reasoning in the current macro environment, not just the
ticker's own news.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import requests

CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "macro_brief.json"
CACHE_TTL_SECONDS = 3600  # 1 hour


# Broad-market and thematic ETFs whose news captures different angles
BROAD_SOURCES = [
    ("SPY", "整體美股 / 市場情緒"),
    ("QQQ", "科技股 / AI / 雲端"),
    ("SMH", "半導體 / 晶片"),
    ("TLT", "債券 / 利率 / 通膨"),
]

# Quantitative macro indicators (yfinance tickers, no extra API key needed)
QUANT_INDICATORS = [
    ("^TNX", "10y Treasury yield"),
    ("^VIX", "VIX (volatility)"),
    ("DX-Y.NYB", "Dollar Index"),
    ("GC=F", "Gold futures"),
    ("CL=F", "Crude oil futures"),
    ("BTC-USD", "Bitcoin"),
]


def get_macro_indicators() -> dict:
    """Fetch current values of key macro indicators via yfinance."""
    import yfinance as yf
    out = {}
    for ticker, label in QUANT_INDICATORS:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="5d")
            if hist.empty or "Close" not in hist.columns:
                continue
            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[0])
            change_pct = (current - prev) / prev * 100 if prev else 0
            out[ticker] = {
                "label": label,
                "value": current,
                "5d_change_pct": change_pct,
            }
        except Exception:
            continue
    return out


def format_indicators(indicators: dict) -> str:
    """Human-readable summary of indicators for the prompt."""
    if not indicators:
        return ""
    lines = ["量化指標（過去 5 天）："]
    for ticker, info in indicators.items():
        v = info["value"]
        c = info["5d_change_pct"]
        arrow = "↗" if c > 0.5 else ("↘" if c < -0.5 else "→")
        if ticker == "^TNX":
            lines.append(f"- 10年期美債殖利率: {v:.2f}% {arrow} ({c:+.1f}% 5d)")
        elif ticker == "^VIX":
            level = "高恐慌" if v > 30 else ("正常" if v > 15 else "低波動")
            lines.append(f"- VIX 恐慌指數: {v:.1f} ({level}) {arrow}")
        elif ticker == "DX-Y.NYB":
            lines.append(f"- 美元指數 DXY: {v:.2f} {arrow} ({c:+.1f}% 5d)")
        elif ticker == "GC=F":
            lines.append(f"- 黃金 (期貨): ${v:.0f} {arrow} ({c:+.1f}% 5d)")
        elif ticker == "CL=F":
            lines.append(f"- 原油 WTI: ${v:.1f} {arrow} ({c:+.1f}% 5d)")
        elif ticker == "BTC-USD":
            lines.append(f"- 比特幣: ${v:,.0f} {arrow} ({c:+.1f}% 5d)")
    return "\n".join(lines)


def _read_cache():
    if not CACHE_PATH.exists():
        return None
    try:
        data = json.loads(CACHE_PATH.read_text())
        if time.time() - data.get("ts", 0) > CACHE_TTL_SECONDS:
            return None
        return data
    except Exception:
        return None


def _write_cache(brief: str, sources_used: list[str]):
    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    CACHE_PATH.write_text(json.dumps({
        "ts": time.time(),
        "brief": brief,
        "sources_used": sources_used,
    }, ensure_ascii=False, indent=2))


def _fetch_headlines() -> tuple[list[str], list[str]]:
    """Returns (all_headlines, sources_used)."""
    import yfinance as yf
    headlines = []
    sources_used = []
    for ticker, tag in BROAD_SOURCES:
        try:
            t = yf.Ticker(ticker)
            raw = getattr(t, "news", None) or []
            for item in raw[:5]:
                content = item.get("content") if isinstance(item, dict) else None
                if content:
                    title = content.get("title") or content.get("summary")
                    publisher = (content.get("provider") or {}).get("displayName", "")
                else:
                    title = item.get("title", "") if isinstance(item, dict) else ""
                    publisher = item.get("publisher", "") if isinstance(item, dict) else ""
                if title:
                    headlines.append(f"[{tag}] {title}" + (f" ({publisher})" if publisher else ""))
            sources_used.append(ticker)
        except Exception:
            continue
    return headlines, sources_used


def _summarize(headlines: list[str], indicators: dict, api_key: str) -> str:
    """Ask LLM to summarize the macro environment."""
    quant_block = format_indicators(indicators)
    prompt = (("以下是當前**量化指標**：\n\n" + quant_block + "\n\n") if quant_block else "") + \
"""以下是當前美股市場、半導體產業、債券利率的最新新聞標題（從 SPY / QQQ / SMH / TLT 抓取）：

""" + "\n".join(f"- {h}" for h in headlines) + """

請以**繁體中文** 250 字以內，總結當前**總體環境**，必須涵蓋（**整合量化指標 + 新聞 narrative**）：

1. **總體情緒**：market risk-on / risk-off？看 VIX + SPY 新聞
2. **利率 / 通膨**：10y yield 趨勢 + Fed narrative
3. **科技 / 半導體 narrative**：AI、晶片、HBM、半導體週期
4. **地緣 / 政策 / 商品**：中美、選舉、關稅、油價/金價/美元走勢
5. **近期關鍵 catalyst**：FOMC、財報季、政策日

格式：**5 個 bullet points**，每點一行，每行 40-60 字（要顯式提到量化指標值）。
不要保證未來，誠實寫不確定。"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "你是嚴謹的總體策略分析師。"},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 400,
            "temperature": 0.3,
        },
        timeout=45,
    )
    if response.status_code != 200:
        return f"(macro brief unavailable: {response.status_code})"
    return response.json()["choices"][0]["message"]["content"]


def get_macro_brief(force_refresh: bool = False) -> dict:
    """Returns a dict with 'brief' and 'sources_used'. Cached for 1 hour."""
    if not force_refresh:
        cached = _read_cache()
        if cached:
            return cached

    # Need Groq key
    from scout.thesis_writer import _get_groq_key
    api_key = _get_groq_key()
    if not api_key:
        return {
            "ts": time.time(),
            "brief": "(macro brief unavailable: no Groq API key)",
            "sources_used": [],
        }

    headlines, sources = _fetch_headlines()
    indicators = get_macro_indicators()
    if not headlines and not indicators:
        return {
            "ts": time.time(),
            "brief": "(macro brief unavailable: could not fetch news or indicators)",
            "sources_used": [],
            "indicators": {},
        }

    brief = _summarize(headlines, indicators, api_key)
    out = {
        "ts": time.time(),
        "brief": brief,
        "sources_used": sources,
        "indicators": indicators,
        "indicators_formatted": format_indicators(indicators),
    }
    _write_cache_full(out)
    return out


def _write_cache_full(payload: dict):
    """Write full payload (brief + indicators)."""
    CACHE_PATH.parent.mkdir(exist_ok=True, parents=True)
    CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def main():
    """Smoke test."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true",
                        help="force re-fetch even if cache fresh")
    args = parser.parse_args()

    print("🌍 Fetching macro context...\n")
    data = get_macro_brief(force_refresh=args.refresh)
    print("Sources used :", ", ".join(data.get("sources_used", [])))
    print("Cache age (s):", int(time.time() - data.get("ts", 0)))
    print()
    print("─" * 60)
    print(data.get("brief", "(empty)"))
    print("─" * 60)


if __name__ == "__main__":
    main()
