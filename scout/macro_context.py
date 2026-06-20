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


def _summarize(headlines: list[str], api_key: str) -> str:
    """Ask LLM to summarize the macro environment."""
    prompt = """以下是當前美股市場、半導體產業、債券利率的最新新聞標題（從 SPY / QQQ / SMH / TLT 抓取）：

""" + "\n".join(f"- {h}" for h in headlines) + """

請以**繁體中文** 200 字以內，總結當前**總體環境**，必須涵蓋：

1. **總體情緒**：市場目前 risk-on 還是 risk-off？什麼主導？
2. **利率 / 通膨**：聯準會走向、市場預期
3. **科技 / 半導體 narrative**：AI、晶片、HBM 等
4. **地緣 / 政策**：中美、選舉、關稅、能源
5. **近期關鍵 catalyst**：即將到來的事件（FOMC、財報、政策日）

格式：**5 個 bullet points**，每點一行，每行 30-50 字。
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
    if not headlines:
        return {
            "ts": time.time(),
            "brief": "(macro brief unavailable: could not fetch ETF news)",
            "sources_used": [],
        }

    brief = _summarize(headlines, api_key)
    _write_cache(brief, sources)
    return {"ts": time.time(), "brief": brief, "sources_used": sources}


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
