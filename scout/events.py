"""Material-event timeline per stock.

For each ticker, compile a chronological list of events that could move
the price. Sources:
  1. Stock splits (yfinance .splits)
  2. Dividend policy changes (scout.dividend_info)
  3. Upcoming + recent earnings (scout.earnings_calendar)
  4. Major news events (extracted from yfinance news headlines)

Each event has: date, type, title, impact (high/medium/low), source.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, date

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scout.dividend_info import detect_policy_changes, get_latest_dividend
from scout.earnings_calendar import get_next_earnings, format_earnings_summary
from scout.thesis_writer import fetch_news


def _get_splits(ticker: str) -> list[dict]:
    """Stock splits — yfinance .splits"""
    try:
        import yfinance as yf
        sp = yf.Ticker(ticker).splits
        out = []
        for d, ratio in sp.items():
            out.append({
                "date": d.strftime("%Y-%m-%d"),
                "type": "split",
                "title": f"股票分拆 {float(ratio):g} : 1",
                "impact": "high",
                "source": "yfinance",
            })
        return out
    except Exception:
        return []


def _dividend_change_events(ticker: str) -> list[dict]:
    changes = detect_policy_changes(ticker)
    out = []
    for c in changes:
        impact = (
            "high"   if abs(c["change_pct"]) >= 50 else
            "medium" if abs(c["change_pct"]) >= 20 else
            "low"
        )
        direction = "📈 調漲" if c["change_pct"] > 0 else "📉 調降"
        out.append({
            "date": c["date"],
            "type": "dividend_change",
            "title": f"配息政策變更：{direction} "
                     f"${c['from_per_share']:.4f} → ${c['to_per_share']:.4f} "
                     f"({c['change_pct']:+.1f}%)",
            "impact": impact,
            "source": "yfinance dividends",
        })
    return out


def _earnings_events(ticker: str) -> list[dict]:
    info = get_next_earnings(ticker)
    if not info:
        return []
    summary = format_earnings_summary(info)
    days = info.get("days_until")
    if days is None:
        return []
    if days < 0:
        impact = "low"   # already happened
        title = f"上次財報: {info.get('date')} ({-days} 天前)"
    elif days < 7:
        impact = "high"
        title = f"⚠️ 即將財報: {info.get('date')} ({days} 天後 — 高 IV)"
    elif days < 21:
        impact = "medium"
        title = f"📅 即將財報: {info.get('date')} ({days} 天後)"
    else:
        impact = "low"
        title = f"下次財報: {info.get('date')} ({days} 天後)"
    return [{
        "date": info.get("date", ""),
        "type": "earnings",
        "title": title,
        "impact": impact,
        "source": "yfinance calendar",
    }]


# Keywords that flag a news headline as potentially material
MATERIAL_KEYWORDS = [
    # English
    "acquires", "acquisition", "merger", "buyout", "takeover",
    "lawsuit", "investigation", "subpoena", "doj", "ftc", "sec ",
    "downgrade", "upgrade", "price target",
    "earnings beat", "earnings miss", "guidance", "raises", "cuts",
    "ceo", "cfo", "resigns", "steps down", "fired", "hires",
    "split", "buyback", "dividend",
    "fda approval", "fda rejection", "clinical", "phase 3",
    "tariff", "sanction", "ban", "export control",
    "partnership", "deal", "contract win",
    "recall", "outage", "breach", "hack",
    # Chinese
    "收購", "併購", "訴訟", "下修", "調升", "升評", "降評",
    "目標價", "升任", "辭職", "下台", "罰款", "停損", "拆股",
    "回購", "庫藏股", "股利", "配息", "關稅", "禁令", "管制",
]


def _news_events(ticker: str, max_items: int = 8) -> list[dict]:
    """Filter recent news headlines for material-event keywords."""
    headlines = fetch_news(ticker, max_items=max_items)
    out = []
    for h in headlines:
        lower = h.lower()
        # Match if any material keyword is in the headline
        hits = [k for k in MATERIAL_KEYWORDS if k.lower() in lower]
        if not hits:
            continue
        impact = "high" if len(hits) >= 2 else "medium"
        out.append({
            "date": date.today().strftime("%Y-%m-%d"),  # yfinance news doesn't give date easily
            "type": "news",
            "title": f"📰 {h}",
            "impact": impact,
            "source": "yfinance news",
            "keywords": hits[:3],
        })
    return out


def get_events(ticker: str, lookback_days: int = 365) -> list[dict]:
    """Compile the full event timeline for a ticker, most recent first."""
    events = []
    events.extend(_get_splits(ticker))
    events.extend(_dividend_change_events(ticker))
    events.extend(_earnings_events(ticker))
    events.extend(_news_events(ticker))

    # Filter by lookback (only for events with a real date)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
    out = []
    for ev in events:
        try:
            evt_date = pd.Timestamp(ev["date"])
            if evt_date >= cutoff:
                out.append(ev)
        except (ValueError, TypeError):
            out.append(ev)  # keep events with non-parseable dates

    out.sort(key=lambda e: e["date"], reverse=True)
    return out


def events_to_bullets(events: list[dict], max_items: int = 8) -> list[str]:
    """Human-readable bullets for the report."""
    if not events:
        return []
    icon_map = {"high": "🔴", "medium": "🟡", "low": "⚪"}
    bullets = []
    for ev in events[:max_items]:
        icon = icon_map.get(ev["impact"], "⚪")
        bullets.append(f"{icon} **{ev['date']}** — {ev['title']}")
    return bullets


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    args = parser.parse_args()
    events = get_events(args.ticker)
    print(f"📅 Events for {args.ticker} ({len(events)} found)")
    for b in events_to_bullets(events):
        print(f"  {b}")
