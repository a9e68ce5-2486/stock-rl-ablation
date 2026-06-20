"""Next earnings date lookup.

Knowing if earnings is in 5 days vs 60 days dramatically changes a trade:
- Pre-earnings: high implied vol → bigger move possible (either direction)
- Just after earnings: catalyst known, less uncertainty
- 30+ days out: technical pattern more meaningful than catalyst

We use yfinance's .calendar (no extra API key needed).
"""
from __future__ import annotations
from datetime import datetime, date


def get_next_earnings(ticker: str) -> dict:
    """Returns {date, days_until, est_eps_avg, est_revenue_avg} or {} on fail."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        if not cal:
            return {}
        # yfinance returns either dict or DataFrame depending on version
        if isinstance(cal, dict):
            # Newer style
            edates = cal.get("Earnings Date", [])
            if not edates:
                return {}
            if isinstance(edates, list):
                next_date = edates[0]
            else:
                next_date = edates
            if not next_date:
                return {}
            if hasattr(next_date, "date"):
                next_date = next_date.date()
            today = date.today()
            days = (next_date - today).days
            return {
                "date": str(next_date),
                "days_until": int(days),
                "est_eps_avg": cal.get("Earnings Average", None),
                "est_revenue_avg": cal.get("Revenue Average", None),
            }
        # DataFrame style (older)
        try:
            # cal is a DataFrame with columns like "Earnings Date", "Earnings Average"
            edate = cal.loc["Earnings Date"][0] if "Earnings Date" in cal.index else None
            if edate is None:
                return {}
            if hasattr(edate, "date"):
                edate = edate.date()
            today = date.today()
            days = (edate - today).days
            return {"date": str(edate), "days_until": int(days)}
        except Exception:
            return {}
    except Exception:
        return {}


def format_earnings_summary(info: dict) -> str:
    """Human-readable summary for prompt injection."""
    if not info:
        return "(無法取得下次財報日期)"
    days = info.get("days_until")
    date_str = info.get("date", "")
    if days is None:
        return f"下次財報: {date_str}"
    if days < 0:
        return f"上次財報: {date_str} ({-days} 天前)"
    if days < 7:
        return f"⚠️ 下次財報: {date_str}（**{days} 天後 — 高 IV，事件風險大**）"
    if days < 21:
        return f"📅 下次財報: {date_str}（{days} 天後 — 接近，需考慮 hold-through）"
    if days < 60:
        return f"下次財報: {date_str}（{days} 天後）"
    return f"下次財報: {date_str}（{days} 天後 — 不影響短期判斷）"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MU")
    args = parser.parse_args()
    info = get_next_earnings(args.ticker)
    print(f"Ticker: {args.ticker}")
    print(f"Raw info: {info}")
    print(f"Summary: {format_earnings_summary(info)}")
