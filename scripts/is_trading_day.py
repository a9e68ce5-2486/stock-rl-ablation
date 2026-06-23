#!/usr/bin/env python3
"""Exit 0 if there is a NYSE close to report on; exit 1 otherwise.

Strategy: ask yfinance for SPY's latest available close date. If that date
is within the last 2 calendar days (TW local), we have fresh data worth
briefing on. Otherwise (weekends, US holidays) skip.

This is more robust than computing 'yesterday in ET' by hand:
  - yfinance returns the correct most-recent trading day automatically
  - DST shifts, midnight rollovers, and holiday edge cases are handled
  - Works the same whether we're in TW summer or US winter

Falls back to a hardcoded weekend + holiday list if yfinance is offline.
"""
from __future__ import annotations
import sys
import warnings
from datetime import date, datetime, timedelta, timezone

warnings.filterwarnings("ignore")

# NYSE 2026-2028 holidays (fallback only — yfinance is preferred)
NYSE_HOLIDAYS: set[date] = {
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14),
    date(2028, 5, 29), date(2028, 6, 19), date(2028, 7, 4),
    date(2028, 9, 4), date(2028, 11, 23), date(2028, 12, 25),
}


def _check_via_yfinance() -> tuple[bool, str] | None:
    """Returns (should_run, reason). None if yfinance unavailable."""
    try:
        import yfinance as yf
        hist = yf.Ticker("SPY").history(period="10d")
        if hist.empty:
            return None
        last_close = hist.index[-1].date()
        today_local = date.today()
        days_ago = (today_local - last_close).days
        if days_ago <= 2:
            return True, f"Latest SPY close: {last_close} ({days_ago} 天前)"
        return False, (f"Latest SPY close: {last_close} ({days_ago} 天前) — "
                       "no new market data since last run")
    except Exception:
        return None


def _check_via_calendar() -> tuple[bool, str]:
    """Fallback: today in ET (UTC-4 = EDT). Returns (should_run, reason)."""
    # During summer DST it's UTC-4; in winter UTC-5. EDT is most common
    # business-hours offset for users running in TW morning.
    now_et = datetime.now(timezone(timedelta(hours=-4)))
    today_et = now_et.date()
    if today_et.weekday() >= 5:
        return False, f"Today ET ({today_et}) was {today_et.strftime('%A')}"
    if today_et in NYSE_HOLIDAYS:
        return False, f"Today ET ({today_et}) was NYSE holiday"
    return True, f"Today ET ({today_et}) was a trading day"


def main():
    result = _check_via_yfinance()
    if result is None:
        # yfinance unavailable, use fallback
        result = _check_via_calendar()
    ok, reason = result
    print(reason)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
