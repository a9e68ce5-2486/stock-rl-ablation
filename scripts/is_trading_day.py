#!/usr/bin/env python3
"""Exit 0 if yesterday US/Eastern time was a NYSE trading day; exit 1 otherwise.

We run the daily pipeline in Taiwan morning, which corresponds to the US
previous business day's close. So we should run only when YESTERDAY ET
was a trading day:
  - Skip if yesterday was Saturday / Sunday
  - Skip if yesterday was a NYSE holiday

This way Taiwan Monday (= US Sunday) gets skipped (no Friday close to
report on Monday because Friday's already 3 days old), but Tuesday-Saturday
TW run with Mon-Fri ET data.

Actually we want: only run on days where we have FRESH data, i.e. the
PRIOR ET day was a trading day.
"""
from __future__ import annotations
import sys
from datetime import date, datetime, timedelta, timezone

# NYSE 2026-2028 holidays (full-day closures; early-close days like
# day-after-Thanksgiving are still trading days so not in this list).
NYSE_HOLIDAYS: set[date] = {
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day observed (Jul 4 is Sat)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),
    date(2027, 1, 18),
    date(2027, 2, 15),
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),
    date(2027, 6, 18),   # Juneteenth observed (Jun 19 is Sat)
    date(2027, 7, 5),    # observed
    date(2027, 9, 6),
    date(2027, 11, 25),
    date(2027, 12, 24),  # observed (Dec 25 is Sat)
    # 2028
    date(2028, 1, 17),
    date(2028, 2, 21),
    date(2028, 4, 14),
    date(2028, 5, 29),
    date(2028, 6, 19),
    date(2028, 7, 4),
    date(2028, 9, 4),
    date(2028, 11, 23),
    date(2028, 12, 25),
}


def is_yesterday_trading_day() -> tuple[bool, str]:
    """Was yesterday (US/Eastern) a NYSE trading day?"""
    # Use UTC-5 as approximation of Eastern (close enough for date math)
    now_et = datetime.now(timezone(timedelta(hours=-5)))
    yesterday = (now_et - timedelta(days=1)).date()
    if yesterday.weekday() >= 5:
        return False, f"Yesterday ET ({yesterday}) was {yesterday.strftime('%A')}"
    if yesterday in NYSE_HOLIDAYS:
        return False, f"Yesterday ET ({yesterday}) was NYSE holiday"
    return True, f"Yesterday ET ({yesterday}) was a trading day"


def main():
    ok, reason = is_yesterday_trading_day()
    print(reason)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
