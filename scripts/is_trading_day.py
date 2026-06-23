#!/usr/bin/env python3
"""Exit 0 if today (Taiwan time) should produce a fresh briefing; exit 1 if not.

Mapping logic (Taiwan-centric):
    TW Sun → US Sat   (no US trading)      → skip
    TW Mon → US Sun   (no US trading)      → skip
    TW Tue → US Mon   close                → run (unless US holiday)
    TW Wed → US Tue   close                → run (unless US holiday)
    TW Thu → US Wed   close                → run (unless US holiday)
    TW Fri → US Thu   close                → run (unless US holiday)
    TW Sat → US Fri   close                → run (unless US holiday)

The "US relevant date" is simply yesterday in Taipei — since the US is ~12-13h
behind, at any TW morning the most recent US close is the day-before's session.

Skips when that day was a NYSE holiday (hardcoded list 2026-2028).
"""
from __future__ import annotations
import sys
from datetime import date, timedelta

# NYSE 2026-2028 full-day closures (yyyy-mm-dd in US local trading dates).
# Early-close days like Black Friday are STILL trading days, not in this set.
NYSE_HOLIDAYS: set[date] = {
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
    # 2028
    date(2028, 1, 17), date(2028, 2, 21), date(2028, 4, 14),
    date(2028, 5, 29), date(2028, 6, 19), date(2028, 7, 4),
    date(2028, 9, 4), date(2028, 11, 23), date(2028, 12, 25),
}


def main():
    today_tw = date.today()
    us_relevant = today_tw - timedelta(days=1)  # what US trading day we'd report on
    wd = us_relevant.weekday()  # 0=Mon, 5=Sat, 6=Sun

    if wd == 5:
        print(f"今天台灣 {today_tw} → 昨日對應美股 {us_relevant} (週六，無交易)，跳過")
        sys.exit(1)
    if wd == 6:
        print(f"今天台灣 {today_tw} → 昨日對應美股 {us_relevant} (週日，無交易)，跳過")
        sys.exit(1)
    if us_relevant in NYSE_HOLIDAYS:
        print(f"今天台灣 {today_tw} → 美股 {us_relevant} 是 NYSE 假日，跳過")
        sys.exit(1)

    print(f"今天台灣 {today_tw} → 對應美股 {us_relevant} 收盤資料，執行")
    sys.exit(0)


if __name__ == "__main__":
    main()
