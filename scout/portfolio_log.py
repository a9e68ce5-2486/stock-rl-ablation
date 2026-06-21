"""Append daily portfolio snapshot to Excel.

Records once per trading day, one row per ticker (lots aggregated):
  日期 | 股票名稱 | 股數 | 均價 | 收盤股價 | 損益 | 損益率

Idempotent: re-running the same day overwrites that day's rows
rather than appending duplicates.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

LOG_PATH = PROJECT_ROOT / "results" / "portfolio_log.xlsx"


def _load_existing() -> pd.DataFrame:
    """Return existing log df, empty if file doesn't exist."""
    if not LOG_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(LOG_PATH)
    except Exception:
        return pd.DataFrame()


def _drop_today(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Remove rows matching today (idempotent append)."""
    if df.empty:
        return df
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"]).dt.date
    return df[df["日期"] != today]


def append_daily_snapshot(stats_list: list[dict], cash_usd: float = 0,
                          spy_close: float | None = None,
                          when: date | None = None):
    """Write one row per ticker (lots aggregated) for today.

    Columns: 日期 | 股票名稱 | 股數 | 均價 | 收盤股價 | 損益 | 損益率
    """
    today = when or date.today()
    existing = _load_existing()

    new_rows = []
    for s in stats_list:
        new_rows.append({
            "日期": today,
            "股票名稱": s["ticker"],
            "股數": round(s["shares"], 5),
            "均價": round(s["avg_cost"], 4),
            "收盤股價": round(s["current_price"], 4),
            "損益": round(s["pl_dollar"], 2),
            "損益率": round(s["pl_pct"], 2),
        })

    existing = _drop_today(existing, today)
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = combined.sort_values(["日期", "股票名稱"])

    LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(LOG_PATH, engine="openpyxl") as writer:
        combined.to_excel(writer, index=False, sheet_name="portfolio")

    return {
        "log_path": str(LOG_PATH),
        "rows_appended": len(new_rows),
        "total_rows": len(combined),
    }


if __name__ == "__main__":
    # Quick test
    fake_stats = [{
        "ticker": "TEST",
        "shares": 10,
        "avg_cost": 100,
        "current_price": 110,
        "pl_dollar": 100,
        "pl_pct": 10,
    }]
    info = append_daily_snapshot(fake_stats)
    print(f"✓ Log appended to {info['log_path']}")
    print(f"  New rows: {info['rows_appended']}, Total rows: {info['total_rows']}")
