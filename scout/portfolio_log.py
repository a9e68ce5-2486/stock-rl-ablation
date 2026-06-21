"""Append daily portfolio snapshot to a multi-sheet Excel file.

Records once per trading day:
  - Per-position row (close price, shares, avg cost, P/L)
  - Portfolio-level summary row (total value, total P/L, daily change)

Excel structure:
  results/portfolio_log.xlsx
    Sheet "positions": one row per (date, ticker, broker) lot
    Sheet "totals"   : one row per date with portfolio-level aggregates

Idempotent: if today's row already exists for a position, it's overwritten
rather than duplicated (so re-running the same day doesn't pollute).
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

LOG_PATH = PROJECT_ROOT / "results" / "portfolio_log.xlsx"


def _load_existing() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (positions_df, totals_df), empty if file doesn't exist."""
    if not LOG_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        positions = pd.read_excel(LOG_PATH, sheet_name="positions")
        totals = pd.read_excel(LOG_PATH, sheet_name="totals")
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return positions, totals


def _drop_today(df: pd.DataFrame, today: date,
                ticker: str | None = None,
                broker: str | None = None) -> pd.DataFrame:
    """Remove rows matching today (+ ticker / broker) for idempotent append."""
    if df.empty:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date
    mask = df["date"] != today
    if ticker is not None:
        mask = mask | (df["ticker"] != ticker)
    if broker is not None and "broker" in df.columns:
        mask = mask | (df["broker"] != broker)
    return df[mask]


def append_daily_snapshot(stats_list: list[dict], cash_usd: float = 0,
                          spy_close: float | None = None,
                          when: date | None = None):
    """Write per-position + portfolio-total rows for today.

    stats_list: output of compute_position_stats for each (aggregated) position
                — but we expand to per-lot rows for tracking which broker.
    """
    today = when or date.today()
    positions, totals = _load_existing()

    # === Per-lot rows ===
    new_rows = []
    for s in stats_list:
        ticker = s["ticker"]
        current_price = s["current_price"]
        lots = s.get("lots", [])
        if not lots:
            # synthetic single-lot from aggregated stats
            lots = [{
                "shares": s["shares"],
                "avg_cost": s["avg_cost"],
                "broker": "—",
            }]
        for lot in lots:
            lshares = float(lot.get("shares", 0))
            lcost = float(lot.get("avg_cost", 0))
            broker = lot.get("broker", "—")
            lvalue = lshares * current_price
            lbasis = lshares * lcost
            lpl = lvalue - lbasis
            lpl_pct = (current_price / lcost - 1) * 100 if lcost else 0
            new_rows.append({
                "date": today,
                "ticker": ticker,
                "broker": broker,
                "shares": lshares,
                "avg_cost": round(lcost, 4),
                "close_price": round(current_price, 4),
                "value": round(lvalue, 2),
                "cost_basis": round(lbasis, 2),
                "pl_dollar": round(lpl, 2),
                "pl_pct": round(lpl_pct, 2),
                "notes": lot.get("notes", ""),
            })
            # Drop any prior row matching (date, ticker, broker)
            positions = _drop_today(positions, today, ticker=ticker,
                                    broker=broker)

    positions = pd.concat([positions, pd.DataFrame(new_rows)],
                          ignore_index=True)
    positions = positions.sort_values(["date", "ticker", "broker"])

    # === Portfolio-total row ===
    total_value = sum(s["current_value"] for s in stats_list) + cash_usd
    total_cost = sum(s["cost_basis"] for s in stats_list)
    total_pl = total_value - cash_usd - total_cost
    total_pl_pct = (total_pl / total_cost * 100) if total_cost else 0

    # Look up yesterday's total value for daily change
    daily_change_pct = None
    if not totals.empty:
        prev = totals.sort_values("date").iloc[-1]
        prev_total = float(prev.get("total_value_incl_cash", 0))
        if prev_total > 0:
            daily_change_pct = (total_value / prev_total - 1) * 100

    totals = _drop_today(totals, today)
    new_total = {
        "date": today,
        "n_positions": len([s for s in stats_list]),
        "total_value_incl_cash": round(total_value, 2),
        "total_equity_value": round(total_value - cash_usd, 2),
        "cash_usd": round(cash_usd, 2),
        "total_cost": round(total_cost, 2),
        "total_pl_dollar": round(total_pl, 2),
        "total_pl_pct": round(total_pl_pct, 2),
        "daily_change_pct": (round(daily_change_pct, 3)
                             if daily_change_pct is not None else None),
        "spy_close": round(spy_close, 2) if spy_close else None,
    }
    totals = pd.concat([totals, pd.DataFrame([new_total])], ignore_index=True)
    totals = totals.sort_values("date")

    # === Write back ===
    LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
    with pd.ExcelWriter(LOG_PATH, engine="openpyxl") as writer:
        positions.to_excel(writer, sheet_name="positions", index=False)
        totals.to_excel(writer, sheet_name="totals", index=False)

    return {
        "log_path": str(LOG_PATH),
        "rows_appended": len(new_rows) + 1,
        "today_total_value": total_value,
        "today_pl": total_pl,
        "today_pl_pct": total_pl_pct,
        "daily_change_pct": daily_change_pct,
    }


if __name__ == "__main__":
    # Quick test with synthetic data
    fake_stats = [{
        "ticker": "TEST",
        "shares": 10,
        "avg_cost": 100,
        "current_price": 110,
        "current_value": 1100,
        "cost_basis": 1000,
        "pl_dollar": 100,
        "pl_pct": 10,
        "lots": [{"shares": 10, "avg_cost": 100, "broker": "test"}],
    }]
    info = append_daily_snapshot(fake_stats, cash_usd=50)
    print(f"✓ Log appended to {info['log_path']}")
    print(f"  Rows: {info['rows_appended']}")
