"""Dividend metadata: most recent payment, frequency, policy changes.

This module focuses on the "current" state of a stock's dividend, not the
cumulative history that scout/portfolio.py was using before. Users want to
know:
  - The most recent payment (per share + total based on holdings)
  - How often the company pays (monthly / quarterly / semi-annual / annual)
  - When the dividend policy changed (e.g. NVDA spiked from \$0.01 to \$0.25)

These signals matter for current yield calculations, holding decisions, and
flagging structural events that affect the stock price.
"""
from __future__ import annotations
import pandas as pd

# Thresholds
POLICY_CHANGE_PCT = 10.0  # ≥ 10% change in dividend = "policy change"


def _fetch_dividends(ticker: str) -> pd.Series:
    try:
        import yfinance as yf
        d = yf.Ticker(ticker).dividends
        return d if d is not None else pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def get_latest_dividend(ticker: str) -> dict:
    """Return the most recent dividend payment, or empty dict if none."""
    divs = _fetch_dividends(ticker)
    if divs.empty:
        return {}
    last_date = divs.index[-1]
    last_amount = float(divs.iloc[-1])
    days_ago = (pd.Timestamp.now(tz=last_date.tz) - last_date).days
    return {
        "date": last_date.strftime("%Y-%m-%d"),
        "per_share": last_amount,
        "days_ago": days_ago,
        "payments_total": len(divs),
    }


def infer_frequency(ticker: str) -> dict:
    """Look at recent payment gaps to classify frequency."""
    divs = _fetch_dividends(ticker)
    if len(divs) < 2:
        return {"label": "未知", "median_days": None, "implied_per_year": 0}
    # Use last 6 payments for gap inference
    recent = divs.iloc[-6:]
    dates = recent.index
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    median_gap = float(pd.Series(gaps).median())

    if median_gap < 20:
        label = "月配"
    elif median_gap < 50:
        label = "雙月配"
    elif median_gap < 110:
        label = "季配"
    elif median_gap < 200:
        label = "半年配"
    elif median_gap < 400:
        label = "年配"
    else:
        label = "不定期"

    per_year = round(365 / median_gap, 1) if median_gap else 0
    return {"label": label, "median_days": median_gap, "implied_per_year": per_year}


def detect_policy_changes(ticker: str,
                          threshold_pct: float = POLICY_CHANGE_PCT,
                          max_results: int = 5) -> list[dict]:
    """Find dates where dividend amount changed by ≥ threshold_pct from prior.

    Returns most recent first.
    """
    divs = _fetch_dividends(ticker)
    if len(divs) < 2:
        return []
    changes = []
    prev = float(divs.iloc[0])
    for date, amt in list(divs.items())[1:]:
        amt = float(amt)
        if prev > 0:
            change_pct = (amt - prev) / prev * 100
            if abs(change_pct) >= threshold_pct:
                changes.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "from_per_share": prev,
                    "to_per_share": amt,
                    "change_pct": change_pct,
                })
        prev = amt
    # Most recent first
    changes.sort(key=lambda c: c["date"], reverse=True)
    return changes[:max_results]


def expected_annual_yield(ticker: str, current_price: float) -> dict:
    """Estimate annual yield from latest dividend × implied annual frequency."""
    latest = get_latest_dividend(ticker)
    freq = infer_frequency(ticker)
    if not latest or freq["implied_per_year"] == 0 or current_price <= 0:
        return {}
    annual = latest["per_share"] * freq["implied_per_year"]
    yield_pct = annual / current_price * 100
    return {
        "annual_per_share": annual,
        "current_yield_pct": yield_pct,
    }


def dividend_snapshot(ticker: str, shares: float = 1.0,
                      current_price: float = 0) -> dict:
    """One-stop snapshot for the portfolio report."""
    latest = get_latest_dividend(ticker)
    freq = infer_frequency(ticker)
    changes = detect_policy_changes(ticker)
    yld = expected_annual_yield(ticker, current_price) if current_price else {}
    return {
        "latest": latest,
        "frequency": freq,
        "policy_changes": changes,
        "yield": yld,
        "latest_total_dollar": (latest["per_share"] * shares) if latest else 0,
    }


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--shares", type=float, default=10.0)
    parser.add_argument("--price", type=float, default=0)
    args = parser.parse_args()
    snap = dividend_snapshot(args.ticker, args.shares, args.price)
    print(json.dumps(snap, indent=2, ensure_ascii=False))
