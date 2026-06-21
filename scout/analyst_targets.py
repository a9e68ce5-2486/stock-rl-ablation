"""Analyst price targets, fair-value-ish estimates, and consensus rating.

yfinance exposes the Wall Street analyst consensus via Ticker.info:
  - targetMeanPrice, targetMedianPrice, targetHighPrice, targetLowPrice
  - recommendationKey (buy / hold / sell)
  - numberOfAnalystOpinions
  - forwardPE, trailingPE, pegRatio, marketCap

There's no clean "intrinsic fair value" field (Morningstar charges for that),
so we use:
  - "目標價" = analyst mean target (Street consensus on what they think it's worth)
  - "估值區間" = high/low from analyst range
  - "Fair value (簡易)" = forward EPS × sector-average forward PE
    (computed only if forwardPE available; cheap proxy)

These are NOT investment advice — analyst targets are often anchored
to the current price and revised after moves, not before.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_analyst_data(ticker: str) -> dict:
    """Pull analyst targets + recommendation + a few valuation ratios."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return {}

    return {
        "target_mean": info.get("targetMeanPrice"),
        "target_median": info.get("targetMedianPrice"),
        "target_high": info.get("targetHighPrice"),
        "target_low": info.get("targetLowPrice"),
        "recommendation": info.get("recommendationKey", "—"),
        "recommendation_mean": info.get("recommendationMean"),
        "num_analysts": info.get("numberOfAnalystOpinions", 0),
        "forward_pe": info.get("forwardPE"),
        "trailing_pe": info.get("trailingPE"),
        "peg_ratio": info.get("pegRatio") or info.get("trailingPegRatio"),
        "forward_eps": info.get("forwardEps"),
        "trailing_eps": info.get("trailingEps"),
        "market_cap": info.get("marketCap"),
        "price_to_book": info.get("priceToBook"),
        "beta": info.get("beta"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
    }


def compute_upside(current_price: float, target: float | None) -> float | None:
    if not target or current_price <= 0:
        return None
    return (target - current_price) / current_price * 100


# Sector-average forward P/E proxies (rough; for "naive fair value" calc only)
# Source: SP&P/NYU Damodaran tables (updated 2024), conservative levels.
SECTOR_AVG_FWD_PE = {
    "SMH": 22,       # Semis
    "PAVE": 18,      # Networking
    "WCLD": 28,      # Cloud / SaaS
    "TAN": 16,       # Solar
    "XBI": 20,       # Biotech
    "XLV": 18,       # Healthcare
    "XLY": 20,       # Consumer disc
    "XLE": 12,       # Energy
    "XLF": 14,       # Financials
    "XLI": 18,       # Industrials
    "DRIV": 22,      # EV / Auto innovation
    "BLOK": 25,      # Crypto-adjacent
    "ITA": 18,       # Defense
    "XLB": 14,       # Materials
    "ARKK": 25,      # Innovation / growth
    "IGM": 22,       # Tech
}


def naive_fair_value(forward_eps: float | None, sector_avg_pe: float | None) -> float | None:
    """Forward EPS × sector-average forward P/E. Very rough."""
    if not forward_eps or not sector_avg_pe:
        return None
    return float(forward_eps) * float(sector_avg_pe)


def get_targets_snapshot(ticker: str, current_price: float = 0,
                         sector_etf: str = "") -> dict:
    """One-stop snapshot for the portfolio report."""
    data = get_analyst_data(ticker)
    if not data:
        return {}

    upside_mean = compute_upside(current_price, data.get("target_mean"))
    upside_high = compute_upside(current_price, data.get("target_high"))
    upside_low = compute_upside(current_price, data.get("target_low"))

    sector_pe = SECTOR_AVG_FWD_PE.get(sector_etf, 20)  # default 20
    fair_value_naive = naive_fair_value(data.get("forward_eps"), sector_pe)
    fair_upside = compute_upside(current_price, fair_value_naive)

    return {
        "target_mean": data.get("target_mean"),
        "target_high": data.get("target_high"),
        "target_low": data.get("target_low"),
        "target_median": data.get("target_median"),
        "upside_mean_pct": upside_mean,
        "upside_high_pct": upside_high,
        "upside_low_pct": upside_low,
        "recommendation": data.get("recommendation"),
        "recommendation_mean": data.get("recommendation_mean"),
        "num_analysts": data.get("num_analysts"),
        "forward_pe": data.get("forward_pe"),
        "trailing_pe": data.get("trailing_pe"),
        "peg_ratio": data.get("peg_ratio"),
        "forward_eps": data.get("forward_eps"),
        "fair_value_naive": fair_value_naive,
        "fair_upside_pct": fair_upside,
        "sector_pe_used": sector_pe,
        "fifty_two_week_high": data.get("fifty_two_week_high"),
        "fifty_two_week_low": data.get("fifty_two_week_low"),
    }


def render_targets_bullets(snap: dict) -> list[str]:
    """Markdown bullets for the report."""
    if not snap:
        return []
    bullets = []
    tm = snap.get("target_mean")
    th = snap.get("target_high")
    tl = snap.get("target_low")
    if tm:
        um = snap.get("upside_mean_pct")
        upside_str = f" (upside {um:+.1f}%)" if um is not None else ""
        bullets.append(f"**目標價（平均）**: ${tm:.2f}{upside_str}")
    if th and tl:
        uh = snap.get("upside_high_pct")
        ul = snap.get("upside_low_pct")
        bullets.append(f"**目標價區間**: ${tl:.2f} ~ ${th:.2f} "
                       f"(downside {ul:+.1f}% / upside {uh:+.1f}%)")
    if snap.get("num_analysts"):
        rec = snap.get("recommendation", "—").upper()
        rmean = snap.get("recommendation_mean")
        rec_str = f" (mean {rmean:.2f})" if rmean else ""
        bullets.append(f"**分析師評等**: {rec}{rec_str} "
                       f"({snap['num_analysts']} 位分析師)")
    if snap.get("forward_pe"):
        bullets.append(f"**Forward P/E**: {snap['forward_pe']:.1f}")
    if snap.get("peg_ratio"):
        peg = snap['peg_ratio']
        verdict = "(便宜)" if peg < 1 else "(合理)" if peg < 2 else "(偏貴)"
        bullets.append(f"**PEG**: {peg:.2f} {verdict}")
    if snap.get("fair_value_naive"):
        fv = snap["fair_value_naive"]
        fu = snap.get("fair_upside_pct")
        fu_str = f" (相對現價 {fu:+.1f}%)" if fu is not None else ""
        bullets.append(f"**簡易 Fair Value** "
                       f"(forward EPS × sector PE {snap['sector_pe_used']}): "
                       f"${fv:.2f}{fu_str}")
    if snap.get("fifty_two_week_high") and snap.get("fifty_two_week_low"):
        bullets.append(f"**52w 區間**: "
                       f"${snap['fifty_two_week_low']:.2f} ~ "
                       f"${snap['fifty_two_week_high']:.2f}")
    return bullets


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--price", type=float, default=0)
    parser.add_argument("--sector", default="SMH")
    args = parser.parse_args()
    snap = get_targets_snapshot(args.ticker, args.price, args.sector)
    print(json.dumps(snap, indent=2, ensure_ascii=False, default=str))
    print()
    for b in render_targets_bullets(snap):
        print(f"  • {b}")
