"""Sector-specific news for each ticker.

For every ticker in our universe, we know its sector. We fetch news from
the corresponding sector ETF and inject those headlines into the thesis,
so the LLM sees what's happening to the broader sector (not just the
ticker itself or the whole market).

Example: MU → SMH (semiconductors) → news about HBM demand, chip cycle, etc.
"""
from __future__ import annotations
from typing import List

# Map ticker → sector ETF (best representative for that ticker's sector)
SECTOR_ETF = {
    # Semiconductors / Chips
    **{t: "SMH" for t in [
        "MU", "AMD", "MRVL", "ON", "AVGO", "QCOM", "LSCC", "MCHP", "MPWR",
        "AMAT", "LRCX", "KLAC", "AMKR", "CRUS", "POWI", "SWKS", "QRVO",
        "INTC", "ACMR", "WDC", "STX",
    ]},
    # Optical / Network / Communication
    **{t: "PAVE" for t in ["LITE", "CIEN", "INFN", "ANET", "NTGR"]},
    **{t: "IGM" for t in ["JNPR", "FFIV", "EXTR"]},

    # Cloud / Software
    **{t: "WCLD" for t in [
        "SNOW", "NET", "DDOG", "ZS", "MDB", "OKTA", "CRWD", "PANW", "FTNT",
        "TEAM", "ADSK", "CDNS", "SNPS", "VEEV", "WDAY", "PAYC", "PCTY",
        "PEGA", "ZBRA", "DOCN", "TWLO", "WIX",
    ]},

    # Solar / Renewable
    **{t: "TAN" for t in [
        "FSLR", "ENPH", "SEDG", "RUN", "ARRY", "NOVA", "DAR",
    ]},
    **{t: "ICLN" for t in ["PLUG"]},

    # Biotech / Healthcare
    **{t: "XBI" for t in [
        "VRTX", "REGN", "ILMN", "MRNA", "BMRN", "ALNY", "INCY", "EXEL",
        "GILD",
    ]},
    **{t: "XLV" for t in ["BMY"]},

    # Consumer Discretionary
    **{t: "XLY" for t in [
        "CMG", "LULU", "ULTA", "DECK", "RH", "BBY", "FIVE", "DKS",
        "DASH", "ABNB", "BKNG", "EXPE", "F", "GM",
    ]},

    # Energy / Oil & Gas
    **{t: "XLE" for t in [
        "APA", "DVN", "CTRA", "FANG", "EOG", "MRO", "OXY",
    ]},

    # Financials
    **{t: "XLF" for t in [
        "SCHW", "BX", "KKR", "MS", "GS", "BAC", "C",
    ]},

    # Industrials / Construction
    **{t: "XLI" for t in [
        "BLDR", "BLD", "URI", "WSO", "RPM", "FAST", "TT", "LII",
    ]},

    # Auto / EV
    **{t: "DRIV" for t in ["RIVN", "LCID", "TSLA"]},

    # Crypto / Bitcoin proxies
    **{t: "BLOK" for t in ["MSTR", "MARA", "RIOT", "HUT", "COIN"]},

    # Materials
    **{t: "XLB" for t in ["CLF", "X", "NUE", "STLD"]},

    # Defense / Aerospace
    **{t: "ITA" for t in ["LMT", "RTX", "NOC", "GD", "LDOS"]},

    # Growth-y misc
    **{t: "ARKK" for t in ["PLTR", "S", "U", "RBLX"]},
}


def get_sector_etf(ticker: str) -> str | None:
    """Return the sector ETF ticker for this stock, or None if unknown."""
    return SECTOR_ETF.get(ticker)


def fetch_sector_news(ticker: str, max_items: int = 4) -> tuple[str, List[str]]:
    """Returns (sector_etf, headlines). Empty headlines if no mapping or fetch fails."""
    etf = get_sector_etf(ticker)
    if not etf:
        return "", []
    try:
        import yfinance as yf
        t = yf.Ticker(etf)
        raw = getattr(t, "news", None) or []
        headlines = []
        for item in raw[:max_items]:
            content = item.get("content") if isinstance(item, dict) else None
            if content:
                title = content.get("title") or content.get("summary")
                publisher = (content.get("provider") or {}).get("displayName", "")
            else:
                title = item.get("title", "") if isinstance(item, dict) else ""
                publisher = item.get("publisher", "") if isinstance(item, dict) else ""
            if title:
                headlines.append(f"{title}" + (f" ({publisher})" if publisher else ""))
        return etf, headlines
    except Exception:
        return etf, []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MU")
    args = parser.parse_args()
    etf, headlines = fetch_sector_news(args.ticker, max_items=5)
    print(f"Ticker: {args.ticker}")
    print(f"Sector ETF: {etf}")
    print(f"Headlines ({len(headlines)}):")
    for h in headlines:
        print(f"  - {h}")
