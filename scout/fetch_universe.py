"""Mid-cap US stock universe for rally detection.

Curated list of ~140 mid-cap US stocks across sectors. NOT survivorship-bias-free
(some current mid-caps were small or didn't exist in 2015) but adequate for v0.

Future improvements:
- Use point-in-time S&P 400 / Russell midcap holdings
- Add delisted stocks (need paid data)
"""
from __future__ import annotations

UNIVERSE = [
    # === Semiconductors ===
    "MU",   # Micron — your benchmark case
    "AMD", "MRVL", "ON", "AVGO", "QCOM", "LSCC", "MCHP", "MPWR",
    "AMAT", "LRCX", "KLAC", "AMKR", "CRUS", "POWI", "SWKS", "QRVO",
    "WDC", "STX",  # Storage
    "INTC",        # Large, useful baseline

    # === Optical / Networking ===
    "LITE",  # Lumentum — your benchmark case
    "ANET", "JNPR", "FFIV", "CIEN", "INFN", "NTGR", "ACMR", "EXTR",

    # === Cloud / Software (some IPO'd post-2015) ===
    "SNOW", "NET", "DDOG", "ZS", "MDB", "OKTA", "CRWD", "PANW", "FTNT",
    "TEAM", "ADSK", "CDNS", "SNPS", "VEEV", "WDAY", "PAYC", "PCTY",
    "PEGA", "ZBRA", "DOCN", "TWLO", "WIX",

    # === Solar / Clean Energy ===
    "FSLR", "ENPH", "SEDG", "RUN", "ARRY", "NOVA", "DAR", "PLUG",

    # === Biotech / Pharma ===
    "VRTX", "REGN", "ILMN", "MRNA", "BMRN", "ALNY", "INCY",
    "EXEL", "BMY", "GILD",

    # === Consumer Discretionary ===
    "CMG", "LULU", "ULTA", "DECK", "RH", "BBY", "FIVE", "DKS",
    "DASH", "ABNB", "BKNG", "EXPE",

    # === Energy / Oil & Gas ===
    "APA", "DVN", "CTRA", "FANG", "EOG", "MRO", "OXY",

    # === Financials ===
    "SCHW", "COIN", "BX", "KKR", "MS", "GS", "BAC", "C",

    # === Industrials ===
    "BLDR", "BLD", "URI", "WSO", "RPM", "FAST", "TT", "LII",

    # === Auto / EV ===
    "RIVN", "LCID", "F", "GM", "TSLA",

    # === Crypto / Bitcoin proxies ===
    "MSTR", "MARA", "RIOT", "HUT",

    # === Materials ===
    "CLF", "X", "NUE", "STLD",

    # === Defense / Aerospace ===
    "LMT", "RTX", "NOC", "GD", "LDOS",

    # === Misc growth ===
    "PLTR", "S", "U", "RBLX",

    # === Benchmark (always include) ===
    "SPY", "QQQ", "IWM",
]


def list_universe() -> list[str]:
    return list(UNIVERSE)


if __name__ == "__main__":
    print(f"Universe size: {len(UNIVERSE)}")
    print(f"Includes MU: {'MU' in UNIVERSE}")
    print(f"Includes LITE: {'LITE' in UNIVERSE}")
    print(f"Includes ENPH: {'ENPH' in UNIVERSE}")
