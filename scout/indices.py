"""Major US + Taiwan market indices snapshot.

Shows current values + 1d / 5d / 30d performance for the major indices
that frame the portfolio context.
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# (ticker, display name, region)
INDICES = [
    ("^GSPC", "S&P 500",        "US"),
    ("^IXIC", "NASDAQ",         "US"),
    ("^DJI",  "Dow Jones",      "US"),
    ("^RUT",  "Russell 2000",   "US"),
    ("^SOX",  "費城半導體",       "US"),
    ("^VIX",  "VIX 恐慌指數",     "US"),
    ("^TWII", "台灣加權指數",     "TW"),
    ("^TWOII", "櫃買指數 OTC",    "TW"),
]


def fetch_index(ticker: str) -> dict:
    """Returns latest close + change pcts (1d, 5d, 30d) + 52w distance."""
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="1y")
        if hist.empty:
            return {}
        close = hist["Close"]
        latest = float(close.iloc[-1])
        ch1d = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) >= 2 else None
        ch5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else None
        ch30d = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) >= 22 else None
        high_52w = float(close.max())
        low_52w = float(close.min())
        from_high = (latest / high_52w - 1) * 100
        from_low = (latest / low_52w - 1) * 100
        return {
            "close": latest,
            "ch1d": ch1d,
            "ch5d": ch5d,
            "ch30d": ch30d,
            "high_52w": high_52w,
            "low_52w": low_52w,
            "from_52w_high_pct": from_high,
            "from_52w_low_pct": from_low,
        }
    except Exception:
        return {}


def get_all_indices() -> list[dict]:
    out = []
    for ticker, name, region in INDICES:
        data = fetch_index(ticker)
        if not data:
            continue
        out.append({"ticker": ticker, "name": name, "region": region, **data})
    return out


def render_indices_table_md(indices: list[dict]) -> str:
    """Markdown table grouped by region."""
    lines = []
    for region in ("US", "TW"):
        regional = [x for x in indices if x["region"] == region]
        if not regional:
            continue
        flag = "🇺🇸" if region == "US" else "🇹🇼"
        lines.append(f"\n#### {flag} {region}")
        lines.append("")
        lines.append("| 指數 | 收盤 | 1d | 5d | 30d | 距 52w 高 | 距 52w 低 |")
        lines.append("|------|------|----|----|-----|----------|----------|")
        for x in regional:
            sign = lambda v: "🟢" if (v or 0) >= 0 else "🔴"
            ch1d_str = f"{sign(x.get('ch1d'))} {x.get('ch1d', 0):+.2f}%" if x.get('ch1d') is not None else "—"
            ch5d_str = f"{x.get('ch5d', 0):+.2f}%" if x.get('ch5d') is not None else "—"
            ch30d_str = f"{x.get('ch30d', 0):+.2f}%" if x.get('ch30d') is not None else "—"
            lines.append(f"| {x['name']} ({x['ticker']}) | "
                         f"{x.get('close', 0):,.2f} | "
                         f"{ch1d_str} | {ch5d_str} | {ch30d_str} | "
                         f"{x.get('from_52w_high_pct', 0):+.1f}% | "
                         f"{x.get('from_52w_low_pct', 0):+.1f}% |")
    return "\n".join(lines)


def render_indices_html(indices: list[dict]) -> str:
    """Compact HTML table grouped by region."""
    blocks = []
    for region in ("US", "TW"):
        regional = [x for x in indices if x["region"] == region]
        if not regional:
            continue
        flag = "🇺🇸" if region == "US" else "🇹🇼"
        rows = []
        for x in regional:
            ch1d = x.get("ch1d") or 0
            ch5d = x.get("ch5d") or 0
            ch30d = x.get("ch30d") or 0
            cls1 = "pos" if ch1d >= 0 else "neg"
            cls5 = "pos" if ch5d >= 0 else "neg"
            cls30 = "pos" if ch30d >= 0 else "neg"
            rows.append(f"""
            <tr>
              <td><strong>{x['name']}</strong></td>
              <td style="color:var(--muted);">{x['ticker']}</td>
              <td>{x.get('close', 0):,.2f}</td>
              <td class="{cls1}">{ch1d:+.2f}%</td>
              <td class="{cls5}">{ch5d:+.2f}%</td>
              <td class="{cls30}">{ch30d:+.2f}%</td>
              <td style="color:var(--muted);">{x.get('from_52w_high_pct', 0):+.1f}%</td>
            </tr>""")
        blocks.append(f"""
        <h3>{flag} {region}</h3>
        <table>
          <thead>
            <tr><th>指數</th><th>Ticker</th><th>收盤</th>
                <th>1d</th><th>5d</th><th>30d</th><th>距 52w 高</th></tr>
          </thead>
          <tbody>{"".join(rows)}</tbody>
        </table>""")
    return "\n".join(blocks)


if __name__ == "__main__":
    indices = get_all_indices()
    print(render_indices_table_md(indices))
