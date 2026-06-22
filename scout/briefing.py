"""Consolidate the daily agent + watchlist reports into a single HTML briefing.

Reads today's results/daily/agent_<date>.md and watchlist_<date>.md, builds
an executive summary at the top (top picks + watchlist tiers in a compact
table), then includes the full reports rendered via marked.js (so any
browser can render markdown without extra deps).

Output: results/daily/briefing_<date>.html
"""
from __future__ import annotations
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DAILY_DIR = PROJECT_ROOT / "results" / "daily"


def extract_agent_summary(md_text: str) -> list[dict]:
    """From agent markdown, pull (rank, ticker, score, price, confidence summary)."""
    picks = []
    # Look for headers like "## #1  LCID — score 0.2437"
    for m in re.finditer(r"##\s+#(\d+)\s+(\S+)\s+—\s+score\s+([\d.]+)", md_text):
        rank, ticker, score = m.group(1), m.group(2), float(m.group(3))
        # Find Price line nearby
        tail = md_text[m.end(): m.end() + 1500]
        price_m = re.search(r"\*\*Price\*\*:\s+\$([\d.,]+)", tail)
        price = price_m.group(1) if price_m else "?"
        # Find confidence in 一句話結論 (within reasonable range)
        conf_m = re.search(r"一句話結論\s*\n+([^\n]+)", tail)
        snippet = conf_m.group(1).strip() if conf_m else ""
        # Try to extract just the confidence keyword
        conf = ""
        for k in ["高", "中高", "中", "中低", "低"]:
            if k in snippet:
                conf = k
                break
        picks.append({
            "rank": rank, "ticker": ticker, "score": score,
            "price": price, "summary": snippet[:120], "confidence": conf,
        })
    return picks


def extract_watch_summary(md_text: str) -> list[dict]:
    """Extract ticker, recommendation tier, net score, change %."""
    items = []
    # Header like "## NVDA  🟢 **ACCUMULATE**"
    for m in re.finditer(r"##\s+([A-Z]+)\s+([🟢🟡🔴⚪]+)\s+\*\*(\w+)\*\*", md_text):
        ticker, emoji, tier = m.group(1), m.group(2), m.group(3)
        tail = md_text[m.end(): m.end() + 800]
        price_m = re.search(r"\*\*Price\*\*:\s+\$([\d.,]+)", tail)
        change_m = re.search(r"\*\*1d\*\*:\s+([+-][\d.]+)%.+?\*\*5d\*\*:\s+([+-][\d.]+)%.+?\*\*30d\*\*:\s+([+-][\d.]+)%",
                              tail)
        net_m = re.search(r"Net signal score\*\*:\s+([+-]?\d+)", tail)
        items.append({
            "ticker": ticker,
            "emoji": emoji,
            "tier": tier,
            "price": price_m.group(1) if price_m else "?",
            "ch1d": change_m.group(1) if change_m else "0",
            "ch5d": change_m.group(2) if change_m else "0",
            "ch30d": change_m.group(3) if change_m else "0",
            "net": net_m.group(1) if net_m else "0",
        })
    return items


def extract_portfolio_summary(md_text: str) -> str:
    """Pull the 投組總覽 block + a per-position quick table."""
    # Get the summary block (between "## 投組總覽" and the next "## ...")
    m = re.search(r"##\s+投組總覽\s*\n+(.+?)(?=\n##\s)", md_text, re.DOTALL)
    summary = m.group(1).strip() if m else ""

    # Split into per-position sections — accept optional flag emoji and
    # both alphanumeric (US) + numeric.TW (Taiwan) ticker formats.
    section_re = re.compile(
        r"##\s+(🇺🇸|🇹🇼)?\s*([0-9A-Z.]+)\s+([🟢🟡🔴⚪]+)"
        r"\s+\*\*(\w+)\*\*\s+\(([🟢🔴])\s+([+-][\d.]+)%\)"
    )
    rows = []
    matches = list(section_re.finditer(md_text))
    for i, m in enumerate(matches):
        flag, ticker, emoji, tier, pl_emoji, pl_pct = m.groups()
        currency = "TWD" if flag == "🇹🇼" else "USD"
        sym = "NT$" if currency == "TWD" else "$"
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        section = md_text[m.end():next_start]

        # All money regexes now accept both $ and NT$
        money = r"(?:NT)?\$"
        price_m = re.search(rf"\*\*現價\*\*:\s+{money}([\d.,]+)", section)
        shares_m = re.search(r"\*\*持股\*\*:\s+([\d.]+)", section)
        cost_m = re.search(rf"\*\*加權均價\*\*:\s+{money}([\d.,]+)", section)
        value_m = re.search(rf"\*\*現值\*\*:\s+{money}([\d,.-]+)", section)
        pl_m = re.search(r"\*\*價差損益\*\*:\s+([+-][\d,.-]+)\s+(?:NT)?\$", section)
        latest_div_m = re.search(r"\*\*每股配息（最近）\*\*:\s+\$([\d.]+)", section)
        annual_div_m = re.search(
            r"\*\*預估年配息（總）\*\*:\s+\$([\d,.-]+)", section)
        freq_m = re.search(r"\*\*配息頻率\*\*:\s+([^\(\n]+)", section)
        yield_m = re.search(r"\*\*年化殖利率\*\*:\s+([\d.]+)%", section)
        target_m = re.search(
            r"\*\*目標價（平均）\*\*:\s+\$([\d.]+)\s+\(upside ([+-]?[\d.]+)%\)",
            section)
        rec_m = re.search(r"\*\*分析師評等\*\*:\s+(\w+)", section)
        pe_m = re.search(r"\*\*Forward P/E\*\*:\s+([\d.]+)", section)
        fv_m = re.search(
            r"\*\*簡易 Fair Value\*\*[^\$]*\$([\d.]+)\s+\(相對現價 ([+-]?[\d.]+)%\)",
            section)
        rows.append({
            "ticker": ticker, "tier": tier, "emoji": emoji,
            "pl_pct": pl_pct, "pl_emoji": pl_emoji,
            "currency": currency, "sym": sym,
            "flag": "🇹🇼" if currency == "TWD" else "🇺🇸",
            "price": price_m.group(1) if price_m else "?",
            "shares": shares_m.group(1) if shares_m else "?",
            "cost": cost_m.group(1) if cost_m else "?",
            "value": value_m.group(1) if value_m else "?",
            "pl_dollar": pl_m.group(1) if pl_m else "0",
            "latest_div_per_share": (latest_div_m.group(1)
                                     if latest_div_m else "—"),
            "annual_div_total": (annual_div_m.group(1)
                                 if annual_div_m else "—"),
            "frequency": (freq_m.group(1).strip() if freq_m else "—"),
            "yield_pct": (yield_m.group(1) if yield_m else "—"),
            "target": (target_m.group(1) if target_m else "—"),
            "upside": (target_m.group(2) if target_m else "—"),
            "rec": (rec_m.group(1) if rec_m else "—"),
            "fwd_pe": (pe_m.group(1) if pe_m else "—"),
            "fair_value": (fv_m.group(1) if fv_m else "—"),
            "fair_upside": (fv_m.group(2) if fv_m else "—"),
        })
    if not rows:
        return f"<p>{summary.replace(chr(10), '<br>')}</p>"

    # Render — sort: USD first, then TWD; within each, by P/L descending
    rows.sort(key=lambda r: (r["currency"] != "USD", -float(r["pl_pct"] or 0)))

    trs_pl = []
    trs_div = []
    trs_target = []
    for r in rows:
        try:
            pl_float = float(r["pl_pct"])
        except ValueError:
            pl_float = 0
        cls = "pos" if pl_float >= 0 else "neg"
        sym = r["sym"]

        try:
            upside_float = float(r["upside"])
            ucls = "pos" if upside_float > 0 else "neg"
        except ValueError:
            ucls = ""

        # Table 1: P/L (currency-aware)
        trs_pl.append(f"""
        <tr>
          <td>{r['flag']}</td>
          <td class="ticker">{r['ticker']}</td>
          <td class="tier-{r['tier']}">{r['emoji']} {r['tier']}</td>
          <td>{r['shares']} sh</td>
          <td>{sym}{r['cost']}</td>
          <td>{sym}{r['price']}</td>
          <td>{sym}{r['value']}</td>
          <td class="{cls}">{r['pl_dollar']} {sym}</td>
          <td class="{cls}"><strong>{r['pl_pct']}%</strong></td>
        </tr>""")

        # Table 2: Dividend (currency-aware)
        trs_div.append(f"""
        <tr>
          <td>{r['flag']}</td>
          <td class="ticker">{r['ticker']}</td>
          <td>{sym}{r['latest_div_per_share']}</td>
          <td>{r['frequency']}</td>
          <td>{sym}{r['annual_div_total']}</td>
          <td>{r['yield_pct']}%</td>
        </tr>""")

        # Table 3: Targets / Fair Value (currency-aware)
        trs_target.append(f"""
        <tr>
          <td>{r['flag']}</td>
          <td class="ticker">{r['ticker']}</td>
          <td>{sym}{r['price']}</td>
          <td>{sym}{r['target']}</td>
          <td class="{ucls}"><strong>{r['upside']}%</strong></td>
          <td>{sym}{r['fair_value']}</td>
          <td>{r['fair_upside']}%</td>
          <td>{r['fwd_pe']}</td>
          <td>{r['rec'].upper()}</td>
        </tr>""")

    return f"""
    <p>{summary.replace(chr(10), '<br>')}</p>

    <h3 style="margin-top:24px;">損益</h3>
    <table>
      <thead>
        <tr>
          <th></th><th>Ticker</th><th>建議</th><th>持股</th>
          <th>成本</th><th>現價</th><th>現值</th>
          <th>價差</th><th>價差 %</th>
        </tr>
      </thead>
      <tbody>{''.join(trs_pl)}</tbody>
    </table>

    <h3 style="margin-top:24px;">配息</h3>
    <table>
      <thead>
        <tr>
          <th></th><th>Ticker</th>
          <th>每股（最近）</th><th>頻率</th><th>預估年配息</th><th>殖利率</th>
        </tr>
      </thead>
      <tbody>{''.join(trs_div)}</tbody>
    </table>

    <h3 style="margin-top:24px;">分析師目標價 / Fair Value</h3>
    <table>
      <thead>
        <tr>
          <th></th><th>Ticker</th><th>現價</th>
          <th>目標價</th><th>Upside</th>
          <th>Fair Value</th><th>FV vs 現價</th>
          <th>Forward P/E</th><th>評等</th>
        </tr>
      </thead>
      <tbody>{''.join(trs_target)}</tbody>
    </table>"""


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<title>🦞 Scout Daily Briefing — {date}</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {{
    --bg: #f7f7f5; --card: #ffffff; --ink: #1a1a1a; --muted: #6c6c6c;
    --green: #16a34a; --red: #dc2626; --amber: #d97706; --blue: #2563eb;
    --border: #e5e5e0;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #181816; --card: #232320; --ink: #e8e8e6; --muted: #9a9a98;
             --border: #36362f; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "SF Pro", system-ui, sans-serif;
          background: var(--bg); color: var(--ink); margin: 0; padding: 24px;
          line-height: 1.55; }}
  .wrap {{ max-width: 1000px; margin: 0 auto; }}
  h1 {{ margin: 0 0 8px; font-size: 28px; }}
  .sub {{ color: var(--muted); margin-bottom: 24px; font-size: 14px; }}
  .card {{ background: var(--card); border: 1px solid var(--border);
           border-radius: 12px; padding: 20px; margin-bottom: 20px; }}
  .card h2 {{ margin-top: 0; font-size: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 10px 8px; text-align: left;
            border-bottom: 1px solid var(--border); }}
  th {{ color: var(--muted); font-weight: 500; font-size: 12px; text-transform: uppercase; }}
  td.ticker {{ font-weight: 700; font-size: 16px; }}
  .pos {{ color: var(--green); }}
  .neg {{ color: var(--red); }}
  .tier-BUY, .tier-ACCUMULATE {{ color: var(--green); font-weight: 700; }}
  .tier-HOLD {{ color: var(--muted); font-weight: 700; }}
  .tier-TRIM, .tier-SELL {{ color: var(--red); font-weight: 700; }}
  details {{ background: var(--card); border: 1px solid var(--border);
             border-radius: 12px; margin-bottom: 16px; padding: 12px 18px; }}
  details > summary {{ cursor: pointer; font-weight: 600; font-size: 17px;
                       padding: 6px 0; }}
  details[open] {{ padding-bottom: 24px; }}
  .md {{ font-size: 14.5px; }}
  .md h1, .md h2, .md h3 {{ margin: 1.4em 0 0.5em; }}
  .md h1 {{ font-size: 22px; }}
  .md h2 {{ font-size: 18px; border-top: 1px solid var(--border);
            padding-top: 16px; }}
  .md h3 {{ font-size: 15px; color: var(--muted); }}
  .md pre {{ background: var(--bg); padding: 12px;
             border-radius: 8px; overflow-x: auto; font-size: 12.5px; }}
  .md code {{ background: var(--bg); padding: 2px 5px; border-radius: 4px;
              font-size: 13px; }}
  .md ul {{ padding-left: 22px; }}
  .md li {{ margin: 4px 0; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px;
            font-size: 11px; font-weight: 600; margin-left: 4px; }}
  .b-high {{ background: #dcfce7; color: #166534; }}
  .b-midhi {{ background: #d1fae5; color: #065f46; }}
  .b-mid {{ background: #fef3c7; color: #92400e; }}
  .b-low {{ background: #fee2e2; color: #991b1b; }}
  .footer {{ color: var(--muted); font-size: 12px; margin-top: 32px;
             padding-top: 16px; border-top: 1px solid var(--border); }}
</style>
</head>
<body>
<div class="wrap">

<h1>🦞 Scout Daily Briefing</h1>
<div class="sub">{date} · 自動產生 · 非投資建議</div>

{indices_card}
{portfolio_block_card}
<div class="card">
  <h2>📡 監看清單建議</h2>
  {watch_table}
</div>

<div class="card">
  <h2>🔍 今日候選（top {n_picks}）</h2>
  {picks_table}
</div>

{portfolio_details}

<details>
  <summary>🔍 完整候選報告 (discovery)</summary>
  <div class="md" id="agent-md"></div>
</details>

<div class="footer">
  Generated by Scout @ {time} · Source files in <code>results/daily/</code>
</div>

</div>

<script>
const agentMd = {agent_json};
document.getElementById('agent-md').innerHTML = marked.parse(agentMd);
</script>
</body>
</html>"""


def build_watch_table(items: list[dict]) -> str:
    if not items:
        return "<p>(無監看資料)</p>"
    rows = []
    for it in items:
        ch1d = float(it["ch1d"])
        ch30d = float(it["ch30d"])
        cls1 = "pos" if ch1d > 0 else "neg" if ch1d < 0 else ""
        cls30 = "pos" if ch30d > 0 else "neg" if ch30d < 0 else ""
        rows.append(f"""
        <tr>
          <td class="ticker">{it['ticker']}</td>
          <td class="tier-{it['tier']}">{it['emoji']} {it['tier']}</td>
          <td>${it['price']}</td>
          <td class="{cls1}">{it['ch1d']}%</td>
          <td>{it['ch5d']}%</td>
          <td class="{cls30}">{it['ch30d']}%</td>
          <td>{it['net']}</td>
        </tr>""")
    return f"""
    <table>
      <thead>
        <tr>
          <th>Ticker</th><th>建議</th><th>價格</th>
          <th>1d</th><th>5d</th><th>30d</th><th>Net</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def conf_class(c: str) -> str:
    return {
        "高": "b-high", "中高": "b-midhi", "中": "b-mid",
        "中低": "b-low", "低": "b-low",
    }.get(c, "")


def build_picks_table(picks: list[dict]) -> str:
    if not picks:
        return "<p>(沒有篩出候選)</p>"
    rows = []
    for p in picks:
        conf_html = (f'<span class="badge {conf_class(p["confidence"])}">'
                     f'{p["confidence"]}</span>' if p["confidence"] else "")
        rows.append(f"""
        <tr>
          <td>#{p['rank']}</td>
          <td class="ticker">{p['ticker']}</td>
          <td>${p['price']}</td>
          <td>{p['score']:.3f}</td>
          <td>{conf_html}</td>
        </tr>""")
    return f"""
    <table>
      <thead>
        <tr>
          <th>#</th><th>Ticker</th><th>價格</th><th>Score</th><th>信心</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None,
                        help="YYYY-MM-DD (default: today)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    agent_path = DAILY_DIR / f"agent_{date_str}.md"
    watch_path = DAILY_DIR / f"watchlist_{date_str}.md"

    if not agent_path.exists() and not watch_path.exists():
        print(f"❌ Neither {agent_path.name} nor {watch_path.name} exists.")
        print("   Run ./run.sh daily-test first.")
        sys.exit(1)

    agent_md = agent_path.read_text() if agent_path.exists() else "_(報告不存在)_"
    watch_md = watch_path.read_text() if watch_path.exists() else "_(報告不存在)_"

    # Optional portfolio
    portfolio_path = DAILY_DIR / f"portfolio_{date_str}.md"
    portfolio_md = ""
    portfolio_summary_html = ""
    if portfolio_path.exists():
        portfolio_md = portfolio_path.read_text()
        # Extract portfolio summary section ("## 投組總覽")
        portfolio_summary_html = extract_portfolio_summary(portfolio_md)

    picks = extract_agent_summary(agent_md)
    watch_items = extract_watch_summary(watch_md)

    import json
    portfolio_card = (f"""<div class="card">
  <h2>📊 我的投組</h2>
  {portfolio_summary_html}
</div>""" if portfolio_summary_html else "")

    # Major indices card
    try:
        from scout.indices import get_all_indices, render_indices_html
        indices_data = get_all_indices()
        indices_html = render_indices_html(indices_data)
        indices_card = (f"""<div class="card">
  <h2>📈 主要指數</h2>
  {indices_html}
</div>""" if indices_html else "")
    except Exception as e:
        indices_card = ""

    portfolio_details = (f"""<details>
  <summary>📊 完整投組報告 (portfolio)</summary>
  <div class="md" id="portfolio-md"></div>
</details>
<script>
document.getElementById('portfolio-md').innerHTML = marked.parse({json.dumps(portfolio_md, ensure_ascii=False)});
</script>""" if portfolio_md else "")

    html = HTML_TEMPLATE.format(
        date=date_str,
        time=datetime.now().strftime("%H:%M"),
        n_picks=len(picks),
        indices_card=indices_card,
        portfolio_block_card=portfolio_card,
        watch_table=build_watch_table(watch_items),
        picks_table=build_picks_table(picks),
        agent_json=json.dumps(agent_md, ensure_ascii=False),
        portfolio_details=portfolio_details,
    )

    out_path = (Path(args.out) if args.out
                else DAILY_DIR / f"briefing_{date_str}.html")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"💾 Briefing saved → {out_path}")
    return str(out_path)


if __name__ == "__main__":
    main()
