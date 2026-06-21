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

    # Get per-position headers like "## NVDA  🟢 **ACCUMULATE**  (🟢 +17.1%)"
    rows = []
    for m in re.finditer(
        r"##\s+([A-Z]+)\s+([🟢🟡🔴⚪]+)\s+\*\*(\w+)\*\*\s+\(([🟢🔴])\s+([+-][\d.]+)%\)",
        md_text):
        ticker, emoji, tier, pl_emoji, total_ret = m.groups()
        tail = md_text[m.end(): m.end() + 900]
        price_m = re.search(r"\*\*現價\*\*:\s+\$([\d.,]+)", tail)
        shares_m = re.search(r"\*\*持股\*\*:\s+([\d.]+)", tail)
        cost_m = re.search(r"\*\*加權均價\*\*:\s+\$([\d.,]+)", tail)
        value_m = re.search(r"\*\*現值\*\*:\s+\$([\d,.-]+)", tail)
        pl_m = re.search(r"\*\*價差損益\*\*:\s+([+-][\d,.-]+)\$\s+\(([+-][\d.]+)%\)", tail)
        div_m = re.search(r"\*\*累積配息\*\*:\s+\$([\d.,]+)", tail)
        rows.append({
            "ticker": ticker, "tier": tier, "emoji": emoji,
            "total_return_pct": total_ret, "pl_emoji": pl_emoji,
            "price": price_m.group(1) if price_m else "?",
            "shares": shares_m.group(1) if shares_m else "?",
            "cost": cost_m.group(1) if cost_m else "?",
            "value": value_m.group(1) if value_m else "?",
            "pl_dollar": pl_m.group(1) if pl_m else "0",
            "pl_pct": pl_m.group(2) if pl_m else "0",
            "dividends": div_m.group(1) if div_m else "0",
        })
    if not rows:
        return f"<p>{summary.replace(chr(10), '<br>')}</p>"

    # Render
    trs = []
    for r in rows:
        try:
            tr_float = float(r["total_return_pct"])
        except ValueError:
            tr_float = 0
        cls = "pos" if tr_float >= 0 else "neg"
        trs.append(f"""
        <tr>
          <td class="ticker">{r['ticker']}</td>
          <td class="tier-{r['tier']}">{r['emoji']} {r['tier']}</td>
          <td>{r['shares']} sh</td>
          <td>${r['cost']}</td>
          <td>${r['price']}</td>
          <td>${r['value']}</td>
          <td class="{cls}">{r['pl_dollar']}$</td>
          <td>${r['dividends']}</td>
          <td class="{cls}"><strong>{r['total_return_pct']}%</strong></td>
        </tr>""")

    return f"""
    <p>{summary.replace(chr(10), '<br>')}</p>
    <table style="margin-top:16px;">
      <thead>
        <tr>
          <th>Ticker</th><th>建議</th><th>持股</th>
          <th>成本</th><th>現價</th><th>現值</th>
          <th>價差</th><th>配息</th><th>總報酬 %</th>
        </tr>
      </thead>
      <tbody>{''.join(trs)}</tbody>
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
  <summary>📡 完整監看報告 (watchlist)</summary>
  <div class="md" id="watch-md"></div>
</details>

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
const watchMd = {watch_json};
document.getElementById('agent-md').innerHTML = marked.parse(agentMd);
document.getElementById('watch-md').innerHTML = marked.parse(watchMd);
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
        portfolio_block_card=portfolio_card,
        watch_table=build_watch_table(watch_items),
        picks_table=build_picks_table(picks),
        agent_json=json.dumps(agent_md, ensure_ascii=False),
        watch_json=json.dumps(watch_md, ensure_ascii=False),
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
