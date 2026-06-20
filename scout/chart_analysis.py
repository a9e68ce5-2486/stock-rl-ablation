"""Classical technical chart analysis layered on top of the model.

The current feature set captures the basics (RSI, MA200 distance, drawdown,
momentum, volume) but misses pattern-recognition signals that human chartists
look for. This module computes them at evaluation time per ticker:

  1. MACD       — 12d/26d EMA crossover + 9d signal line
  2. Bollinger  — 20d MA ± 2σ band position
  3. Candlestick — recent hammer / engulfing patterns
  4. Support/Resistance — distance to nearest meaningful pivot
  5. Trend slope — linear regression slope (steepness of trend)

These are exposed two ways:
  - String bullets appended to the technical-signal list shown to user
  - A 'chart_signals' block injected into the LLM thesis prompt
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_macd(close: pd.Series) -> dict:
    """MACD = EMA12 - EMA26.  Signal = EMA9 of MACD.  Histogram = MACD - Signal.
    Returns the latest state."""
    if len(close) < 35:
        return {}
    macd = _ema(close, 12) - _ema(close, 26)
    signal = _ema(macd, 9)
    hist = macd - signal

    latest_macd = float(macd.iloc[-1])
    latest_signal = float(signal.iloc[-1])
    latest_hist = float(hist.iloc[-1])

    # Detect crossover in last 5 days
    cross = "none"
    for i in range(-5, 0):
        if i - 1 < -len(macd):
            continue
        prev_diff = macd.iloc[i - 1] - signal.iloc[i - 1]
        curr_diff = macd.iloc[i] - signal.iloc[i]
        if prev_diff < 0 and curr_diff > 0:
            cross = "bullish"   # golden cross
            break
        if prev_diff > 0 and curr_diff < 0:
            cross = "bearish"   # death cross
            break

    hist_direction = "rising" if latest_hist > hist.iloc[-2] else "falling"
    above_zero = latest_macd > 0

    return {
        "macd": latest_macd,
        "signal": latest_signal,
        "histogram": latest_hist,
        "cross_5d": cross,
        "histogram_dir": hist_direction,
        "above_zero": above_zero,
    }


def compute_bollinger(close: pd.Series, window: int = 20, std_mult: float = 2.0) -> dict:
    if len(close) < window:
        return {}
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + std_mult * std
    lower = ma - std_mult * std

    latest = float(close.iloc[-1])
    latest_ma = float(ma.iloc[-1])
    latest_up = float(upper.iloc[-1])
    latest_low = float(lower.iloc[-1])
    band_width = (latest_up - latest_low) / latest_ma if latest_ma else 0
    # Position in band: 0 = at lower, 0.5 = middle, 1 = at upper
    position = (latest - latest_low) / (latest_up - latest_low + 1e-9)
    position = float(np.clip(position, -0.5, 1.5))

    state = (
        "below_lower"   if position < 0   else
        "near_lower"    if position < 0.2 else
        "near_middle"   if position < 0.8 else
        "near_upper"    if position < 1.0 else
        "above_upper"
    )
    return {
        "upper": latest_up,
        "middle": latest_ma,
        "lower": latest_low,
        "position": position,
        "state": state,
        "band_width_pct": band_width * 100,
    }


def detect_candlestick_patterns(ohlc: pd.DataFrame, lookback: int = 5) -> list[str]:
    """Detect hammer / engulfing / doji in the last `lookback` days."""
    if len(ohlc) < lookback + 2:
        return []
    patterns = []
    recent = ohlc.iloc[-lookback:]

    for idx in range(len(recent)):
        row = recent.iloc[idx]
        # Need O/H/L/C
        o, h, l, c = (row.get("Open"), row.get("High"),
                      row.get("Low"), row.get("Close"))
        if any(v is None or pd.isna(v) for v in (o, h, l, c)):
            continue
        body = abs(c - o)
        full = h - l
        if full <= 0:
            continue
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l

        # Hammer: small body, long lower wick (≥2× body), upper wick small
        if (body / full < 0.4 and lower_wick > 2 * body
            and upper_wick < body and c > o):
            patterns.append(f"hammer ({recent.index[idx].date()})")

        # Doji: tiny body
        if body / full < 0.1:
            patterns.append(f"doji ({recent.index[idx].date()})")

    # Bullish engulfing: current green body fully covers prev red body
    if len(ohlc) >= 2:
        prev = ohlc.iloc[-2]
        curr = ohlc.iloc[-1]
        p_o, p_c = prev.get("Open"), prev.get("Close")
        c_o, c_c = curr.get("Open"), curr.get("Close")
        if all(v is not None and not pd.isna(v) for v in (p_o, p_c, c_o, c_c)):
            if p_c < p_o and c_c > c_o and c_o < p_c and c_c > p_o:
                patterns.append(f"bullish engulfing ({ohlc.index[-1].date()})")
            elif p_c > p_o and c_c < c_o and c_o > p_c and c_c < p_o:
                patterns.append(f"bearish engulfing ({ohlc.index[-1].date()})")

    # Dedupe while preserving order
    seen = set()
    out = []
    for p in patterns:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def compute_support_resistance(close: pd.Series) -> dict:
    """Find nearest support (recent low) and resistance (recent high)."""
    if len(close) < 60:
        return {}
    latest = float(close.iloc[-1])
    sup_60 = float(close.iloc[-60:].min())
    sup_252 = float(close.iloc[-252:].min()) if len(close) >= 252 else sup_60
    res_60 = float(close.iloc[-60:].max())
    res_252 = float(close.iloc[-252:].max()) if len(close) >= 252 else res_60

    return {
        "to_60d_low_pct": (sup_60 - latest) / latest * 100,
        "to_60d_high_pct": (res_60 - latest) / latest * 100,
        "to_252d_low_pct": (sup_252 - latest) / latest * 100,
        "to_252d_high_pct": (res_252 - latest) / latest * 100,
    }


def compute_trend_slope(close: pd.Series, window: int = 60) -> dict:
    """Slope of linear regression on the last `window` days, normalized."""
    if len(close) < window:
        return {}
    recent = close.iloc[-window:]
    x = np.arange(len(recent))
    y = recent.values
    # Slope per day, then express as % of mean
    slope, _ = np.polyfit(x, y, 1)
    mean = y.mean()
    pct_per_week = slope * 5 / mean * 100 if mean else 0
    trend = (
        "strong uptrend"   if pct_per_week > 1.5  else
        "mild uptrend"     if pct_per_week > 0.3  else
        "sideways"         if pct_per_week > -0.3 else
        "mild downtrend"   if pct_per_week > -1.5 else
        "strong downtrend"
    )
    return {"slope_pct_per_week": pct_per_week, "trend": trend}


def analyze_chart(ticker: str, history: pd.DataFrame) -> dict:
    """Run all chart analyses. `history` should have at least 'Close' column,
    and optionally OHLC for candlestick. Returns a dict of all signals.
    """
    close = history["Close"].astype(float) if "Close" in history.columns else history.iloc[:, 0].astype(float)
    out = {
        "macd": compute_macd(close),
        "bollinger": compute_bollinger(close),
        "support_resistance": compute_support_resistance(close),
        "trend": compute_trend_slope(close, window=60),
        "candlesticks": detect_candlestick_patterns(history)
            if all(c in history.columns for c in ("Open", "High", "Low", "Close")) else [],
    }
    return out


def signals_to_bullets(analysis: dict) -> list[str]:
    """Human-readable bullets for the user-facing report."""
    bullets = []

    macd = analysis.get("macd", {})
    if macd:
        cross = macd.get("cross_5d", "none")
        hist_dir = macd.get("histogram_dir", "")
        if cross == "bullish":
            bullets.append("**MACD 5 日內出現黃金交叉**（多頭訊號）")
        elif cross == "bearish":
            bullets.append("⚠️ MACD 5 日內出現死亡交叉（空頭訊號）")
        else:
            if hist_dir == "rising" and not macd.get("above_zero"):
                bullets.append("MACD histogram 從負轉正中（可能正在轉折）")
            elif hist_dir == "rising":
                bullets.append("MACD 多頭動能延續")

    bb = analysis.get("bollinger", {})
    if bb:
        state = bb.get("state", "")
        if state == "below_lower":
            bullets.append("**跌破 Bollinger 下緣** — 統計上超賣")
        elif state == "near_lower":
            bullets.append("接近 Bollinger 下緣")
        elif state == "above_upper":
            bullets.append("⚠️ 突破 Bollinger 上緣 — 統計上超買")

    sr = analysis.get("support_resistance", {})
    if sr:
        to_60_low = sr.get("to_60d_low_pct", 0)
        to_60_high = sr.get("to_60d_high_pct", 0)
        if abs(to_60_low) < 3:
            bullets.append(f"⚠️ 接近 60 日低點（剩 {abs(to_60_low):.1f}%）— 關鍵支撐")
        if to_60_high > 0 and to_60_high < 5:
            bullets.append(f"接近 60 日高點 (距離 {to_60_high:.1f}%) — 壓力")

    trend = analysis.get("trend", {})
    if trend:
        t = trend.get("trend", "")
        pct = trend.get("slope_pct_per_week", 0)
        if t and "down" in t:
            bullets.append(f"60 日趨勢: {t} ({pct:+.1f}% / 週)")
        elif t == "sideways":
            bullets.append(f"60 日盤整（{pct:+.1f}% / 週）— 可能在 base building")
        elif t and "up" in t:
            bullets.append(f"60 日趨勢: {t} ({pct:+.1f}% / 週)")

    candles = analysis.get("candlesticks", [])
    if candles:
        # Show up to 2 most recent
        for c in candles[-2:]:
            if "hammer" in c:
                bullets.append(f"**K 棒：{c}** — 經典反轉訊號")
            elif "bullish engulfing" in c:
                bullets.append(f"**K 棒：{c}** — 強多頭吞噬")
            elif "bearish engulfing" in c:
                bullets.append(f"⚠️ K 棒：{c} — 空頭吞噬")
            elif "doji" in c:
                bullets.append(f"K 棒：{c} — 多空僵持")

    return bullets


def signals_to_prompt_block(analysis: dict) -> str:
    """Compact block for the LLM prompt."""
    lines = []
    macd = analysis.get("macd", {})
    if macd:
        lines.append(f"- MACD: {macd.get('macd', 0):.2f}, "
                     f"signal {macd.get('signal', 0):.2f}, "
                     f"5日內{macd.get('cross_5d', 'no')}交叉, "
                     f"histogram {macd.get('histogram_dir', '')}")
    bb = analysis.get("bollinger", {})
    if bb:
        lines.append(f"- Bollinger: 在 band 內 {bb.get('position', 0)*100:.0f}% 位置 "
                     f"({bb.get('state', 'unknown')})，band 寬度 {bb.get('band_width_pct', 0):.1f}%")
    sr = analysis.get("support_resistance", {})
    if sr:
        lines.append(f"- 距 60d 低點: {sr.get('to_60d_low_pct', 0):+.1f}% / "
                     f"距 60d 高點: {sr.get('to_60d_high_pct', 0):+.1f}% / "
                     f"距 252d 低: {sr.get('to_252d_low_pct', 0):+.1f}% / "
                     f"距 252d 高: {sr.get('to_252d_high_pct', 0):+.1f}%")
    trend = analysis.get("trend", {})
    if trend:
        lines.append(f"- 60d 趨勢: {trend.get('trend', 'unknown')} "
                     f"({trend.get('slope_pct_per_week', 0):+.2f}% / 週)")
    candles = analysis.get("candlesticks", [])
    if candles:
        lines.append(f"- 最近 5 日 K 棒型態: {', '.join(candles[-3:])}")
    return "\n".join(lines) if lines else ""


def fetch_history(ticker: str, days: int = 300) -> pd.DataFrame:
    """Wrapper to pull OHLC for a ticker."""
    try:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period=f"{days}d", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="MU")
    args = parser.parse_args()
    h = fetch_history(args.ticker)
    if h.empty:
        print(f"❌ no data for {args.ticker}")
        sys.exit(1)
    res = analyze_chart(args.ticker, h)
    print(f"📊 Chart analysis for {args.ticker}\n")
    print("── User-facing bullets ──")
    for b in signals_to_bullets(res):
        print(f"  • {b}")
    print("\n── LLM prompt block ──")
    print(signals_to_prompt_block(res))
