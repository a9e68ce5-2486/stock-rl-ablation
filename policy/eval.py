"""Evaluate a saved PPO model on train + test periods. Saves results to results/.

Usage:
    ./run.sh eval                        # default checkpoint + run_id
    python policy/eval.py <model.zip> <run_id> <config_name>
"""
from __future__ import annotations
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3 import PPO
from data.fetch_prices import fetch, compute_features
from env.portfolio_env import PortfolioEnv
from policy.results_io import save_eval_result, print_compare_table

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"]
WINDOW = 60
DEFAULT_MODEL = PROJECT_ROOT / "policy" / "checkpoints" / "ppo_portfolio_final.zip"

# Eval-time env: keep constraints (deployment realism), disable training stochasticity
EVAL_ENV_KWARGS = dict(
    window=WINDOW,
    max_weight_per_asset=0.40,
    min_cash=0.10,
    sell_cost_bps=30.0,
    buy_cost_bps=10.0,
    cvar_weight=0.0,
    crisis_oversample_prob=0.0,
    crash_injection_prob=0.0,
)


def evaluate(model_path: Path, start: str, end: str, label: str,
             run_id: str, config_name: str):
    prices = fetch(TICKERS, start, end)
    feats = compute_features(prices, TICKERS)
    env = PortfolioEnv(prices, feats, TICKERS, **EVAL_ENV_KWARGS)
    model = PPO.load(str(model_path))

    obs, _ = env.reset()
    daily_returns: list[float] = []
    turnovers: list[float] = []
    weights_history: list[np.ndarray] = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(action)
        daily_returns.append(info["net_return"])
        turnovers.append(info["turnover"])
        weights_history.append(info["weights"])
        if done:
            break

    r = np.array(daily_returns)
    cum = r.cumsum()
    max_dd = float((cum - np.maximum.accumulate(cum)).min())
    sharpe = float(r.mean() / (r.std() + 1e-9) * np.sqrt(252))
    avg_turnover = float(np.mean(turnovers))
    avg_weights_arr = np.array(weights_history).mean(axis=0)

    spy_returns = (prices["Close"]["SPY"].pct_change().loc[feats.index].iloc[WINDOW:]).values
    spy_returns = spy_returns[: len(r)]
    spy_cum = float(spy_returns.sum())
    spy_sharpe = float(spy_returns.mean() / (spy_returns.std() + 1e-9) * np.sqrt(252))

    metrics = {
        "steps": len(r),
        "cum_return": float(cum[-1]),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "avg_turnover": avg_turnover,
        "spy_return": spy_cum,
        "spy_sharpe": spy_sharpe,
    }
    avg_weights = {t: float(avg_weights_arr[i]) for i, t in enumerate(TICKERS)}
    avg_weights["CASH"] = float(avg_weights_arr[-1])

    # Print
    print(f"\n{'='*60}")
    print(f"📊 {label} — {start} ~ {end} | config={config_name}")
    print(f"{'='*60}")
    print(f"Steps evaluated:    {metrics['steps']}")
    print(f"Cumulative return:  {metrics['cum_return'] * 100:>+7.2f}%")
    print(f"Annualized Sharpe:  {metrics['sharpe']:>+7.2f}")
    print(f"Max drawdown:       {metrics['max_dd'] * 100:>+7.2f}%")
    print(f"Avg daily turnover: {metrics['avg_turnover'] * 100:>+7.2f}%")
    print(f"📈 Benchmark (SPY buy & hold):")
    print(f"  Cumulative return: {metrics['spy_return'] * 100:>+7.2f}%")
    print(f"  Annualized Sharpe: {metrics['spy_sharpe']:>+7.2f}")
    print(f"⚖️  Average portfolio weights:")
    for t in TICKERS:
        bar = "█" * int(avg_weights[t] * 50)
        print(f"  {t:>6}: {avg_weights[t] * 100:>5.1f}% {bar}")
    cash_bar = "█" * int(avg_weights["CASH"] * 50)
    print(f"  {'CASH':>6}: {avg_weights['CASH'] * 100:>5.1f}% {cash_bar}")

    # Persist
    save_eval_result(
        run_id=run_id,
        config_name=config_name,
        period_label=label,
        period_start=start,
        period_end=end,
        metrics=metrics,
        avg_weights=avg_weights,
        model_path=str(model_path),
        full_returns=daily_returns,
    )
    return metrics, avg_weights


def main():
    # Args: [model_path] [run_id] [config_name]
    model_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MODEL
    run_id = sys.argv[2] if len(sys.argv) > 2 else f"run_{model_path.stem}"
    config_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        sys.exit(1)
    print(f"🦞 Loading model: {model_path.name} (run_id={run_id}, config={config_name})\n")

    evaluate(model_path, "2015-01-01", "2024-01-01", "in_sample", run_id, config_name)
    evaluate(model_path, "2024-01-01", "2026-06-01", "out_of_sample", run_id, config_name)

    print(f"\n{'='*60}")
    print(f"💾 Results saved to results/runs.csv + results/runs/{run_id}.json")
    print(f"{'='*60}")
    print("\n📋 All runs so far:\n")
    print_compare_table()


if __name__ == "__main__":
    main()
