# 📈 Stock RL — Portfolio allocation with deep RL

Daily-frequency US equity portfolio allocation using PPO.

Built 2026-06-19 by Victor (Sheng-Hung Tseng) as a learning project.

## Goal

Train an RL agent that, given recent price/volume/macro signals, outputs a portfolio
weight vector over ~7 mega-cap stocks + cash. Daily rebalancing. Reward is risk-adjusted
return (Sharpe-proxy) minus transaction costs and concentration.

## Phase 1 (current — MVP)

| Component | Status |
|-----------|--------|
| `data/fetch_prices.py` | ✅ yfinance OHLCV + cache + basic features |
| `env/portfolio_env.py` | ✅ Gym env with Sharpe reward, turnover cost |
| `policy/train_ppo.py` | ✅ PPO baseline (MlpPolicy) on M-series MPS |
| `backtest/walkforward.py` | ⏳ Phase 2 |
| News / macro fusion | ⏳ Phase 2 |

## Run

Use the `run.sh` wrapper — it auto-activates the venv:

```bash
cd ~/Documents/GitHub/AI\ agent/stock-rl

./run.sh test-data     # 1. Sanity check data pipeline
./run.sh test-env      # 2. Sanity check the env
./run.sh train         # 3. Train (10-20 min on MacBook Air M-series)
./run.sh tensorboard   # 4. Watch training live (in another terminal)

./run.sh               # No args = show help
./run.sh install <pkg> # Install package into the venv
./run.sh shell         # Drop into venv Python REPL
```

The wrapper handles `source stock-rl-env/bin/activate` for you, so you never have to think about which Python is active.

## Design decisions (and the reasons)

### Action = portfolio weights (softmax) + cash
Combines B (trade decisions) and C (allocation) into a single continuous control.

### State = past 60 days of per-ticker features
- Log return
- 20-day rolling vol
- 20-day and 60-day momentum
- Volume z-score

60-day window captures both short-term momentum and 1-quarter mean reversion.

### Reward = differential Sharpe − concentration penalty − turnover cost
- Differential Sharpe → encourages **consistent** wins, not lottery tickets
- Concentration (Herfindahl) penalty → prevents "all in NVDA"
- Turnover cost (10 bps default) → discourages day-trading-like behavior

### PPO not SAC (despite my earlier advice)
- PPO is easier to debug → catch reward-function bugs first
- After Phase 1 is stable, swap to SAC (off-policy, better sample efficiency)

### MlpPolicy not Transformer
- Phase 1 = make pipeline work
- Phase 2 = swap to Recurrent PPO or Decision Transformer for sequential modeling

## Known limitations (deliberate, for Phase 1)

- ⚠️ **Survivorship bias**: training on today's mega caps (they won the last decade by definition)
- ⚠️ **No macro/news yet**: just price-derived features
- ⚠️ **In-sample only**: training and quick-eval on same range. Real test = walk-forward on 2024+
- ⚠️ **No execution friction beyond bps cost**: no slippage, market impact, or fill-or-kill

These are all Phase 2/3 concerns.

## Phase 2 (next)

- Add FRED macro indicators (10Y yield, VIX, USD index, etc.)
- Walk-forward backtest with proper train/test split
- Add news embeddings via `all-MiniLM-L6-v2` (runs on CPU)
- Try Recurrent PPO

## Phase 3 (research)

- Decision Transformer (offline RL on years of history)
- Multi-modal fusion (text + numerical via cross-attention)
- Point-in-time S&P 500 to fix survivorship bias
- Move to Colab Free T4 for actual experiments

## Honest expectations

This will probably:
- ✅ Train without crashing
- ✅ Show training loss decrease
- ✅ Beat random allocation on training distribution
- ❌ Beat SPY buy-and-hold on out-of-sample 2024-2026
- ❌ Be profitable in real trading

Most retail RL trading attempts lose to a coin flip. The point is **learning the methodology**, not making money. If you want to actually invest, just buy SPY.
