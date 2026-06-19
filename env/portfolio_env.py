"""Portfolio allocation environment for RL — Phase 2.

Adds (over Phase 1):
  • Hard position constraints (max per asset, min cash)
  • Asymmetric transaction costs (sell > buy)
  • CVaR-augmented reward (downside-aware)
  • Crisis-period oversampling on reset
  • Random crash injection during training

All new behaviors are off by default (backward compat) — turn on via constructor args.

Critical: state at time `t` uses data through `t-1` close. Action executed at `t` open.
This prevents look-ahead bias.

References (see BIBLIOGRAPHY.md):
  Moody & Saffell 2001         — Differential Sharpe reward
  Jiang, Xu, Liang 2017        — Portfolio-as-softmax action
  Bellemare et al 2017 (C51)   — Distributional RL → CVaR motivation
  Tobin et al 2017             — Domain randomization → crash injection
  López de Prado 2018          — Why ML funds fail (constraints discipline)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


# Historical crisis windows for oversampling (must lie within training data range)
DEFAULT_CRISIS_DATES = [
    "2018-10-03",  # Q4 2018 selloff start
    "2018-12-24",  # Christmas Eve bottom
    "2020-02-20",  # COVID peak before crash
    "2020-03-23",  # COVID bottom
    "2022-01-03",  # 2022 bear market start
    "2022-09-12",  # CPI shock day
]


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        prices: pd.DataFrame,
        features: pd.DataFrame,
        tickers: list[str],
        window: int = 60,
        # Costs (asymmetric: sell typically harder to fill cleanly)
        sell_cost_bps: float = 10.0,
        buy_cost_bps: float = 10.0,
        # Reward shaping
        concentration_lambda: float = 0.1,
        sharpe_window: int = 30,
        cvar_weight: float = 0.0,          # ∈ [0,1]; 0 = pure Sharpe, 1 = pure CVaR
        cvar_pct: float = 5.0,             # CVaR over worst pct% of recent returns
        # Hard constraints
        max_weight_per_asset: float = 1.0,  # 1.0 = no cap
        min_cash: float = 0.0,              # 0.0 = no floor
        # Training-time stochasticity (set to 0 for clean eval)
        crisis_oversample_prob: float = 0.0,
        crisis_dates: list[str] | None = None,
        crash_injection_prob: float = 0.0,
        crash_size_range: tuple[float, float] = (0.05, 0.15),
        seed: int | None = None,
    ):
        super().__init__()
        common = features.index.intersection(prices.index)
        self.features = features.loc[common].astype(np.float32)
        self.close = prices["Close"][tickers].loc[common].astype(np.float32)
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.window = window

        # Costs
        self.sell_cost = sell_cost_bps / 10_000
        self.buy_cost = buy_cost_bps / 10_000

        # Reward
        self.lam = concentration_lambda
        self.sharpe_window = sharpe_window
        self.cvar_weight = cvar_weight
        self.cvar_pct = cvar_pct

        # Constraints
        self.max_weight = max_weight_per_asset
        self.min_cash = min_cash

        # Stochasticity (training only)
        self.crisis_prob = crisis_oversample_prob
        self.crisis_dates = crisis_dates if crisis_dates is not None else DEFAULT_CRISIS_DATES
        self.crash_prob = crash_injection_prob
        self.crash_range = crash_size_range

        # Resolve crisis date → index (drop dates outside data range)
        date_index = self.features.index
        self._crisis_indices = []
        for d in self.crisis_dates:
            try:
                ts = pd.Timestamp(d)
                if ts >= date_index.min() and ts <= date_index.max():
                    idx = date_index.get_indexer([ts], method="nearest")[0]
                    if idx >= self.window:
                        self._crisis_indices.append(idx)
            except (ValueError, KeyError):
                pass

        self._rng = np.random.default_rng(seed)

        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window * n_features,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(self.n_assets + 1,), dtype=np.float32
        )

        self._reset_state()

    def _reset_state(self, start_t: int | None = None):
        if start_t is not None:
            self.t = start_t
        else:
            self.t = self.window
        self.weights = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.weights[-1] = 1.0
        self.return_history: list[float] = []

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Crisis oversampling
        start_t = None
        if self.crisis_prob > 0 and self._crisis_indices and self._rng.random() < self.crisis_prob:
            chosen_idx = self._rng.choice(self._crisis_indices)
            # Start a bit BEFORE the crisis so the agent experiences entering it
            start_t = max(self.window, chosen_idx - 5)

        self._reset_state(start_t=start_t)
        return self._obs(), {}

    def _obs(self) -> np.ndarray:
        window_df = self.features.iloc[self.t - self.window : self.t]
        return window_df.values.flatten().astype(np.float32)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        ex = np.exp(x)
        return ex / ex.sum()

    def _apply_constraints(self, weights: np.ndarray) -> np.ndarray:
        """Project weights onto the constraint set:
        - 0 ≤ w_i ≤ max_weight  (per asset)
        - w_cash ≥ min_cash
        - sum w = 1
        """
        # Cap per-asset
        if self.max_weight < 1.0:
            # Allow cash to be larger than max_weight (it's a special asset)
            weights[:-1] = np.minimum(weights[:-1], self.max_weight)

        # Enforce cash floor
        if self.min_cash > 0 and weights[-1] < self.min_cash:
            deficit = self.min_cash - weights[-1]
            non_cash_total = weights[:-1].sum()
            if non_cash_total > deficit:
                weights[:-1] *= (non_cash_total - deficit) / non_cash_total
                weights[-1] = self.min_cash

        # Renormalize (handles any accumulated drift)
        s = weights.sum()
        if s > 0:
            weights = weights / s
        return weights

    def step(self, action: np.ndarray):
        # Action → portfolio weights (softmax → constraint projection)
        raw_weights = self._softmax(action)
        new_weights = self._apply_constraints(raw_weights.copy())

        # Today's realized returns: close[t] / close[t-1] - 1
        today_returns = (
            self.close.iloc[self.t].values / self.close.iloc[self.t - 1].values - 1
        )

        # Optional crash injection (training-time only)
        if self.crash_prob > 0 and self._rng.random() < self.crash_prob:
            lo, hi = self.crash_range
            shock = -self._rng.uniform(lo, hi)
            today_returns = today_returns + shock  # systemic shock — hits all assets

        asset_returns = np.concatenate([today_returns, [0.0]])

        # Portfolio return: we earn on what we HELD (old weights), then rebalance
        portfolio_return = float((self.weights * asset_returns).sum())

        # Asymmetric transaction costs
        delta = new_weights - self.weights
        buy_amount = float(np.maximum(delta, 0).sum())
        sell_amount = float(np.maximum(-delta, 0).sum())
        cost_drag = self.sell_cost * sell_amount + self.buy_cost * buy_amount

        net_return = portfolio_return - cost_drag
        self.return_history.append(net_return)

        # === Reward design ===
        recent = np.array(self.return_history[-self.sharpe_window :])
        if len(recent) >= 5:
            mean = recent.mean()
            std = recent.std() + 1e-6
            sharpe_proxy = (net_return - mean) / std
        else:
            sharpe_proxy = net_return * 10

        # CVaR: average of worst cvar_pct% of recent returns
        cvar = 0.0
        if self.cvar_weight > 0 and len(recent) >= 10:
            threshold = np.percentile(recent, self.cvar_pct)
            tail = recent[recent <= threshold]
            cvar = float(tail.mean()) if len(tail) > 0 else 0.0

        # Blend Sharpe + CVaR; scale CVaR (raw daily returns are small ~0.01)
        return_reward = (1 - self.cvar_weight) * sharpe_proxy + self.cvar_weight * (cvar * 30)

        # Concentration (Herfindahl)
        concentration = float((new_weights ** 2).sum())
        conc_penalty = -self.lam * concentration

        reward = return_reward + conc_penalty

        # Advance
        self.weights = new_weights
        self.t += 1
        terminated = self.t >= len(self.features) - 1

        info = {
            "portfolio_return": portfolio_return,
            "net_return": net_return,
            "turnover": float(np.abs(delta).sum()),
            "weights": new_weights.copy(),
            "sharpe_proxy": sharpe_proxy,
            "cvar": cvar,
        }
        return self._obs(), float(reward), terminated, False, info


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.fetch_prices import fetch, compute_features

    tickers = ["AAPL", "MSFT", "NVDA", "SPY"]
    prices = fetch(tickers, "2018-01-01", "2024-01-01")
    feats = compute_features(prices, tickers)

    # Phase 2 config: hard constraints + CVaR + crisis sampling + crash injection
    env = PortfolioEnv(
        prices, feats, tickers,
        window=60,
        sell_cost_bps=30, buy_cost_bps=10,
        max_weight_per_asset=0.30, min_cash=0.10,
        cvar_weight=0.5,
        crisis_oversample_prob=0.3,
        crash_injection_prob=0.02,
    )
    obs, _ = env.reset()
    print(f"Obs shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Crisis indices available: {len(env._crisis_indices)}")

    total_reward = 0.0
    info = None
    for _ in range(200):
        action = env.action_space.sample()
        obs, r, done, _, info = env.step(action)
        total_reward += r
        if done:
            break
    print(f"Random policy 200 steps: total_reward={total_reward:.3f}")
    print(f"Last weights: {info['weights']}")
    print(f"  Max position: {info['weights'][:-1].max():.3f} (limit 0.30)")
    print(f"  Cash:         {info['weights'][-1]:.3f} (floor 0.10)")
