"""Train PPO on PortfolioEnv. Uses a named config from policy/configs.py.

Run via wrapper:
    ./run.sh train                              # uses CONFIG env var or "phase2_relaxed"
    CONFIG=phase1_baseline ./run.sh train       # override config
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from data.fetch_prices import fetch, compute_features
from env.portfolio_env import PortfolioEnv
from policy import configs


# === Config ===
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"]
TRAIN_START = "2015-01-01"
TRAIN_END = "2024-01-01"
WINDOW = 60

# Read config name from env var, default to Plan A (relaxed Phase 2)
CONFIG_NAME = os.environ.get("CONFIG", "phase2_relaxed")
RUN_ID = os.environ.get("RUN_ID", CONFIG_NAME)
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "200000"))

SAVE_DIR = PROJECT_ROOT / "policy" / "checkpoints"
SAVE_DIR.mkdir(exist_ok=True)
MODEL_PATH = SAVE_DIR / f"ppo_{RUN_ID}.zip"


def make_env(env_kwargs: dict):
    prices = fetch(TICKERS, TRAIN_START, TRAIN_END)
    feats = compute_features(prices, TICKERS)
    return PortfolioEnv(prices, feats, TICKERS, **env_kwargs)


def main():
    env_kwargs = configs.get(CONFIG_NAME)

    print("=" * 60)
    print(f"📦 Config: {CONFIG_NAME}  |  run_id: {RUN_ID}")
    print(f"   Total steps: {TOTAL_TIMESTEPS:,}")
    print("=" * 60)
    print("Env settings:")
    for k, v in env_kwargs.items():
        print(f"  {k:>28} = {v}")
    print()

    env = DummyVecEnv([lambda: make_env(env_kwargs)])
    print(f"   N assets: {len(TICKERS)} + cash")
    print(f"   Obs dim : {env.observation_space.shape[0]}")
    print(f"   Action  : {env.action_space.shape[0]}-d continuous")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"   Device  : {device}")

    print("\n🦞 Building PPO...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
        verbose=1,
        device=device,
        tensorboard_log=str(PROJECT_ROOT / "tb_logs" / RUN_ID),
    )

    print(f"\n🚀 Training {RUN_ID} for {TOTAL_TIMESTEPS:,} steps...\n")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000, TOTAL_TIMESTEPS // 4),
        save_path=str(SAVE_DIR),
        name_prefix=f"ppo_{RUN_ID}",
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)
    model.save(str(MODEL_PATH))
    print(f"\n✅ Model saved: {MODEL_PATH}")
    print(f"   Eval: python3 policy/eval.py {MODEL_PATH} {RUN_ID} {CONFIG_NAME}")


if __name__ == "__main__":
    main()
