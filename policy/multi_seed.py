"""Run a config across multiple random seeds to measure result variance.

Single-seed RL results have huge variance. To know whether a config is genuinely
better (and not just lucky), train and eval it with multiple seeds, then look at
mean + standard deviation of Sharpe.

Usage:
    ./run.sh multi-seed                                   # default: phase3_minimal × 5 seeds
    CONFIG=crash_only N_SEEDS=5 ./run.sh multi-seed
    CONFIG=phase3_minimal N_SEEDS=3 TOTAL_TIMESTEPS=100000 ./run.sh multi-seed
"""
from __future__ import annotations
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policy import configs
from policy.results_io import load_runs_table


CONFIG_NAME = os.environ.get("CONFIG", "phase3_minimal")
N_SEEDS = int(os.environ.get("N_SEEDS", "5"))
SEED_START = int(os.environ.get("SEED_START", "0"))  # resume from this seed index
TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))


def run_one(seed: int) -> bool:
    run_id = f"{CONFIG_NAME}_seed{seed}"

    print()
    print("█" * 60)
    print(f"█  Seed {seed} — {CONFIG_NAME}")
    print(f"█  Run ID: {run_id}")
    print("█" * 60)
    print()

    env = os.environ.copy()
    env["CONFIG"] = CONFIG_NAME
    env["RUN_ID"] = run_id
    env["TOTAL_TIMESTEPS"] = str(TOTAL_TIMESTEPS)
    env["PYTHONHASHSEED"] = str(seed)
    env["SEED"] = str(seed)

    t0 = time.time()
    train_proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "policy" / "train_ppo.py")],
        env=env,
    )
    if train_proc.returncode != 0:
        print(f"❌ Train failed: seed {seed}")
        return False

    model_path = PROJECT_ROOT / "policy" / "checkpoints" / f"ppo_{run_id}.zip"
    eval_proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "policy" / "eval.py"),
         str(model_path), run_id, CONFIG_NAME],
        env=env,
    )
    if eval_proc.returncode != 0:
        print(f"❌ Eval failed: seed {seed}")
        return False

    print(f"⏱️  Seed {seed} done in {(time.time() - t0)/60:.1f} min")
    return True


def summarize():
    """Read CSV, pull rows for this config's seeds, compute mean/std of Sharpe."""
    rows = load_runs_table()
    by_period = {"in_sample": [], "out_of_sample": []}

    for row in rows:
        if not row["run_id"].startswith(f"{CONFIG_NAME}_seed"):
            continue
        period = row["period"]
        if period not in by_period:
            continue
        try:
            by_period[period].append({
                "seed": row["run_id"].split("_seed")[-1],
                "sharpe": float(row["sharpe"]),
                "cum_return": float(row["cum_return"]),
                "max_dd": float(row["max_dd"]),
            })
        except (ValueError, KeyError):
            continue

    print("\n" + "=" * 60)
    print(f"📊 Multi-seed summary for {CONFIG_NAME}")
    print("=" * 60)

    for period_name, items in by_period.items():
        if not items:
            continue
        sharpes = [x["sharpe"] for x in items]
        rets = [x["cum_return"] for x in items]
        dds = [x["max_dd"] for x in items]
        n = len(sharpes)
        s_mean = statistics.mean(sharpes)
        s_std = statistics.stdev(sharpes) if n > 1 else 0.0
        r_mean = statistics.mean(rets)
        r_std = statistics.stdev(rets) if n > 1 else 0.0
        dd_mean = statistics.mean(dds)

        print(f"\n  📅 {period_name}  (n={n} seeds)")
        print(f"     Sharpe     : {s_mean:>+6.3f}  ± {s_std:.3f}")
        print(f"     Cum Return : {r_mean*100:>+6.2f}% ± {r_std*100:.2f}%")
        print(f"     Avg Max DD : {dd_mean*100:>+6.2f}%")
        print(f"     Individual : {[f'{x:+.2f}' for x in sharpes]}")

    # Statistical significance test: is mean OOS Sharpe meaningfully positive?
    oos = by_period.get("out_of_sample", [])
    if len(oos) >= 3:
        sharpes = [x["sharpe"] for x in oos]
        mean = statistics.mean(sharpes)
        std = statistics.stdev(sharpes) if len(sharpes) > 1 else 1.0
        # Rough t-statistic (mean - 0) / (std / sqrt(n))
        t_stat = mean / (std / (len(sharpes) ** 0.5) + 1e-9)
        print(f"\n  📐 Significance check (OOS Sharpe > 0):")
        print(f"     mean / SE = {t_stat:.2f}")
        if abs(t_stat) > 2:
            verdict = "✅ Likely a real effect (|t|>2)"
        elif abs(t_stat) > 1:
            verdict = "⚠️  Probably noise (|t|<2)"
        else:
            verdict = "❌ Almost certainly noise (|t|<1)"
        print(f"     Verdict   : {verdict}")
    elif oos:
        print(f"\n  ⚠️  Need ≥3 seeds for significance test (have {len(oos)})")


def main():
    if CONFIG_NAME not in configs.list_configs():
        print(f"❌ Unknown config: {CONFIG_NAME}")
        print(f"   Available: {configs.list_configs()}")
        sys.exit(1)

    seed_end = SEED_START + N_SEEDS
    est_min = N_SEEDS * (TOTAL_TIMESTEPS / 100_000) * 8
    print(f"🌱 Multi-seed run: {CONFIG_NAME} seeds [{SEED_START}..{seed_end-1}] × {TOTAL_TIMESTEPS:,} steps")
    print(f"   Estimated time: ~{est_min:.0f} minutes")
    print()

    for seed in range(SEED_START, seed_end):
        run_one(seed)

    summarize()
    print(f"\n💾 Full results in results/runs.csv")


if __name__ == "__main__":
    main()
