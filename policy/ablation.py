"""Run an ablation study across multiple configs.

Trains a model for each named config in policy/configs.py at a (lower)
timestep budget, then evals each on the same in-sample + out-of-sample windows.
All results land in results/runs.csv for side-by-side comparison.

Usage:
    ./run.sh ablation                    # default: 100k steps × 7 configs (~1-2 hr)
    TOTAL_TIMESTEPS=50000 ./run.sh ablation   # faster (~30 min) but less converged
    CONFIGS="phase1_baseline,phase2_relaxed" ./run.sh ablation   # only some
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from policy import configs
from policy.results_io import print_compare_table


# Default ablation set — covers single-mechanism isolation + relaxed phase 2
DEFAULT_ABLATIONS = [
    "phase1_baseline",
    "constraints_only",
    "asym_cost_only",
    "cvar_only",
    "crisis_only",
    "crash_only",
    "phase2_relaxed",
]

TOTAL_TIMESTEPS = int(os.environ.get("TOTAL_TIMESTEPS", "100000"))


def _selected_configs() -> list[str]:
    env = os.environ.get("CONFIGS", "")
    if env.strip():
        names = [s.strip() for s in env.split(",") if s.strip()]
    else:
        names = DEFAULT_ABLATIONS
    # Validate
    available = configs.list_configs()
    bad = [n for n in names if n not in available]
    if bad:
        print(f"❌ Unknown configs: {bad}")
        print(f"   Available: {available}")
        sys.exit(1)
    return names


def run_one(config_name: str) -> bool:
    """Train + eval a single config. Returns True on success."""
    run_id = config_name  # keep them aligned

    print()
    print("█" * 60)
    print(f"█  Running ablation: {config_name}")
    print(f"█  ({TOTAL_TIMESTEPS:,} steps)")
    print("█" * 60)
    print()

    env = os.environ.copy()
    env["CONFIG"] = config_name
    env["RUN_ID"] = run_id
    env["TOTAL_TIMESTEPS"] = str(TOTAL_TIMESTEPS)

    t0 = time.time()
    train_proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "policy" / "train_ppo.py")],
        env=env,
    )
    if train_proc.returncode != 0:
        print(f"❌ Train failed for {config_name}")
        return False

    model_path = PROJECT_ROOT / "policy" / "checkpoints" / f"ppo_{run_id}.zip"
    eval_proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "policy" / "eval.py"),
         str(model_path), run_id, config_name],
        env=env,
    )
    if eval_proc.returncode != 0:
        print(f"❌ Eval failed for {config_name}")
        return False

    elapsed = time.time() - t0
    print(f"\n⏱️  {config_name} done in {elapsed/60:.1f} min")
    return True


def main():
    names = _selected_configs()
    total = len(names)
    est_minutes = total * (TOTAL_TIMESTEPS / 100_000) * 8  # rough estimate

    print(f"🧪 Ablation study: {total} configs × {TOTAL_TIMESTEPS:,} steps")
    print(f"   Estimated time: ~{est_minutes:.0f} minutes")
    print(f"   Configs: {names}\n")

    successes = []
    failures = []
    for name in names:
        ok = run_one(name)
        (successes if ok else failures).append(name)

    print("\n" + "=" * 60)
    print(f"🦞 Ablation complete: {len(successes)}/{total} succeeded")
    if failures:
        print(f"   Failed: {failures}")
    print("=" * 60)
    print("\n📋 Final comparison table:\n")
    print_compare_table()
    print(f"\n💾 Full results: results/runs.csv")
    print(f"💡 Tip: open results/runs.csv in Excel/Numbers for sortable view")


if __name__ == "__main__":
    main()
