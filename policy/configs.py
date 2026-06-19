"""Named environment configurations for training and ablation studies.

Each config is a dict of kwargs for PortfolioEnv. The naming convention:
    phase1_baseline   — Phase 1 (no constraints, no robustness tricks)
    + each Phase 2 mechanism in isolation
    phase2_full       — what hurt us
    phase2_relaxed    — Plan A (relaxed strengths)
"""
from __future__ import annotations

# Base (= Phase 1 — original behavior)
_PHASE1 = dict(
    window=60,
    sell_cost_bps=10.0,
    buy_cost_bps=10.0,
    concentration_lambda=0.1,
    sharpe_window=30,
    cvar_weight=0.0,
    max_weight_per_asset=1.0,
    min_cash=0.0,
    crisis_oversample_prob=0.0,
    crash_injection_prob=0.0,
)


def _override(**kwargs):
    out = dict(_PHASE1)
    out.update(kwargs)
    return out


CONFIGS = {
    # === Baseline (no Phase 2 mechanism) ===
    "phase1_baseline": _PHASE1,

    # === Each Phase 2 mechanism in isolation ===
    "constraints_only": _override(
        max_weight_per_asset=0.40,
        min_cash=0.10,
    ),
    "asym_cost_only": _override(
        sell_cost_bps=30.0,
        buy_cost_bps=10.0,
    ),
    "cvar_only": _override(
        cvar_weight=0.25,
    ),
    "crisis_only": _override(
        crisis_oversample_prob=0.10,
    ),
    "crash_only": _override(
        crash_injection_prob=0.005,
    ),

    # === Compound — the original Phase 2 (overdosed) ===
    "phase2_full": _override(
        sell_cost_bps=30.0,
        buy_cost_bps=10.0,
        cvar_weight=0.50,
        max_weight_per_asset=0.30,
        min_cash=0.10,
        crisis_oversample_prob=0.30,
        crash_injection_prob=0.020,
    ),

    # === Plan A — Phase 2 with all strengths halved ===
    "phase2_relaxed": _override(
        sell_cost_bps=30.0,
        buy_cost_bps=10.0,
        cvar_weight=0.25,
        max_weight_per_asset=0.40,
        min_cash=0.10,
        crisis_oversample_prob=0.10,
        crash_injection_prob=0.005,
    ),

    # === Phase 3 — minimal: only the ablation winners (crash + constraints) ===
    # Hypothesis: crash_only (OOS Sharpe +0.53) and constraints_only (OOS +0.16)
    # were the only useful mechanisms. Compose just these two — skip the harmful
    # ones (cvar, crisis, asym_cost). Test if they have negative interaction
    # like the rest of Phase 2.
    "phase3_minimal": _override(
        crash_injection_prob=0.005,
        max_weight_per_asset=0.40,
        min_cash=0.10,
    ),
}


def get(name: str) -> dict:
    if name not in CONFIGS:
        raise KeyError(f"Unknown config: {name}. Available: {list(CONFIGS)}")
    return dict(CONFIGS[name])


def list_configs():
    return list(CONFIGS.keys())
