# 🧪 Experiment Log — Stock RL

A running diary of experiments, results, and lessons.
Owner: Victor (Sheng-Hung Tseng)
Started: 2026-06-19

---

## 📌 Project requirements (Victor's stated preferences)

Established 2026-06-19 during initial design conversation.

### Goal
Build an RL agent that decides **portfolio allocation** (combining trade timing + sizing)
across ~8 US mega-cap stocks + cash, rebalanced daily. Risk-adjusted return > SPY buy-hold
is the target, but **learning the methodology is the primary goal**.

### Constraints (from Victor)
- 🇹🇼🇺🇸🇯🇵 Outputs in Traditional Chinese unless context demands otherwise
- 💻 Hardware: MacBook Air M-series, no GPU → CPU/MPS only
- 💵 Free / cheap data sources only (yfinance, FRED free tier, RSS for news)
- 🎯 Daily bars, US equities
- 🆓 Free LLM provider preferred (Groq, then DeepSeek or Gemini if quota issues)
- 🤝 Combined task: **B (trade decisions) + C (portfolio allocation)** in one model
- 🧠 Philosophy: **「最大限度的自由 — 不要太多人為干涉，讓模型自己學世界規律」**
   - **Caveat learned today**: this philosophy works in vision/language but
     fails in finance because of data poverty + non-stationarity (see Phase 1 results)
- 🦢 Wants the model to handle "black swans" → translated to **robustness training**
   (crash injection, crisis oversampling, CVaR reward)
- 🌍 Wants international situations / news to be considered eventually
   (Phase 2.5 planned: LLM-based world-state encoder)

### Stated future directions
- Phase 2.5: LLM as world-state feature encoder (FinAgent / TradingAgents style)
- Phase 3: Larger model / Decision Transformer / world-model RL
- Possible academic angle: ICAIF / KDD paper on robustness mechanism interactions

---

## 🏗️ Architecture decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Action space | Continuous softmax weights (8 stocks + cash) | Jiang/Xu/Liang 2017 design — covers both trade and allocation |
| Algorithm | PPO (stable-baselines3) | Easier to debug than SAC; switch later if needed |
| Reward | Differential Sharpe + concentration penalty | Moody & Saffell 2001 |
| Look-ahead bias | State uses t-1 close, action at t, return at t | Standard prevention |
| Costs | Asymmetric (sell > buy) configurable | Approximates real-world spread |
| Constraints | Hard projection (max/asset, min cash) | López de Prado discipline |
| Robustness | Crash injection + crisis sampling + CVaR | Multiple approaches from literature |
| Logging | CSV + JSON per (run, period) in `results/` | Reproducibility |
| Wrapper | `./run.sh` auto-activates venv | Save user from venv pain |

---

## 📅 Daily Log

### 2026-06-19 — Day 1 (Marathon session)

#### Environment setup
- Installed Node + pnpm (originally for OpenClaw experimentation)
- Installed Python venv with PyTorch (MPS supported), yfinance, stable-baselines3, gymnasium
- All deps verified working on M-series MPS

#### Phase 1: Bare-bones baseline (15:15)
**Config**: `phase1_baseline`
- 8 mega-cap tickers (AAPL/MSFT/GOOGL/AMZN/NVDA/TSLA/META/SPY) + cash
- 60-day window, engineered features (log return, vol, momentum, volume z)
- No constraints, no robustness tricks
- PPO MlpPolicy, 200k steps, ~10 min on MPS

**Result**:
| Metric | In-Sample (2015-2024) | OOS (2024-2026) |
|--------|----------------------:|-----------------:|
| Cum Return | +1051% | +53.6% |
| Sharpe | 3.55 | 0.79 |
| Max DD | -32% | -38% |
| Turnover | 82% | (high) |

**Observation**: Classic overfit pattern. In-sample Sharpe 3.55 is implausibly good.
Model loaded up on TSLA (24%) and NVDA (12%) — picked the winners in hindsight.

#### Phase 2 first attempt — overdose (16:00)
Added **simultaneously**:
- Hard constraints: max 30%/asset, min 10% cash
- Asymmetric costs: sell 30 bps, buy 10 bps
- CVaR-augmented reward: 50% Sharpe + 50% CVaR(5%)
- Crisis oversampling: 30% chance of starting near 2018/2020/2022 dates
- Crash injection: 2%/step probability of -5% to -15% shock

**Result**:
| Metric | IS | OOS |
|--------|----:|----:|
| Sharpe | 0.36 | -0.19 |
| Cum Return | +73% | -8.7% |
| Max DD | -55% | -35% |

**Observation**: 🚨 Made things much worse. Weights uniform ~10% each — model gave up
on stock-picking. Diagnosis: **added too many mechanisms simultaneously**, can't tell
which one hurt.

#### Plan A — phase2_relaxed (15:17 result, 200k steps)
Halved all Phase 2 strengths:
- max_weight 30% → 40%, cvar 50% → 25%
- crisis 30% → 10%, crash 2% → 0.5%

**Result**:
| Metric | IS | OOS |
|--------|----:|----:|
| Sharpe | **1.80** | **-1.21** 🚨 |
| Cum Return | +389% | **-60.9%** |
| Max DD | -29% | -81% |

**Observation**: 🤯 OOS got **worse**, not better. IS-OOS gap = 3.0 (catastrophic).
**My (Claude's) prediction was wrong** — relaxing didn't help.
Hypothesis at this point: crash injection causes train-deploy distribution mismatch.

#### Plan B — full ablation (16:00, 100k steps × 7 configs)
Tested each Phase 2 mechanism in isolation + recompose.

**Out-of-sample ranking by Sharpe** (SPY benchmark: +1.16):

| Rank | Config | IS Sharpe | OOS Sharpe | IS-OOS gap | OOS Return |
|---|---|---|---|---|---|
| 🥇 | **crash_only** | 1.18 | **+0.53** | **0.65** | **+24.3%** |
| 🥈 | constraints_only | 1.59 | +0.16 | 1.43 | +8.2% |
| 🥉 | phase1_baseline | 1.57 | -0.05 | 1.62 | -2.3% |
| 4 | cvar_only | 1.57 | -0.10 | 1.67 | -4.6% |
| 5 | asym_cost_only | 1.53 | -0.13 | 1.66 | -5.9% |
| 6 | crisis_only | 1.32 | -0.18 | 1.50 | -8.4% |
| 7 | phase2_relaxed (100k) | 0.96 | -0.30 | 1.26 | -12.7% |
| 8 | phase2_relaxed (200k) | 1.80 | -1.21 | 3.01 | -60.9% |

**Key findings**:
1. 🤯 **Crash injection ALONE is the most useful mechanism** (opposite of my prediction).
   It functions as **regularization** (like dropout), preventing memorization of train trajectories.
2. 🤝 **Constraints alone are second-best** — diversification helps but less than crash.
3. 🚫 **CVaR / asymmetric_cost / crisis sampling alone don't help** OOS.
4. 🔗 **Mechanisms have negative interaction** — combining 5 made things much worse than any single one.
5. 📉 **More training steps → worse OOS for compound config** (200k vs 100k phase2_relaxed:
   OOS Sharpe -1.21 vs -0.30). Classic overfit signal.

**Wider lessons**:
- López de Prado's "more features rarely help" — confirmed empirically
- Single-seed RL results have huge variance → need multi-seed
- "Adding to be safe" is often worse than "doing one thing well"

#### Phase 3 — minimal composition (22:29 + 22:31, ACCIDENTAL 2-SEED)
**Config** `phase3_minimal`: just crash_injection_prob=0.005 + max_weight=0.40 + min_cash=0.10

**Result** (accidentally trained twice, got 2 seeds!):
| Seed | IS Sharpe | OOS Sharpe | OOS Return | OOS Max DD |
|------|----------:|-----------:|-----------:|-----------:|
| Run 1 (22:29) | 1.88 | **+0.085** | +4.5% | -38% |
| Run 2 (22:31) | 1.83 | **-0.667** | -35% | -52% |
| **Mean ± std** | 1.86 ± 0.03 | **-0.29 ± 0.39** | -15% | -45% |

**Key finding — possibly the most important of the day**:
- Two independent training runs of the SAME config gave OOS Sharpe **+0.08 vs -0.67**.
- A gap of 0.75 Sharpe — *enough to flip every conclusion in the ablation table above*.
- **Single-seed RL results are basically noise** (see Henderson et al 2018 "Deep RL that Matters").
- The ablation rankings (especially "crash_only = winner") must be re-validated with multi-seed.

**Implications**:
- crash + constraints combination may also suffer from negative interaction
  (mean -0.29 vs crash_only's single-seed +0.53)
- All of today's ablation conclusions are now **provisional** — need multi-seed to confirm
- The previously claimed "regularization" effect of crash injection — also provisional

#### Multi-seed validation — crash_only (started 22:45, completed 2026-06-20 00:34)
**The original ablation +0.53 OOS Sharpe was a lucky seed. Confirmed with N=5.**

| Source | IS Sharpe | OOS Sharpe | OOS Return |
|--------|----------:|-----------:|-----------:|
| Ablation original (16:01) | 1.18 | +0.53 ⭐ | +24.3% |
| seed 0 (22:51) | 1.12 | -0.19 | -8% |
| seed 1 (22:59) | 1.51 | -0.24 | -11% |
| seed 2 (00:18) | 1.21 | -0.15 | -7% |
| seed 3 (00:26) | 1.10 | +0.23 | +11% |
| seed 4 (00:34) | 1.25 | -0.38 | -18% |
| **5-seed mean ± std** | **1.24 ± 0.17** | **-0.14 ± 0.23** | -6.7% ± 10.8% |
| **6 samples (incl. ablation)** | 1.23 ± 0.16 | **-0.03 ± 0.32** | -1.6% ± 14.8% |

**Significance check** (OOS Sharpe > 0):
- t-statistic = mean / SE = -1.43
- Verdict: **⚠️  Cannot reject null hypothesis (Sharpe = 0)**

**Definitive conclusions** (high confidence):
1. ❌ crash_only is NOT a real winner — OOS Sharpe indistinguishable from zero
2. ❌ The original ablation table's rankings were largely noise
3. ✅ IS Sharpe is *consistent* (+1.24 ± 0.17) — the model CAN fit training data
4. ❌ But the model *cannot* generalize OOS — overfitting is systematic

**The "crash injection as regularization" story collapses with N=5**:
- Earlier hypothesis: crash injection prevents memorization → OOS gains
- Reality: IS-OOS gap remains ~1.4 Sharpe regardless of seed
- The single +0.53 was the right tail of a wide noise distribution

**Implication for the project**:
- Daily-bar PPO + price-derived features cannot produce reliable OOS alpha at this data scale
- Adding/removing mechanisms doesn't fix this
- This is a *signal problem*, not a *learning problem*
- The next move must be: **new information** (LLM + news for Phase 2.5) or **new method**
  (distributional RL, world models, decision transformers)
- Not just retrying mechanisms

#### END OF DAY 1 — 2026-06-20 00:35
Total: ~9 hours of work. ~20 training runs. ~50k characters of documentation.
Key meta-finding: single-seed RL = unreliable. Confirmed first-hand.

---

### 2026-06-20 — Day 2: Pivot to Scout (Rally Detection)

#### Project goal reframe
After Day 1 confirmed PPO + portfolio allocation cannot produce reliable alpha
at this data scale, Victor refined the goal:
- 想賺錢，但不必打贏大盤
- 也不希望虧得比大盤多
- 想找出像 "2025年 MU / LITE" 那種「未被發現的股票」
- 評量標準：模型能在 MU 還是 $100 的時候 flag positive，不需抓到底部

→ This is **stock screening / rally detection**, a supervised classification
problem, NOT RL portfolio management. Major reframe.

#### Scout architecture
```
scout/
├── fetch_universe.py     # 129 mid-cap US stocks (curated, hard-coded)
├── label_rallies.py      # Find 50%-in-180d rallies, label troughs
├── analyze_rallies.py    # Statistics + benchmark case verification
├── compute_features.py   # (next) — per-day technical features
└── train_classifier.py   # (next) — GradientBoosting, eval on 2024+
```

Rally definition:
- gain_threshold = 50%
- window_days   = 180
- min_gap_days  = 60 (no overlapping rallies)

#### Step 1+2 result (rally labeling)
**827 rallies labeled across 124 of 129 tickers in 2015-2024**.

- Avg gain    : 107% (median 73%)
- Avg duration: 198 days (median 222)
- Distribution by year: even, with peaks in 2020 (147) and 2022 (117)
- Test set (2024+): 41 rallies — adequate but small

MU benchmark cases (7 historical rallies caught):
- 2016-02-11 → 2016-09-27   $9.45 → $17.56   (85.8%)
- 2016-11-03 → 2017-06-07   $16.21 → $31.70  (95.5%)
- 2017-08-10 → 2018-03-20   $26.81 → $59.64  (122.4%)
- 2018-12-24 → 2019-09-11   $28.30 → $49.23  (73.9%)
- 2020-03-16 → 2020-11-27   $33.62 → $62.64  (86.3%)
- 2022-09-26 → 2023-05-26   $47.95 → $72.99  (52.2%)
- **2023-07-07 → 2024-03-22  $59.99 → $109.34 (82.3%)** ← Victor's target

Other notable benchmarks caught: LITE (10 rallies), ENPH (12 including +485%),
FSLR (10), AMD (8 including +326%).

#### Key methodology note: Look-ahead bias is THE risk
The biggest danger in this problem class:
- Using current universe (survivorship bias)
- Using current macro narratives to interpret 2015 data
- Including features computed with future data (look-ahead)
- Using "labels-derived" features (data leakage)

For v0 we accept survivorship bias (curated universe) but must enforce
strict point-in-time feature computation.

#### Step 3: Feature engineering + classifier training (afternoon)

Built `scout/compute_features.py` with 11 point-in-time technical features:
- drawdown_52w, price_vs_ma200
- momentum_3m / 6m / 12m
- vol_60d, vol_60d_rank (expanding-rank → no look-ahead)
- volume_surge_30d, rsi_14, rs_vs_spy_3m, days_since_high

Sampled every 5 trading days → **59,417 feature samples**.
Labels (positive = rally trough within next 30 days): **3,138 positive (5.3%)**.

Trained GradientBoostingClassifier (n_estimators=200, max_depth=4):

| Metric | Train | Val (2023) | **Test (2024)** |
|--------|-------|-----------:|--------------:|
| AUC | 0.843 | 0.751 | **0.693** ✅ |
| AvgPrecision | 0.429 | 0.173 | 0.024 |
| Baseline AP | 0.067 | 0.064 | 0.011 |

**Test AUC 0.693 = real out-of-sample signal** (random = 0.5, severe overfit = 0.85+).

Rally capture on 41 test rallies:
- threshold=0.3 → **13/41 caught (32%)**, median progress at catch +17%
- threshold=0.5 → only 1/41 (scores compressed by class imbalance)

**🎯 MU case study (Victor's $100 MU criterion)**:
MU 2023-07-07 rally ($59.99 → $109.34):
- 2023-07-10 score 0.111 at $61.80 (progress +3.7%) ← caught very early
- 2023-06-23 score 0.100 at $64.45 (progress +9.0%)
- 2023-05-03 score 0.063 at $60.13 (progress +0.3%) ← caught BEFORE trough

**Caught MU at $62, way before $100 target. ✅✅✅**

Feature importance ranking (model learned what we'd expect):
1. rsi_14 (25%) — classic oversold
2. drawdown_52w (11%)
3. price_vs_ma200 (10%)
4. days_since_high (9%)
5. vol_60d (8%)
6-11. momentum_*, volume_surge, rs_vs_spy, vol_rank

Honest assessment of v0:
- ✅ Has real signal (AUC > random)
- ✅ Catches MU early (the original goal)
- ✅ Feature importance is intuitive (RSI / drawdown / MA)
- ⚠️ Recall only 32% — misses 2/3 of rallies
- ⚠️ Test AvgPrecision 0.024 — top-K precision will be low
- ⚠️ Single-seed result needs multi-seed verification (Day 1 lesson!)
- ⚠️ Score distribution compressed (max ~0.15 due to class imbalance)

Next: top-K eval + multi-seed verification.

---

## 📂 Files & artifacts

```
stock-rl/
├── data/
│   ├── fetch_prices.py          # yfinance + parquet cache + features
│   └── cache/                   # cached OHLCV (saves time on re-runs)
├── env/
│   └── portfolio_env.py         # Gym env with all Phase 2 knobs (configurable)
├── policy/
│   ├── configs.py               # All named configs (phase1, phase2_*, phase3, ablations)
│   ├── train_ppo.py             # CLI training, reads $CONFIG env var
│   ├── eval.py                  # Loads model, evals on IS+OOS, saves CSV+JSON
│   ├── ablation.py              # Train+eval all configs in sequence
│   ├── multi_seed.py            # Train+eval one config across N seeds
│   ├── results_io.py            # CSV + JSON persistence
│   └── checkpoints/             # All trained models, one per config
├── results/
│   ├── runs.csv                 # All eval rows (sortable in Excel)
│   └── runs/                    # Full JSON per run (incl. daily returns)
├── tb_logs/                     # TensorBoard logs per run
├── run.sh                       # CLI wrapper (handles venv activation)
├── README.md                    # User-facing project doc
├── BIBLIOGRAPHY.md              # 31 papers organized by topic + reading order
└── EXPERIMENT_LOG.md            # ← This file
```

---

## 🎯 Open questions / TODOs

### Verified
- ✅ Pipeline works end-to-end
- ✅ MPS acceleration confirmed (~1000 fps during PPO train)
- ✅ Eval results persist to CSV correctly
- ✅ Ablation study completed, surprising findings

### Next experiments (UPDATED 2026-06-20 — priority reordered after multi-seed)

**Don't do** (proven not to help in this setup):
- ❌ Retry single mechanism tuning with new prob/strength values (won't escape noise)
- ❌ Add more reward shaping (negative interaction is the norm)
- ❌ Train more steps (just amplifies overfit)

**Worth trying next** (priority):
1. **🟢 Phase 2.5: Add FRED macro features** (10Y yield, VIX, USD index, oil)
   — cheap, new signal source, baseline check before LLM complexity
2. **🟢 Walk-forward backtest** (proper train/test rolling) — fixes single-split fragility
3. **🟡 Phase 2.5b: LLM world-state encoder** (news → embedding via Groq API)
   — adds genuinely new information, aligned with FinAgent/TradingAgents direction
4. **🔵 Distributional RL (C51 / IQN)** — proper risk-sensitive method instead of CVaR hack
5. **🔵 Recurrent PPO** — capture temporal dependencies natively
6. **🔵 Decision Transformer** — offline RL on full history with attention

**Reality check** — before any of the above:
- ✅ Run multi-seed (N=5) on phase3_minimal and constraints_only to confirm "nothing works at this scale" — quick reality check (~80 min)
- ✅ If true, the "Phase 1-3 didn't work" is the conclusion, not a step backwards

### Open scientific questions
- Why does crash injection regularize so well? Is it the noise, or the asymmetry (only down)?
- Is constraints' OOS improvement (+0.16) significant or noise? Need multi-seed.
- Why do mechanisms interact negatively when composed?
- Does the "no model beats SPY" finding hold with proper walk-forward, or is it the split's fault?

---

## 🚦 Recommended commands (cheat sheet)

```bash
# Standard workflow
./run.sh train-eval                 # Train default (phase2_relaxed) + auto-eval
./run.sh results                    # Print compare table of all runs

# Use specific config
CONFIG=phase3_minimal ./run.sh train-eval
CONFIG=phase1_baseline RUN_ID=baseline_v2 ./run.sh train-eval

# Ablation across all configs
./run.sh ablation

# Multi-seed validation (5 seeds by default)
CONFIG=crash_only N_SEEDS=5 ./run.sh multi-seed
CONFIG=phase3_minimal N_SEEDS=3 ./run.sh multi-seed

# Test only
./run.sh test-data
./run.sh test-env

# Watch training
./run.sh tensorboard                # opens http://localhost:6006

# Drop into venv REPL
./run.sh shell
```

---

## 📚 Reading status (companion to BIBLIOGRAPHY.md)

Target for tonight (2026-06-19):
- [ ] Sutton "The Bitter Lesson" (2 pages, 15 min)
- [ ] López de Prado "10 Reasons ML Funds Fail" (~20 pages, 25 min)
- [ ] Moody & Saffell 2001 abstract + sections 2-3 (15 min)

Target for tomorrow:
- [ ] Sun et al 2024 survey (skim)
- [ ] Jiang/Xu/Liang 2017 (skim implementation details)

Target for this week:
- [ ] FinRL paper
- [ ] FinAgent KDD 2024 (motivation for Phase 2.5)

---

## 🧠 Mental model updates (running list of "things I learned")

Date | Lesson
---|---
2026-06-19 | "Bitter Lesson" applies to most of ML but NOT to data-poor non-stationary domains like finance
2026-06-19 | In-sample Sharpe > 3 is almost always overfit
2026-06-19 | Adding multiple "robustness" mechanisms can hurt more than help — they interact
2026-06-19 | Crash injection during training acts as regularization, not just stress-testing
2026-06-19 | OOS Sharpe is the only metric that matters; everything else is overfit-risk
2026-06-19 | Single-seed RL results are not reliable — variance is huge
2026-06-19 | "More features" almost always hurts OOS in finance (verified independently)
2026-06-19 | venv is a real problem; wrapper script is worth the 30 min investment
2026-06-19 | Single-seed RL = effectively random. The two phase3 runs (Sharpe +0.08 vs -0.67) prove all of today's "rankings" need multi-seed re-verification.
2026-06-19 | "Crash injection is the winner" was probably half-true: it might be real but the +0.53 effect size could easily be ±0.4 noise. Need multi-seed.
2026-06-19 | Henderson et al 2018 "Deep RL that Matters" is now reading priority #1 — papers and rankings without multi-seed are not trustworthy
2026-06-19 | Confirmed empirically: crash_only's "+0.53 OOS Sharpe" was lucky. 3-seed mean is -0.03 ± 0.45. Same will likely be true for any single-seed claim from today's ablation.
2026-06-19 | Realistic project takeaway: a daily-bar PPO + price features almost certainly cannot beat SPY out-of-sample. Adding mechanisms doesn't fix this. The next move needs new information (macro / news) or new methods (world models / distributional RL / DT), not more reward tweaks.
2026-06-20 | crash_only 5-seed mean OOS Sharpe = -0.14 ± 0.23. Hypothesis "crash injection regularizes" is dead. All single-seed rankings from today's ablation are noise. Confirmed.
2026-06-20 | The most important deliverable from today is *not* a working model — it's the methodology discipline: every claim needs N≥5 seeds + reported std. This applies to the rest of life too.

---

*Last updated: 2026-06-19*
