# 📚 Bibliography — Stock RL Project

Papers and resources referenced in the design of this project, organized by topic.
Reading priority marked: ⭐⭐⭐⭐⭐ (must-read) → ⭐ (optional).

---

## 🏛️ Foundational RL-for-Trading

### ⭐⭐⭐⭐⭐ Moody & Saffell (2001)
**"Learning to Trade via Direct Reinforcement"**
*IEEE Transactions on Neural Networks, 12(4), 875-889*

The grandparent paper. First to propose:
- **Differential Sharpe Ratio** as RL reward (you use this in `env.py`)
- Recurrent RL for trading
- Explicit handling of transaction costs

- 🔗 https://papers.neurips.cc/paper/1551-reinforcement-learning-for-trading.pdf
- Cited by your env's reward design.

### ⭐⭐⭐⭐ Deng et al. (2017)
**"Deep Direct Reinforcement Learning for Financial Signal Representation and Trading"**
*IEEE TNNLS, 28(3), 653-664*

Moody's idea + Deep Learning. Introduces **task-aware BPTT** for gradient vanishing.

- 🔗 https://ieeexplore.ieee.org/document/7407387

### ⭐⭐⭐⭐⭐ Jiang, Xu, Liang (2017)
**"A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem"**
*arXiv:1706.10059*

First to use:
- **Portfolio weights as softmax-output action** (your env uses this)
- CNN policy over price history
- Crypto market experiments

- 🔗 https://arxiv.org/abs/1706.10059
- Cited by your action-space design.

---

## 🛠️ Open-Source Frameworks

### ⭐⭐⭐⭐⭐ FinRL — Liu et al. (2020)
**"FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance"**
*NeurIPS 2020 Deep RL Workshop*

The canonical library. Compare your structure to theirs.

- 🔗 Paper: https://arxiv.org/abs/2011.09607
- 🔗 Code:  https://github.com/AI4Finance-Foundation/FinRL

### ⭐⭐⭐⭐ TradeMaster (Sun et al. 2024)
A more recent / fuller framework on top of FinRL.

- 🔗 https://github.com/TradeMaster-NTU/TradeMaster

---

## 🧠 RL Algorithm Papers (you use these implicitly via SB3)

### ⭐⭐⭐⭐ Schulman et al. (2017)
**"Proximal Policy Optimization Algorithms"**
*arXiv:1707.06347*

The PPO paper. You use stable-baselines3's implementation.

- 🔗 https://arxiv.org/abs/1707.06347

### ⭐⭐⭐ Haarnoja et al. (2018)
**"Soft Actor-Critic"**
*arXiv:1801.01290*

The off-policy alternative we'd upgrade to in Phase 3.

- 🔗 https://arxiv.org/abs/1801.01290

---

## 📉 Risk-Sensitive & Distributional RL

### ⭐⭐⭐⭐⭐ Bellemare, Dabney, Munos (2017)
**"A Distributional Perspective on Reinforcement Learning"** (C51)
*ICML 2017*

Foundation for "learn the whole return distribution, not just expectation." The motivation for our CVaR-augmented reward.

- 🔗 https://arxiv.org/abs/1707.06887

### ⭐⭐⭐⭐ Dabney et al. (2017)
**"Distributional Reinforcement Learning with Quantile Regression"** (QR-DQN)

- 🔗 https://arxiv.org/abs/1710.10044

### ⭐⭐⭐⭐ Dabney et al. (2018)
**"Implicit Quantile Networks"** (IQN)

- 🔗 https://arxiv.org/abs/1806.06923

### ⭐⭐⭐ Tamar, Glassner, Mannor (2015)
**"Optimizing the CVaR via Sampling"**

CVaR formalization that motivates the `cvar_weight` term in our reward.

- 🔗 https://arxiv.org/abs/1404.3862

### ⭐⭐⭐⭐ Risk-Averse Distributional RL (2020)
**"Improving Robustness via Risk Averse Distributional Reinforcement Learning"**

- 🔗 https://arxiv.org/abs/2005.00585
- 🔗 Code: https://github.com/Silvicek/cvar-algorithms

### ⭐⭐⭐ Natural Gas Futures CVaR (2025)
**"Risk-averse policies for natural gas futures trading using distributional reinforcement learning"**

Empirical demonstration: C51 with CVaR objective beats classical RL by 32%.

- 🔗 https://arxiv.org/abs/2501.04421

---

## 🌍 World Models & Model-Based RL

### ⭐⭐⭐⭐⭐ Sutton (2019)
**"The Bitter Lesson"** (essay, not paper)

Two pages. Required reading for any ML practitioner. Argues that scale + compute beats hand-crafted priors.

- 🔗 http://www.incompleteideas.net/IncIdeas/BitterLesson.html

### ⭐⭐⭐⭐⭐ Ha & Schmidhuber (2018)
**"World Models"**
*NeurIPS 2018*

The original. Learn a dynamics model, then plan in imagination.

- 🔗 https://arxiv.org/abs/1803.10122
- 🔗 https://worldmodels.github.io/ (interactive)

### ⭐⭐⭐⭐⭐ Hafner et al. (2020, 2021, 2023)
**Dreamer V1 / V2 / V3**

Current SOTA for model-based RL. Dreamer V3 (2023) uses fixed hyperparams across all benchmarks.

- 🔗 V1: https://arxiv.org/abs/1912.01603
- 🔗 V3: https://arxiv.org/abs/2301.04104

### ⭐⭐⭐ Racanière et al. (2017)
**"Imagination-Augmented Agents for Deep Reinforcement Learning"** (I2A)

The "use imagined rollouts as extra training data" pattern.

- 🔗 https://arxiv.org/abs/1707.06203

### ⭐⭐⭐⭐ Tobin et al. (2017)
**"Domain Randomization for Transferring Deep Neural Networks"**

The motivation for our `crash_injection_prob`. Train on randomized variations of the env → robust to real-world differences.

- 🔗 https://arxiv.org/abs/1703.06907

---

## 🎨 Generative Models for Finance (Phase 3 directions)

### ⭐⭐⭐⭐ Yoon, Jarrett, van der Schaar (2019)
**"Time-series Generative Adversarial Networks"** (TimeGAN)
*NeurIPS 2019*

First general-purpose time-series GAN. Used in many finance papers.

- 🔗 https://papers.nips.cc/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html

### ⭐⭐⭐⭐ Wiese et al. (2020)
**"Quant GANs: Deep Generation of Financial Time Series"**

GAN specifically tuned for finance — captures fat tails, volatility clustering.

- 🔗 https://arxiv.org/abs/1907.06673

### ⭐⭐⭐⭐⭐ DARL (2025)
**"Diffusion-Augmented Reinforcement Learning for Robust Portfolio Optimization under Stress Scenarios"**

Hot off the press: use DDPM to generate stress scenarios of varying intensity.

- 🔗 https://arxiv.org/abs/2510.07099

---

## 🤖 LLM-Based Trading Agents (Phase 2.5 direction)

### ⭐⭐⭐⭐⭐ FinAgent (Zhang et al., KDD 2024)
**"A Multimodal Foundation Agent for Financial Trading"**

Multi-modal fusion: text + numerical + chart. Closest match to your "world knowledge → RL" idea.

- 🔗 https://personal.ntu.edu.sg/boan/papers/KDD24_FinAgent.pdf

### ⭐⭐⭐⭐ TradingAgents (Xiao et al. 2024)
**"TradingAgents: Multi-Agents LLM Financial Trading Framework"**

LLM agents play roles (fundamental analyst, sentiment analyst, etc.). Inspires "multi-role ensemble" prompting.

- 🔗 https://arxiv.org/abs/2412.20138

### ⭐⭐⭐⭐ FinGPT (Liu et al. 2023)
Open-source FinLLM. Could be fine-tuned with LoRA for cheap.

- 🔗 https://arxiv.org/abs/2306.06031

### ⭐⭐⭐ BloombergGPT (Bloomberg 2023)
**"BloombergGPT: A Large Language Model for Finance"**

50B param model trained on Bloomberg's corpus. Industrial benchmark.

- 🔗 https://arxiv.org/abs/2303.17564

### ⭐⭐⭐⭐ News-Aware Direct RL (2025)
**"News-Aware Direct Reinforcement Trading for Financial Markets"**

Skip the LLM agent layer — feed news embeddings directly to RL.

- 🔗 https://arxiv.org/abs/2510.19173

### ⭐⭐⭐ Pretrained LLM + LoRA as Decision Transformer (2024)
- 🔗 https://arxiv.org/abs/2411.17900

---

## 📊 Survey Papers (read these first if you want breadth)

### ⭐⭐⭐⭐⭐ Sun et al. (2024) — Best entry survey
**"A taxonomy of literature reviews and experimental study of deep reinforcement learning in portfolio management"**
*Artificial Intelligence Review*

Compares A2C / DDPG / PPO / SAC / TD3 on the same portfolio task. Empirical, opinionated.

- 🔗 https://link.springer.com/article/10.1007/s10462-024-11066-w

### ⭐⭐⭐⭐ Mosavi et al. (2020)
**"Comprehensive Review of Deep Reinforcement Learning Methods and Applications in Economics"**

- 🔗 https://arxiv.org/abs/2004.01509

---

## 🦢 Black Swan / Tail Risk Philosophy

### ⭐⭐⭐⭐⭐ Taleb (2007) — "The Black Swan"
The book. You don't need to read all of it; chapters 1-4 are the core argument.

### ⭐⭐⭐⭐ Taleb (2012) — "Antifragile"
Even more important than Black Swan for your trading project. Don't just survive volatility — profit from it.

### ⭐⭐⭐⭐ Mark Spitznagel — Universa Investments
Spitznagel runs Universa using Taleb's principles. **+3,612% return in March 2020 COVID crash** while the world fell apart.

- 🔗 Strategy explainer: https://www.marketsentiment.co/p/tail-risk-hedging

### ⭐⭐⭐⭐⭐ López de Prado (2018)
**"The 10 Reasons Most Machine Learning Funds Fail"**
*Journal of Portfolio Management*

Reality check for everyone training ML on finance. He's both academic and practitioner (cofounded HFT firm).

- 🔗 https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3104816

### ⭐⭐⭐⭐⭐ López de Prado (2018) — book
**"Advances in Financial Machine Learning"**

Bible of careful ML in finance. Chapters on:
- Sample uniqueness (your training data isn't iid)
- Sequential bootstrap
- Cross-validation pitfalls in finance (purged k-fold)
- Backtest overfitting

---

## 🏭 Industry Practice (mostly not papers — articles & books)

### Risk Management Infrastructure
- Position sizing rules (Kelly criterion)
- Hard drawdown limits
- Real-time pre-trade risk checks
- Multi-strategy diversification

### Renaissance Technologies / Two Sigma / DE Shaw
- Renaissance: unified model across asset classes
- Two Sigma: alternative data (satellite, credit card)
- DE Shaw: long-form computational finance research

### Tail Risk Hedging
- Spitznagel's Universa: OTM puts as insurance
- 0.5-1% drag per year, 1000%+ in crashes
- Asymmetric payoff (Taleb's "barbell strategy")

---

## 🛠️ Statistical Methods Used in This Project

### Block Bootstrap (Politis & Romano, 1994)
**"The Stationary Bootstrap"** *Journal of the American Statistical Association*

For resampling time series in a way that preserves serial dependence. Use this in Phase 2.5 for synthetic training trajectories.

### Differential Sharpe Ratio
Original: Moody & Saffell (above). The formula:
```
diff_sharpe_t = (R_t - μ_t) / σ_t
```
where μ and σ are EMAs.

---

## 🎓 Suggested Reading Order

**Week 1 — Foundation**
1. Sutton "Bitter Lesson" (15 min)
2. Moody & Saffell 2001 (must read end-to-end)
3. Jiang Xu Liang 2017

**Week 2 — Phase 2 grounding**
4. Bellemare 2017 C51 (skim, understand idea)
5. Sun et al 2024 survey
6. López de Prado "10 Reasons" article

**Week 3 — Phase 2.5 LLM direction**
7. FinAgent (KDD 2024)
8. TradingAgents 2024
9. News-Aware Direct RL 2025

**Week 4 — Phase 3 robustness**
10. Ha & Schmidhuber "World Models"
11. Hafner Dreamer V3
12. DARL 2025

**Background (any time)**
- Taleb "Black Swan" / "Antifragile"
- López de Prado "Advances in Financial Machine Learning" (book)

---

## 🚦 Conferences to Watch

| Conference | Relevant tracks |
|------------|-----------------|
| **NeurIPS** | All RL, world models, generative |
| **ICML** | Same |
| **ICLR** | RL theory, distributional RL |
| **AAAI** | Applied RL, multi-agent |
| **ICAIF** ⭐ | ACM International Conf on AI in Finance — your direct target |
| **KDD** | FinAgent etc. — data mining + finance |

If you publish, **ICAIF** (Nov annually) is the natural venue.
