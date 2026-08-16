# Exploration Logbook

**Project**: Phase Transitions in Wealth Inequality: an Active Learning Approach on Kesten Stochastic Dynamics  
**Stage**: Preliminary (v1.1.0) | **Last Updated**: 2026-08-16

---

## 📍 Current Exploration Status

| Dimension | Status |
| :--- | :--- |
| Core Simulation Engine | ✅ N=100,000, T=500, Numba JIT + NumPy fallback, stable |
| Active Learning Agent | ✅ 30 rounds UCB+GP exploration completed, random baseline control built |
| Phase Transition Diagram (D1) | ✅ 1D controlled scan complete (fixed c=0.10, reflect, lognormal) |
| Left-Tail Model Selection | ✅ AIC/BIC comparison across 3 candidate distributions complete |
| Empirical Calibration | ✅ World Bank API integration, US-China 1980–2023 alignment complete |
| Counterfactual Policy Simulation | ✅ Poverty subsidy S>0 & tax τ>0 counterfactuals complete |
| Non-linear Behavioral Modes | ✅ Keynesian MPC + Prospect Theory (standalone module) |
| Complex Network Topologies | ✅ Scale-Free + Small-World networks (standalone module) |
| Statistical Testing | ⚠️ Welch p ≈ 0.28, AI vs Random difference not yet strongly significant |

---

## 🔍 Major Discoveries

### F1 ｜ Absorbing Boundary Triggers Structural Collapse (Unexpected)
Switching boundary conditions to `absorb` (wealth collapses to zero upon reaching bottom) causes power-law goodness-of-fit $R^2$ to plunge from $>0.85$ to $<0.50$, destroying the Pareto distribution. Boundary conditions act as structural factors determining physical phases.

### F2 ｜ Policy Failure Boundary
In high shock risk regions ($p > 0.04$), applying progressive tax $\tau = 0.15$ fails to prevent poverty rate explosion, revealing a physical limit for redistribution policies.

### F3 ｜ US-China Parameter Structural Differences
World Bank data calibration reveals China's optimal parameters feature high $c$ (high dissipation/consumption/tax) and low $p$ (low shock risk), whereas the US exhibits the opposite—qualitatively matching distinct socioeconomic structures.

### F4 ｜ Candidate Phase Transition Threshold $R^* \approx 0.15 \sim 0.20$
Controlled scans place the candidate transition region at $R^* \approx 0.15 \sim 0.20$.

---

## 👥 Peer Feedback & Responses

### Written Peer Feedback

> *"The R* = p/c plot cannot prove a phase transition—each point alters multiple variables simultaneously, creating confounding."*

→ **Responded**: Implemented 1D controlled scan, fixing $c$, boundary, and income distribution, scanning solely along $p$ to eliminate confounding.

> *"Prove that power-law fits better than lognormal or other candidate distributions before discussing exponents."*

→ **Responded**: Added AIC/BIC + Likelihood Ratio Tests comparing power-law, lognormal, and exponential models.

> *"Cross-seed reproducibility logic in D1 is insufficient—comparing current parameters against means of different parameter combinations is not a controlled experiment."*

→ **Responded**: Updated to test relative jump significance ($\Delta \beta > 0.3$) across 10 independent verification seeds at identical parameter values.

### Conference Verbal Suggestions

> *"Provide qualitative behavioral modes from economic textbooks to the AI to observe emergent results."*

→ **Responded**: Built `behavioral_simulator.py` implementing Keynesian MPC $c(W) \sim W^{-\alpha}$ and Prospect Theory $p(W) \sim 1/W$.

> *"Tsinghua has an AI Agent Society framework (tsinghua-fib-lab/agentsociety)."*

→ **Recorded**: Added as reference [E6] for future multi-agent society architecture integration.

---

## 🔁 Exploration Iteration Record

> Data source: `outputs/kesten_exploration_log.csv`, 60 exploration rounds (30 random baseline + 30 active learning).

### Stage 1: Random Baseline (Rounds 1–30, RandomAgent)
Uniform random sampling without directionality ($\beta_{\text{left}}$ fluctuates 0.92–4.48).

### Stage 2: Controlled Active Learning (Rounds 31–60, AdaptiveAgent)
Monotonic candidate jump in $\beta_{\text{left}}$ around $R^* \approx 0.10 \sim 0.15$.

---

## 🛡️ Methodological & Scientific Rigor Revisions

1. **Phase Jump Terminology**: Designated as **Candidate Phase Transition**; added Finite-Size Scaling ($N \in [5k, 50k]$).
2. **Tail Estimation**: Introduced **MLE Hill Estimator**, **Bootstrap 95% CIs**, and cutoff sensitivity analysis.
3. **Model Selection & Calibration**: Explicitly labeled as **Custom Pseudo-LR Comparison** and **Exploratory Macro Calibration**.
