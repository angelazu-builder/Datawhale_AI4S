# Datawhale AI for Research Camp · Open Exploration Track
## Problem Definition Document (v2)

**Project Title**: Candidate Phase Transitions in Wealth Inequality: An Active Learning Framework for Kesten Stochastic Dynamics  
**Date**: 2026-08-16  

---

## 1. Why this problem is worthy of AI exploration

### 1.1 Scientific Background & Core Bottlenecks
Extreme inequality and Pareto power-law tails in social wealth distribution represent a classic interdisciplinary puzzle across statistical physics and macroeconomics (Econophysics). The Kesten stochastic process ($W_{t+1} = A_t \cdot W_t + B_t$) captures wealth dynamics under random multiplicative returns $A_t$ and additive income $B_t$.

However, traditional non-linear statistical physics models face severe **high-dimensional parameter grid-search bottlenecks**:
- Shock probability $p$, dissipation rate $c$, boundary conditions (`reflect` vs `absorb`), income distributions (`lognormal` vs `pareto`), progressive tax rate $\tau$, and minimum welfare subsidy $S$ constitute a massive continuous/discrete parameter space.
- Manual grid selection by researchers often conducts coarse linear scans, frequently missing **non-continuous phase transition thresholds** and critical mutation boundaries.

### 1.2 Necessity and Uniqueness of AI Intervention
- **Autonomous Hypothesis Search (Active Learning)**: Using a Gaussian Process (GP) Surrogate model with Upper Confidence Bound (UCB) acquisition functions, the AI agent autonomously queries regions with highest variance/gradient, locating critical thresholds $R^* = p/c$ with minimal sampling cost.
- **Mutation & Anomaly Signal Detection**: Deploying CUSUM statistical change-point detection, the AI monitors left-tail decay exponents $\beta_{\text{left}}$ in real-time, automatically discovering structural collapses (e.g., power-law loss under `absorb` boundaries) missed by coarse grids.

---

## 2. Problem Definition & Environment Design

### 2.1 State Space & Controls
We define the simulation environment as a stochastic dynamics system with $N = 100,000$ agents over time horizon $T = 500$.

- **Fixed Controls**:
  - Agent count $N = 100,000$
  - Time steps $T = 500$
  - Initial wealth distribution $W_0 \sim \text{Lognormal}(\mu=1.0, \sigma=0.5)$
- **Exploration Variables**:
  - Shock probability $p \in [0.001, 0.20]$
  - Dissipation rate $c \in [0.01, 0.50]$
  - Boundary conditions $\text{Boundary} \in \{\text{reflect}, \text{absorb}\}$
  - Policy levers: Progressive tax rate $\tau \in [0.0, 0.30]$, Welfare subsidy $S \in [0.0, 0.05]$

### 2.2 Candidate Phase Transition Definition
Under strict controlled variables (fixing $c=0.10$, `reflect`, `lognormal`), scanning along 1D parameter axis $p$. Define ratio $R^* = p/c$.
When the poverty left-tail exponent $\beta_{\text{left}}$ exhibits a non-continuous jump ($\Delta \beta > 0.3$ consistent across 10 independent verification seeds), the critical threshold is designated as a **Candidate Phase Transition**.

---

## 3. Core Scientific Discoveries

### F1 ｜ Absorbing Boundary Triggers Structural Collapse
Switching boundary conditions to `absorb` (wealth collapses to zero at lower bound) causes power-law goodness-of-fit $R^2$ to plunge from $>0.85$ to $<0.50$, destroying the Pareto distribution. Boundary conditions act as structural factors determining physical phases.

### F2 ｜ Policy Hard Limits
In high shock risk regions ($p > 0.04$), applying progressive tax $\tau = 0.15$ fails to prevent poverty rate explosion, revealing a physical limit for redistribution policies.

### F3 ｜ Keynesian MPC Triggers Power-Law Collapse
Introducing Keynesian marginal propensity to consume ($c(W) \sim W^{-\alpha}$) completely destroys the Pareto tail ($\beta_{\text{left}} \to 0.000$), confirming that non-linear consumption behaviors disrupt power-law scaling.

---

## 4. Verification & Rigor

1. **Finite-Size Scaling**: Evaluates $N \in [5k, 10k, 20k, 50k, 100k]$ to verify convergence of $R^*(N)$ at the thermodynamic limit.
2. **MLE & Bootstrap CIs**: Uses Hill Estimator with 200 Bootstrap resamples for 95% CIs and 30%–60% cutoff sensitivity analysis.
3. **Model Selection**: Employs Custom Pseudo-Likelihood Ratio Comparison (AIC/BIC) to confirm power-law superiority over lognormal and exponential models in candidate phase regions.
