# Candidate Phase Transitions in Wealth Inequality: An Active Learning Framework for Kesten Stochastic Dynamics

**Datawhale AI for Research Camp · Open Exploration Track Final Research Report**  
**Stage**: Preliminary (v1.1.0) | **Date**: 2026-08-16  

---

## 1. Overview & Background

In statistical physics and econophysics, the Kesten stochastic process ($W_{t+1} = A_t \cdot W_t + B_t$) is a classic mathematical framework describing wealth inequality, Pareto tails, and extreme wealth concentration.

This project constructs a Numba JIT-accelerated multi-agent stochastic dynamics simulation environment, deploys an AI Active Learning Agent using Gaussian Process (GP) surrogates and UCB acquisition, and connects real World Bank API data to calibrate physical parameters with macro-level data from the US and China.

---

## 2. Alignment with Challenge Evaluation Criteria

### 1. Problem Definition & Environment Design (45%)
Addresses poverty left-tail exponent $\beta_{\text{left}}$ and wealth concentration. Fixed: $N=100,000, T=500$; Explored: shock probability $p$, dissipation $c$, tax $\tau$, welfare $S$.

### 2. Exploration Process & Scientific Signals (35%)
- **Positive Finding**: Identifies candidate phase transition threshold $R^* = p/c$ and mutation signals.
- **Anomaly**: Discovers physical collapse of power-law distribution ($R^2 < 0.5$) under absorbing boundary conditions.
- **Negative Result**: Reveals policy failure boundaries ($p > 0.04$) where tax redistribution alone cannot stop poverty explosion.

### 3. Reproducibility & Extension (20%)
Includes BaselineAgent and RandomAgent dual controls, 10-seed controlled verification, Welch's t-test, Cohen's d effect size, and AIC/BIC model selection.

---

## 3. Core Hypotheses & Findings

- **Hypothesis 1 (Phase Threshold)**: Poverty left-tail exponent $\beta_{\text{left}}$ exhibits a candidate phase jump at $R^* = p/c \approx 0.15 \sim 0.20$.
- **Hypothesis 2 (Active Learning Efficiency)**: Active Learning Agent locates phase boundaries significantly faster than random search under equal budget ($p < 0.01$, Welch t-test, Cohen's d effect size).
- **Hypothesis 3 (Behavioral Modes & Topologies)**: Keynesian MPC $c(W) \sim W^{-\alpha}$ and Prospect Theory Loss Aversion $p(W) \sim 1/W$ break linear Pareto scaling and induce power-law collapse.
- **Hypothesis 4 (Empirical Calibration & Policy)**: Calibrates US and China Gini parameters ($R^*_{\text{US}} = 0.125, R^*_{\text{CHN}} = 0.133$), demonstrating that targeted poverty subsidies prevent poverty rebound.

---

## 4. Visualizations & Figure Analysis

![Figure 1: R* Candidate Phase Transition](../../outputs/phase_transition_R_star.png)  
*Figure 1: Candidate Phase Transition along 1D Parameter Scan ($R^* = p/c$)*

![Figure 2: Left Tail Model Selection](../../outputs/left_tail_fit.png)  
*Figure 2: Left Tail Model Selection (Power-Law vs. Lognormal vs. Exponential)*

![Figure 3: US-China Historical Comparison](../../outputs/us_china_historical_comparison.png)  
*Figure 3: 1980–2023 US-China Historical Gini & Bottom-20% Share*

![Figure 4: Counterfactual Policy Simulation](../../outputs/policy_counterfactual_simulation.png)  
*Figure 4: US-China Counterfactual Policy Simulation*

![Figure 5: Keynesian MPC Power-Law Collapse](../../outputs/behavioral_mode_comparison.png)  
*Figure 5: Power-Law Collapse under Keynesian MPC Behavioral Dynamics*

---

## 5. References

- **[E1]** Kesten, H. (1973). *Random difference equations and Renewal theory for products of random matrices*. Acta Mathematica, 131(1), 207–248.
- **[E2]** Bouchaud, J. P., & Mezard, M. (2000). *Wealth condensation in a simple model of economy*. Physica A, 282(3–4), 536–545.
- **[E3]** Gabaix, X. (2009). *Power laws in economics and finance*. Annual Review of Economics, 1(1), 255–294.
- **[E4]** Yakovenko, V. M., & Rosser, J. B. (2009). *Colloquium: Statistical mechanics of money, wealth, and income*. Reviews of Modern Physics, 81(4), 1703.
- **[E5]** World Bank Open Data (2024). *Gini Index & Income Share Held by Lowest 20%*. https://data.worldbank.org/
- **[E6]** Tsinghua FIB-Lab (2024). *AgentSociety: Large-scale Agent Society Simulator*. https://github.com/tsinghua-fib-lab/agentsociety/
