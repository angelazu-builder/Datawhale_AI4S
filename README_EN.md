**English** · [中文](README.md)

# Candidate Phase Transitions in Wealth Inequality: An Active Learning Framework for Kesten Stochastic Dynamics

[![Version](https://img.shields.io/badge/Version-v1.1.0-blue.svg)](https://github.com/angelazu-builder/Datawhale_AI4S)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-orange.svg)](LICENSE)

> Datawhale AI for Research Camp · Open Exploration Track

---

## 📌 Quick Navigation

- 🏆 **Competition Deliverables**: [docs/submission/](docs/submission/) (Problem Definition v2, Original Version, & Final Research Report in Word/Markdown)
- 📔 **Peer Reviewers & Researchers**: [docs/logbook/LOGBOOK_EN.md](docs/logbook/LOGBOOK_EN.md) (60-round exploration data, experimental iteration logs, & responses to 4 conference feedback items)
- 📊 **Figures & Data Index**: [outputs/](outputs/) (High-resolution plots and raw CSV/JSON experiment logs)
- 📖 **Background References**: [docs/references/](docs/references/) (Econophysics slides, challenge background documents)

---

## 1. Background & Research Question

The Pareto power-law tail of wealth distribution is a universal phenomenon spanning statistical physics, quantitative economics, and social sciences. The **Kesten Stochastic Process** offers its most elegant physical mechanism: wealth evolution at each discrete time step is co-driven by random multiplicative shocks (growth/return) and additive dissipation (consumption/loss).

$$\text{Kesten Dynamics: } W_{t+1} = A_t \cdot W_t + B_t$$

**Core Research Question**: As the ratio of shock intensity to dissipation rate $R^* = p/c$ crosses a critical threshold, does the poverty left-tail decay exponent $\beta_{\text{left}}$ undergo a **non-continuous candidate phase transition**? Grounded in **Finite-Size Scaling ($N \in [5k, 100k]$)** fitted convergence alongside **Maximum Likelihood Estimation (Hill Estimator)** and **Bootstrap 95% Confidence Intervals**, can an Active Learning AI Agent identify this critical boundary more sample-efficiently than random search?

---

## 2. Exploration Trajectory (60 Rounds)

The AI Agent's exploration trajectory comprises two distinct phases, with raw logs persisted in [`outputs/kesten_exploration_log.csv`](outputs/kesten_exploration_log.csv):

**Phase 1: Random Baseline Exploration (Rounds 1–30, RandomAgent)**  
Uniform random sampling over $(p, c, \text{boundary})$. The left-tail exponent $\beta_{\text{left}}$ fluctuates wildly (0.92–4.48), occasionally hitting the candidate critical region (e.g., Round 6: $R^*=0.138, \beta=1.78$; Round 18: $R^*=0.007, \beta=4.48$), but fails to establish a systematic law—wasting query budget on uninformative high-$R^*$ equilibrium states.

**Phase 2: Controlled Active Learning Sweep (Rounds 31–60, AdaptiveAgent)**  
Fixing $c=0.10$, `reflect` boundary, and `lognormal` income distribution, scanning along 1D shock parameter $p$. $\beta_{\text{left}}$ decreases monotonically from 2.44 (low $p$) to 0.87 (high $p$), exhibiting a **non-continuous candidate jump of ~1.4 units** around $R^* \approx 0.10 \sim 0.15$:

```
Random Phase β:          0.92 ░ 4.48 ░ 1.78 ░ 0.97 ░ 2.37 ░ ...  ← Unguided Uniform Sampling
Active Learning Sweep β: 2.44 → 1.05 → 0.97 → 0.87              ← Monotonic Candidate Phase Jump
                            ↑
                     Candidate Transition Jump at R* ≈ 0.10
```

This captures the core paradigm of AI-driven scientific discovery: **transforming sporadic critical signals from unguided exploration into reproducible, systematic phase transition curves**. Full logs are documented in [`docs/logbook/LOGBOOK_EN.md`](docs/logbook/LOGBOOK_EN.md).

---

## 3. Key Experimental Results

> All figures are programmatically generated and fully reproducible via `python3 main.py --mode full`.

### 3.1 Candidate Phase Transition Curve (Finite-Size Scaling)

1D controlled parameter sweep along shock probability $p$ with fixed dissipation $c=0.10$, `reflect` boundary, and `lognormal` income distribution. CUSUM detects a non-continuous candidate jump in $\beta_{\text{left}}$ around $R^* \approx 0.15 \sim 0.20$. Verified via **Finite-Size Scaling ($N \in [5k, 50k]$)**, the jump derivative $\frac{d\beta}{d(1/N)}$ converges systematically as $N$ increases.

![Candidate Phase Transition R* = p/c](outputs/phase_transition_R_star.png)

### 3.2 Unexpected Finding 1: Absorbing Boundaries Trigger Structural Collapse

> 🔴 **Unanticipated Emergent Behavior Prior to Execution.**

When switching boundary conditions from `reflect` to `absorb` (wealth collapses to zero upon hitting lower bound), power-law fit goodness $R^2$ dramatically drops from $> 0.85$ to $< 0.50$—the power-law distribution collapses completely. This demonstrates that **boundary conditions are not mere numerical parameters, but structural factors dictating the physical phase of the system**.

Model selection for the poverty left-tail (Power-Law vs. Lognormal vs. Exponential, evaluated via **Custom Pseudo-LR Comparison** with AIC/BIC):

![Left Tail Model Selection](outputs/left_tail_fit.png)

### 3.3 Unexpected Finding 2: Keynesian MPC Induces Power-Law Breakdown

> 🔴 **Second Key Scientific Discovery Responding to Conference Feedback.**

Encoding Keynesian Marginal Propensity to Consume (MPC: rich dissipation rate decreases with wealth $c(W) \sim W^{-\alpha}$) and Prospect Theory Loss Aversion (poor shock probability increases with falling wealth $p(W) \sim 1/W$) into behavioral dynamics, contrasted against the standard linear Kesten baseline ($p=0.02, c=0.10$):

![Behavioral Modes & Power-Law Collapse](outputs/behavioral_mode_comparison.png)

| Behavioral Mode | $\beta_{\text{left}}$ | Poverty Rate | Scientific Finding |
| :--- | :--- | :--- | :--- |
| Linear Kesten (Baseline) | 1.476 | 0.855 | Standard Power-Law Distribution |
| Keynesian MPC | **0.000** ⚠️ | 0.936 | **Complete Power-Law Collapse** (Accelerated wealth condensation) |
| Prospect Theory Loss Aversion | 1.009 | 0.832 | Slight attenuation of power-law tail |
| Combined (MPC + Loss Aversion) | 0.739 | 0.809 | Significant weakening of power-law scaling |

### 3.4 Active Learning Agent vs. Random Search Baseline & Effect Size

Comparing cumulative exploration reward of UCB + Gaussian Process Surrogate against Random Search under a fixed query budget (30 rounds each). Reporting **Effect Size (Cohen's d = 0.00)** alongside Welch's t-test to transparently convey agent search dynamics under high-variance environments:

![Active Learning vs Random Search](outputs/baseline_comparison.png)

### 3.5 AI Agent Parameter Space Trajectory

![Agent Parameter Trajectory](outputs/agent_trajectory.png)

### 3.6 Empirical Calibration & Counterfactual Policy Evaluation

Using **Exploratory Macro Calibration**, incorporating exact simulated **Gini coefficient $G$ and Bottom-20% share $S_{20}$** directly into Loss $L = (G_{\text{sim}} - G_{\text{target}})^2 + (S_{20,\text{sim}} - S_{20,\text{target}})^2$, evaluating counterfactual policy scenarios:

![US China Historical Calibration](outputs/us_china_historical_comparison.png)

![Counterfactual Policy Simulation](outputs/policy_counterfactual_simulation.png)

---

## 4. Research Design & Scientific Rigor

### Model Formulation

$$W_{t+1} = A_t \cdot W_t + B_t$$

Control Parameter: $R^* = p/c$. Tail estimation algorithms provide both **MLE (Hill Estimator)** and **Bootstrap 95% Confidence Intervals**, verified via 30%, 40%, 50%, and 60% threshold sensitivity analysis.

---

### Alignment Matrix with Evaluation Criteria

| Evaluation Dimension | Design Implementation | Scientific Rigor Revision |
| :--- | :--- | :--- |
| **Problem Definition & Environment (45%)** | Fixed: $N, T$, seeds; Explored: $p, c, \tau, S$, boundary | Renamed phase jump to **Candidate Phase Transition**; Added Finite-Size Scaling $N \in [5k, 50k]$ |
| **Exploration Process & Signals (35%)** | D1 Candidate Phase Threshold + D2 Boundary & MPC Collapse + D3 Policy Failure | Added **MLE Hill Estimator**, **Bootstrap 95% CI**, and **Threshold Sensitivity Tests** |
| **Reproducibility & Extension (20%)** | Dual baselines + Welch's t-test + **Cohen's d Effect Size** + Model Selection | Explicitly labeled as **Custom Pseudo-LR Comparison** and **Exploratory Macro Calibration** |

---

## 5. Research Iteration Log (Peer & Conference Feedback)

This research underwent two major methodological iterations and one rigor revision:

- **Iteration 1 (Peer Reviewer Feedback → Fix Confounded Experiment)**  
  Early phase plots varied $p, c$, boundary, and income distribution simultaneously. → **Fixed**: 1D controlled scan (fixing $c=0.10$, `reflect`, `lognormal`), eliminating confounding factors.
- **Iteration 2 (Conference Feedback → Introduce Economic Behavioral Modes)**  
  Conference feedback: *"Test qualitative behavioral modes from economic textbooks."* → Implemented Keynesian MPC and Prospect Theory, discovering the **first power-law collapse under behavioral dynamics**.
- **Scientific Rigor Revisions**:
  - Clarified candidate phase transition definition; added Finite-Size Scaling.
  - Introduced MLE Hill estimator and Bootstrap CIs.
  - Labeled Custom Pseudo-LR comparison and Exploratory Calibration transparently.

Full iteration log available in [`docs/logbook/LOGBOOK_EN.md`](docs/logbook/LOGBOOK_EN.md).

---

## 6. Repository Architecture (v1.1.0)

```
Datawhale_AI4S / Datawhale夏令营-AI4Research/
├── README.md                      # 🌟 Chinese Main Readme
├── README_EN.md                   # 🌟 English Main Readme
├── LICENSE                        # MIT License
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Package Metadata
├── main.py                        # CLI Runner Entrypoint
│
├── src/                           # 🧠 Core Simulator & Algorithm Package
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Simulation configurations
│   ├── simulator.py               # Numba JIT & NumPy Kesten Simulator
│   ├── behavioral_simulator.py    # Keynesian MPC & Prospect Theory Simulator
│   ├── network_simulator.py       # Scale-Free & Small-World Topology Simulator
│   ├── agent.py                   # Baseline / Random / Active Learning Agents
│   ├── analysis.py                # MLE Hill Estimator, Bootstrap CI, Finite-Size Scaling, Custom LR
│   ├── data_loader.py             # World Bank API Data Fetcher
│   ├── calibration.py             # Gini & B20 Loss Calibration Engine
│   ├── policy_analysis.py         # Counterfactual Policy Simulation Engine
│   ├── logger.py                  # Experiment Logging (JSON/CSV)
│   └── visualizer.py              # Plotting Engine
│
├── docs/                          # 📚 Documentation Hub
│   ├── README.md                  # Documentation Guide
│   ├── submission/                # 🏆 Deliverables (01_Problem_Definition, 03_Final_Report)
│   ├── logbook/                   # 📔 Research Logbook & Iteration Logs (LOGBOOK.md)
│   └── references/                # 📖 Econophysics Slides & Templates
│
├── outputs/                       # 📊 Auto-generated Figures & Logs
│   ├── README.md                  # Schema Index
│   ├── behavioral_mode_comparison.png
│   ├── phase_transition_R_star.png
│   ├── baseline_comparison.png
│   ├── us_china_historical_comparison.png
│   ├── policy_counterfactual_simulation.png
│   ├── agent_trajectory.png
│   ├── left_tail_fit.png
│   ├── scientific_signals.json
│   ├── kesten_exploration_log.csv
│   └── empirical_us_china_data.csv
│
└── notebooks/                     # 📓 Demo Notebooks
    └── exploration_demo.ipynb
```

---

## 7. Quick Reproduction Guide

```bash
# 1. Install dependencies
pip install numpy scipy numba matplotlib pandas

# 2. Run full research pipeline (includes Finite-Size Scaling, all plots & logs ~3 mins)
python3 main.py --mode full

# 3. Run extension modules
python3 main.py --mode behavioral     # Keynesian MPC & Prospect Theory simulation
python3 main.py --mode network        # Complex Network topology simulation
python3 main.py --mode empirical      # World Bank empirical calibration & policy simulation

# 4. Run single experiment
python3 main.py --mode single --p 0.02 --c 0.1 --boundary reflect
```

---

## 8. References

- **[E1]** Kesten, H. (1973). *Random difference equations and Renewal theory for products of random matrices*. Acta Mathematica, 131(1), 207–248.
- **[E2]** Bouchaud, J. P., & Mezard, M. (2000). *Wealth condensation in a simple model of economy*. Physica A, 282(3–4), 536–545.
- **[E3]** Gabaix, X. (2009). *Power laws in economics and finance*. Annual Review of Economics, 1(1), 255–294.
- **[E4]** Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). *Power-law distributions in empirical data*. SIAM Review, 51(4), 661–703.
- **[E5]** World Bank Open Data (2024). *Gini Index & Income Share Held by Lowest 20% (SI.POV.GINI, SI.DST.FRST.20)*. World Bank Group.
- **[E6]** World Inequality Database (WID.world, 2024). *US-China Income Inequality Historical Series (1980–2023)*.
- **[E7]** Tsinghua FIB-Lab (2024). *AgentSociety: Large-scale Agent Society Simulator*. https://github.com/tsinghua-fib-lab/agentsociety/

---

## 9. Future Research Roadmap

1. **Active Learning Agent Evaluation on Larger Budgets**: Extending budget to 100+ rounds to evaluate Bayesian Optimization boundaries under high noise.
2. **Continuous Field Dynamics (v2.0)**: Transitioning from discrete agents to Fokker-Planck wealth density field equations $\partial_t f = -\partial_w[\mu f] + \partial_w^2[Df]$.
3. **LLM Hypothesis Generation**: Feeding `scientific_signals.json` into LLMs to complete an automated hypothesis loop.
