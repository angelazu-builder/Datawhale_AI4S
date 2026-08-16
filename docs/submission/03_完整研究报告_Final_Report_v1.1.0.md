# 财富不平等的候选相变现象：基于 Kesten 随机动力学的主动学习实证研究
## Candidate Phase Transitions in Wealth Inequality: an Active Learning Approach on Kesten Stochastic Dynamics

**Datawhale 夏令营 AI for Research (AI4Science / Econophysics) 开放探索赛道终极研究报告**  
**阶段**: 初赛 (v1.1.0) | **日期**: 2026-08-16  

---

## 一、 项目概述与研究背景 (Overview)

在统计物理与宏观经济学（Econophysics）中，Kesten 随机过程 (Kesten Process) 是描述社会财富不平等、帕累托尾部（Pareto Tail）与极端贫富分化的经典数学模型。

本项目构建了一个高能 Numba JIT 加速的大规模 Agent 随机动力学模拟环境，部署了基于主动学习 (Active Learning / UCB Surrogate) 与变异检测算法的 AI 科研智能体，并接入世界银行 (World Bank) 真实 API 数据，实现了物理模型与中美宏观数据的实证校准和反事实政策推演。

---

## 二、 Datawhale 赛题评分标准深度对齐

### 1. 问题定义与环境设计质量 (45%)
切中物理经济学财富集中度与贫困左尾指数 $\beta_{\text{left}}$ 求解问题。固定 $N=100,000$, $T=500$，开放冲击概率 $p$、耗散率 $c$、累进税率 $\tau$ 与最低生活补贴 $S$。

### 2. 探索过程与科学/研究信号 (35%)
- **正向发现**: 识别候选相变临界阈值 $R^* = p/c$ 及其突变信号。
- **反例/异常**: 捕获吸收界 (`absorb`) 下系统失去幂律分布 ($R^2 < 0.5$) 的物理坍塌现象。
- **负结果**: 揭示高冲击风险下 ($p > 0.04$) 单靠二次分配税收无法遏制贫困率爆炸的政策失效边界。

### 3. 可检查性与可延续性 (20%)
包含 BaselineAgent 与 RandomAgent 等预算双对照组，使用 10-种子控因重复验证、Welch's t-test、Cohen's d 效应量以及 AIC/BIC 候选分布模型选择。

---

## 三、 核心研究假设与实验验证结论

- **假设一 (相变临界边界)**: 系统的贫困左尾衰减指数 $\beta_{\text{left}}$ 在临界比值 $R^* = p/c \approx 0.15 \sim 0.20$ 处存在候选相变跳变点。
- **假设二 (主动学习效率)**: 基于 UCB 代理模型的 Active Learning Agent 在等预算下定位相变边界效率显著高于随机搜索 ($p < 0.01$, Welch t-test, Cohen's d 效应量)。
- **假设三 (非线性行为与网络拓扑)**: 引入凯恩斯 MPC $c(W) \sim W^{-\alpha}$ 与前景理论损失厌恶 $p(W) \sim 1/W$ 或 Scale-Free 幂律网络拓扑 $P(k) \sim k^{-2.5}$ 会打破线性帕累托缩放，诱发财富凝聚与幂律崩溃。
- **假设四 (中美实证校准与政策推演)**: 中美历史基尼系数可精准对齐至 Kesten 参数 ($R^*_{\text{US}} = 0.125, R^*_{\text{CHN}} = 0.133$)，证实中国精准扶贫补贴能有效防止贫困率反弹。

---

## 四、 实验可视化与图表分析

![图 1: 控因 1D 参数路径下的 R* = p/c 候选相变图](../../outputs/phase_transition_R_star.png)  
*图 1: 控因 1D 参数路径下的 $R^* = p/c$ 候选相变跳变图*

![图 2: 财富分布左尾模型选择图](../../outputs/left_tail_fit.png)  
*图 2: 财富分布左尾模型选择 (Power-Law vs Lognormal vs Exponential) 对比图*

![图 3: 1980-2023 中美历史基尼系数与底层 20% 财富占比对比图](../../outputs/us_china_historical_comparison.png)  
*图 3: 1980-2023 中美历史基尼系数与底层 20% 财富占比对比图*

![图 4: 中美历史反事实政策推演对比图](../../outputs/policy_counterfactual_simulation.png)  
*图 4: 中美历史反事实政策推演对比图*

![图 5: 凯恩斯 MPC 行为模式下的幂律分布崩溃对比图](../../outputs/behavioral_mode_comparison.png)  
*图 5: 凯恩斯 MPC 行为模式下的幂律分布崩溃对比图*

---

## 五、 参考文献 (References)

- **[E1]** Kesten, H. (1973). Random difference equations and Renewal theory for products of random matrices. *Acta Mathematica*, 131(1), 207-248.
- **[E2]** Bouchaud, J. P., & Mezard, M. (2000). Wealth condensation in a simple model of economy. *Physica A*, 282(3-4), 536-545.
- **[E3]** Gabaix, X. (2009). Power laws in economics and finance. *Annual Review of Economics*, 1(1), 255-294.
- **[E4]** Yakovenko, V. M., & Rosser, J. B. (2009). Colloquium: Statistical mechanics of money, wealth, and income. *Reviews of Modern Physics*, 81(4), 1703.
- **[E5]** World Bank Open Data. (2024). Gini Index & Income Share Held by Lowest 20%. *https://data.worldbank.org/*
- **[E6]** Tsinghua FIB-Lab. (2024). AgentSociety: Large-scale Agent Society Simulator. *https://github.com/tsinghua-fib-lab/agentsociety*
