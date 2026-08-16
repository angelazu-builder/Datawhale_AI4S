# Datawhale 夏令营 AI for Research · 开放探索赛道
## 初赛问题定义文档 (Problem Definition Document v2)

**项目名称**: 财富不平等的候选相变现象：基于 Kesten 随机动力学的主动学习实证研究  
**英文名称**: Candidate Phase Transitions in Wealth Inequality: an Active Learning Approach on Kesten Stochastic Dynamics  
**作者**: 参赛团队  
**日期**: 2026-08-16  

---

## 1. 为什么该问题值得用 AI 进行探索？ (Why this problem is worthy of AI exploration)

### 1.1 科学背景与核心痛点
社会财富分配的极端不平等与帕累托尾部（Pareto Tail）是统计物理与经济学（Econophysics）交叉领域的经典难题。Kesten 随机过程 ($W_{t+1} = A_t \cdot W_t + B_t$) 刻画了随机收益率 $A_t$ 与随机加性收入 $B_t$ 下财富的动态演化。

然而，传统的非线性统计物理模型面临**高维参数空间网格搜索灾难**：
- 冲击概率 $p$、消费/耗散率 $c$、系统边界条件（反射界 `reflect` vs 吸收界 `absorb`）、收入分布形态（对数正态 `lognormal` vs 帕累托 `pareto`）、累进税率 $\tau$ 以及最低生活补贴 $S$ 构成了庞大的多维离散/连续混合参数空间。
- 人类研究者在人工设定网格时，往往只能进行粗粒度线扫，极其容易遗漏**非连续相变点 (Phase Transition Thresholds)** 与**临界突变边界**。

### 1.2 AI 干预的必要性与不可替代性
- **自主假说搜索 (Active Learning)**：通过高斯过程代理模型 (GP Surrogate) 与上置信界 (UCB) 采集函数，AI 智能体能够自主识别不确定性最高或梯度最大的参数区域，实现以最小采样点精准定位临界相变点 $R^* = p/c$。
- **反例与突变捕获 (Mutation Signal Detection)**：AI 部署 CUSUM 统计变异检验，实时监控贫困左尾衰减指数 $\beta_{\text{left}}$ 的非连续跃迁，自动发现传统粗粒度网格无法捕捉的系统结构坍塌（如吸收界 `absorb` 下的幂律消失）。

---

## 2. 问题定义与实验环境设计 (Problem Definition & Environment Design)

### 2.1 状态空间与控制变量 (State Space & Controls)
我们将模拟环境定义为 $N = 100,000$ 个 Agent 的随机动力学系统，演化时间步 $T = 500$。

- **固定控制量 (Controls)**:
  - Agent 数量 $N = 100,000$
  - 时间步长 $T = 500$
  - 初始财富分布 $W_0 \sim \text{Lognormal}(\mu=1.0, \sigma=0.5)$
- **探索参数变量 (Exploration Variables)**:
  - 外部冲击概率 $p \in [0.001, 0.20]$
  - 消费耗散率 $c \in [0.01, 0.50]$
  - 边界条件 $\text{Boundary} \in \{\text{reflect}, \text{absorb}\}$
  - 政策工具：累进税率 $\tau \in [0.0, 0.30]$，最低生活补贴 $S \in [0.0, 0.05]$

### 2.2 候选相变点定义 (Candidate Phase Transition Definition)
在严格控制变量下（固定 $c=0.10$, `reflect`, `lognormal`），沿单轴 $p$ 进行细粒度扫描。定义临界比值 $R^* = p/c$。
当系统的左尾贫困指数 $\beta_{\text{left}}$ 发生非连续跳变（相对跳变幅度 $\Delta \beta > 0.3$ 且跨 10 个独立验证种子一致跳变）时，标记该临界区间为 **候选相变点 (Candidate Phase Transition)**。

---

## 3. 主要科学发现 (Core Scientific Discoveries)

### F1 ｜ 吸收边界触发系统结构坍塌 (Unexpected Structural Collapse)
当系统边界条件切换为吸收界 (`absorb`，财富触底归零) 时，系统的幂律拟合优度 $R^2$ 从 $>0.85$ 骤降至 $<0.50$，帕累托分布完全失效。这表明边界条件不仅是数值细节，而是决定系统相的关键物理结构参数。

### F2 ｜ 政策有效性的硬边界 (Policy Failure Boundary)
在冲击风险 $p > 0.04$ 的高风险区域，即使施加 $\tau = 0.15$ 的累进再分配税，贫困率仍会爆炸式上升。表明纯再分配政策在面对高频外部冲击时存在物理失效硬边界。

### F3 ｜ 凯恩斯 MPC 行为模式下的幂律崩溃 (Keynesian MPC Power-Law Collapse)
当引入穷人消费边际倾向高的凯恩斯行为函数 $c(W) \sim W^{-\alpha}$ 时，财富分布的帕累托长尾完全消失 ($\beta_{\text{left}} \to 0.000$)，证实了非线性消费行为会对幂律缩放产生毁灭性破坏。

---

## 4. 实验验证与严谨性设计 (Verification & Scientific Rigor)

1. **有限尺寸缩放分析 (Finite-Size Scaling)**: 测试 $N \in [5,000, 10,000, 20,000, 50,000, 100,000]$，验证 $R^*(N)$ 临界点在热力学极限下的收敛性。
2. **最大似然估计与 Bootstrap 置信区间**: 采用 Hill / Newman Estimator 结合 200 次 Bootstrap 重采样求解 95% 置信区间，并进行 30%–60% 截断点敏感性测试。
3. **模型选择检验 (Model Selection)**: 采用 AIC/BIC 似然比比对 (Custom Pseudo-Likelihood Ratio Comparison)，严谨证实幂律分布在特定相区优于对数正态与指数分布。
