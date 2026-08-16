# 财富不平等的相变现象：基于 Kesten 随机动力学的主动学习实证研究

**Phase Transitions in Wealth Inequality: an Active Learning Approach on Kesten Stochastic Dynamics**

[![版本](https://img.shields.io/badge/版本-1.1.0-blue.svg)](https://github.com/angelazu-builder/Datawhale_AI4S)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![许可证](https://img.shields.io/badge/许可证-MIT-orange.svg)](LICENSE)

> Datawhale 夏令营 AI for Research · 开放探索赛道

---

## 一、研究背景与核心问题

财富分布的幂律尾部（Pareto tail）是跨越物理、经济与社会科学的普遍现象。**Kesten 随机过程**给出了它最简洁的物理机制：每步财富以随机乘法冲击（增长）与加法耗散（消费/损失）共同驱动。

**核心研究问题**：当冲击强度与耗散率之比 $R^* = p/c$ 越过某一临界值时，系统贫困左尾的衰减指数 $\beta_{\text{left}}$ 是否会发生**非连续的相变跳变**？AI 主动学习智能体能否比随机搜索更高效地定位这一临界边界，并揭示现有再分配政策的有效性上限？

---

## 二、核心实验结果

> 以下图表均由代码自动生成，可通过 `python3 main.py --mode full` 完整复现。

### 2.1 相变临界曲线（最重要的发现）

固定耗散率 $c=0.10$、边界条件 `reflect`、收入分布 `lognormal`，沿冲击概率 $p$ 进行控因 1D 扫描。图中 CUSUM 检测到 $\beta_{\text{left}}$ 在 $R^* \approx 0.15$–$0.20$ 处发生非连续跳变。

![相变临界曲线 R* = p/c](outputs/phase_transition_R_star.png)

### 2.2 意料之外的发现：边界条件触发结构性坍塌

> 🔴 **这是跑代码前完全没有预期到的发现。**

当边界条件从 `reflect`（反射）切换为 `absorb`（财富触底归零）时，幂律拟合优度 $R^2$ 从 $> 0.85$ 骤降至 $< 0.50$——幂律分布整体失效。这表明**边界条件不只是数值参数，而是决定系统所处"相"的结构性因素**。

左尾分布模型选择（Power-Law vs. Lognormal vs. Exponential，AIC/BIC 比较）：

![左尾分布模型选择](outputs/left_tail_fit.png)

### 2.3 主动学习智能体 vs. 随机搜索

等预算条件下（各 30 轮），UCB + 高斯过程代理模型与随机搜索的累计探索收益对比：

![主动学习 vs. 随机搜索](outputs/baseline_comparison.png)

### 2.4 AI 智能体参数空间采样轨迹

![智能体探索轨迹](outputs/agent_trajectory.png)

### 2.5 中美实证数据校准与反事实政策推演

将模型参数对齐至世界银行 1980–2023 年中美基尼系数，并推演"无精准扶贫补贴"与"维持高累进税"的反事实政策效果：

![中美历史数据校准](outputs/us_china_historical_comparison.png)

![反事实政策推演](outputs/policy_counterfactual_simulation.png)

---

## 三、研究设计

### 模型框架

$$W_{t+1} = A_t \cdot W_t + B_t$$

其中 $A_t$ 为随机乘法冲击因子（冲击概率 $p$），$B_t$ 为加法收入项（耗散率 $c$）。主控参数：$R^* = p/c$。

**固定量**：$N=100{,}000$，$T=500$，5 随机种子，评估指标 $\beta_{\text{left}}$、$R^2$、贫困率。

**可探索参数空间**：

| 类型 | 参数 | 取值范围 |
|------|------|----------|
| 连续物理参数 | 冲击概率 $p$ | $[0.001,\ 0.1]$ |
| 连续物理参数 | 耗散率 $c$ | $[0.01,\ 0.3]$ |
| 政策参数 | 累进税率 $\tau$ | $[0,\ 0.20]$ |
| 政策参数 | 最低生活补贴 $S$ | $[0,\ 0.05]$ |
| 离散机制 | 边界条件 | `reflect` / `absorb` / `soft_clamp` |
| 离散机制 | 收入分布 | `lognormal` / `exponential` / `uniform` |

### 主要发现

| 类型 | 发现 |
|------|------|
| 正向发现 D1 | 相变临界区 $R^* \approx 0.15$–$0.20$，CUSUM 检测，10 种子重复验证 |
| 异常 D2 | `absorb` 边界触发幂律失效，$R^2 < 0.50$（预期之外）|
| 负结果 D3 | $p > 0.04$ 时累进税无法遏制贫困率反弹，政策失效硬边界 |
| 实证锚点 | 中美参数差异与历史体制结构定性吻合 |

---

## 四、工程架构 (v1.1.0)

```
Datawhale夏令营-AI4Research/
├── main.py                    # 统一命令行入口
├── src/
│   ├── simulator.py           # Numba JIT 加速 Kesten 基础模拟器
│   ├── behavioral_simulator.py # [扩展] 凯恩斯 MPC + 前景理论损失厌恶
│   ├── network_simulator.py   # [扩展] Scale-Free + Small-World 网络拓扑
│   ├── agent.py               # Baseline / Random / Active Learning 智能体
│   ├── analysis.py            # CUSUM 变异检验 + AIC/BIC 模型选择
│   ├── data_loader.py         # 世界银行 API 数据抓取
│   ├── calibration.py         # 中美宏观参数校准
│   ├── policy_analysis.py     # 反事实政策推演
│   ├── logger.py              # JSON/CSV 持久化日志
│   └── visualizer.py         # 高清图表生成
├── docs/
│   ├── LOGBOOK.md             # 探索日志（当前状态、发现、下一步）
│   └── 问题定义文档_v2_重写版.docx
└── outputs/                   # 自动生成的图表与科学信号日志
```

---

## 五、快速复现

```bash
# 安装依赖（支持纯 NumPy 降级运行）
pip install numpy scipy numba matplotlib pandas

# 完整科研流水线（约 3–5 分钟）
python3 main.py --mode full

# 仅运行凯恩斯行为扩展
python3 main.py --mode behavioral

# 仅运行网络拓扑扩展
python3 main.py --mode network

# 运行实证校准与政策推演
python3 main.py --mode empirical

# 单次参数实验
python3 main.py --mode single --p 0.02 --c 0.1 --boundary reflect
```

---

## 六、参考文献

- **[E1]** Kesten, H. (1973). *Random difference equations and Renewal theory for products of random matrices*. Acta Mathematica, 131(1), 207–248.
- **[E2]** Bouchaud, J. P., & Mezard, M. (2000). *Wealth condensation in a simple model of economy*. Physica A, 282(3–4), 536–545.
- **[E3]** Gabaix, X. (2009). *Power laws in economics and finance*. Annual Review of Economics, 1(1), 255–294.
- **[E4]** 世界银行开放数据 (2024). *基尼系数与底层 20% 收入占比数据 (SI.POV.GINI, SI.DST.FRST.20)*. World Bank Group.
- **[E5]** World Inequality Database (WID.world, 2024). *中美不平等历史数据系列 (1980–2023)*.
- **[E6]** 清华大学 FIB-Lab (2024). *AgentSociety：大规模社会经济智能体仿真框架*. https://github.com/tsinghua-fib-lab/agentsociety/

---

## 七、后续研究路线

1. **提升 AI 主动学习显著性**：增加预算至 80–100 轮，调优 UCB 参数，Bootstrap 重采样代替 Welch's t-test。
2. **精化相变边界**：密集扫描 $R^* \in [0.12, 0.22]$（步长 0.005），计算一阶导峰值置信区间。
3. **连续场动力学（v2.0）**：从离散 Agent 迁移至 Fokker-Planck 财富密度场方程。
4. **LLM 假说生成**：将 `scientific_signals.json` 喂给大语言模型，实现 AI 驱动的假说迭代闭环。
