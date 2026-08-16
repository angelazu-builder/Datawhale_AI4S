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

## 二、探索迭代过程（共 60 轮）

AI 智能体的探索分为两个阶段，数据均来自 `outputs/kesten_exploration_log.csv`：

**第一阶段：随机基线探索（第 1–30 轮，RandomAgent）**

在 $(p, c, \text{边界条件})$ 空间均匀随机采样。$\beta_{\text{left}}$ 值大幅波动（0.92–4.48），偶尔击中临界区（如第 6 轮 $R^*=0.138$，$\beta=1.78$；第 18 轮 $R^*=0.007$，$\beta=4.48$），但无法形成系统性认识——大量预算浪费在"无聊"的高 $R^*$ 稳态区。

**第二阶段：主动学习控因扫描（第 31–60 轮，AdaptiveAgent）**

固定 $c=0.10$、`reflect` 边界、`lognormal` 收入分布，沿 $p$ 做单参数扫描。$\beta_{\text{left}}$ 从 2.44（极低 $p$）单调降至 0.87（高 $p$），在 $R^* \approx 0.10$–$0.15$ 附近发生约 **1.4 个单位的非连续跳变**：

```
随机阶段 β：  0.92 ░ 4.48 ░ 1.78 ░ 0.97 ░ 2.37 ░ ...  ← 无方向感
主动学习 β：  2.44 → 1.05 → 0.97 → 0.87              ← 单调相变结构清晰可见
                    ↑
              R*≈0.1 处的临界跳变点
```

这正是 AI 驱动科研的核心价值：**把随机探索中偶发的临界信号，转化为可复现的系统性相变曲线**。完整迭代记录见 `docs/LOGBOOK.md`。

---

## 三、核心实验结果

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

### 2.6 经济学行为模式的引入：凯恩斯 MPC 使幂律完全崩溃（第二个预期之外的发现）

> 来自会议建议："把经济学书上的定性行为模式给 AI，看跑出来什么结果。"

将凯恩斯边际消费倾向（MPC：富人耗散率随财富上升而降低 $c(W) \sim W^{-\alpha}$）和前景理论损失厌恶（穷人冲击概率随财富下降而上升 $p(W) \sim 1/W$）编码为可计算的动力学模式，与线性 Kesten 基准对比（$p=0.02$，$c=0.10$）：

| 行为模式 | $\beta_{\text{left}}$ | 贫困率 |
|---------|----------------------|-------|
| 线性 Kesten（基准） | 1.476 | 0.855 |
| 凯恩斯 MPC（富人储蓄率高） | **0.000** ⚠️ | 0.936 |
| 前景理论损失厌恶（穷人风险更高） | 1.009 | 0.832 |
| 组合（MPC + 损失厌恶） | 0.739 | 0.809 |

**凯恩斯 MPC 将 $\beta_{\text{left}}$ 打至 0.000——幂律分布完全消失**，比吸收边界坍塌更极端。机制：富人耗散率随财富增大而降低，财富向顶端加速凝聚，底层完全失去自相似尾部结构。**这是第二个跑代码前没有预期到的发现。**

---

## 四、研究设计

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

**（一）正向发现 D1：相变临界边界**

- 定义：CUSUM 检测到 $\beta_{\text{left}}$ 相对邻域基线发生显著跳变（$\Delta\beta > 0.3$），且 10 个独立验证种子中 $\ge 8/10$ 次复现同方向跳变。
- 结果：在控因 1D 扫描（固定 $c=0.10$, `reflect`, `lognormal`）中，临界区定位于 $R^* \approx 0.15$–$0.20$。
- 意义：首次在无混杂条件下（固定边界条件与收入分布）证实了 Kesten 过程临界相变的存在性。

**（二）异常 D2：吸收边界触发系统结构坍塌（最意外的发现）**

- 当边界条件从 `reflect` 切换为 `absorb`（财富触底归零）时，幂律拟合优度 $R^2$ 从 $> 0.85$ 骤降至 $< 0.50$，幂律分布整体失效。
- 这表明**边界条件不只是数值参数，而是决定系统所处"相"的结构性因素**——这在跑代码前完全没有预期。

**（三）负结果 D3：政策失效存在硬边界**

- 在高冲击风险 $p > 0.04$ 的参数区域，即使施加 $\tau = 0.15$ 的累进税，贫困率仍然无法遏制甚至反弹。
- 纯再分配政策的有效性存在一个由冲击频率决定的硬上限——只有在冲击风险本身受到控制的前提下，税收再分配才有效。这与政策经济学中"精准补贴优于税收再分配"的研究方向定性吻合，但提供了来自物理机制的解释。

**（四）实证校准锚点**

- 世界银行数据校准后：中国呈高耗散率 $c$（消费/税收强）、低冲击风险 $p$；美国相反。
- 反事实推演：若中国 2015–2020 无精准扶贫补贴（$S=0$），贫困率将反弹约 2%；美国若维持低累进税，底层 20% 财富占比持续下降。

---

### 与赛题三维评分标准的对应关系

| 评分维度 | 对应设计 |
|----------|----------|
| **问题定义与环境设计 (45%)** | 固定量：$N$, $T$, 种子, 评估指标；探索量：$p$, $c$, $\tau$, $S$, 边界条件, 收入分布；反馈机制：$\beta_{\text{left}}$ 梯度 + CUSUM 显著量 + UCB 评分 |
| **探索过程与科学信号 (35%)** | D1 相变临界（正向）+ D2 吸收边界坍塌（异常）+ D3 政策失效边界（负结果）+ 中美实证校准 |
| **可检查性与可延续性 (20%)** | 等预算双对照组 + Welch's t-test + 10-种子控因验证 + AIC/BIC 模型选择 + World Bank API 数据透传 |

---

## 五、研究迭代记录（来自会议反馈与导师建议）

本研究经历了两轮重要的方法论迭代，均来自外部反馈：

**迭代一（评审建议 → 修复混杂实验）**  
早期相变图同时改变了 $p$、$c$、边界条件和收入分布，然后压成 $R^*$ 比值——这是混杂实验，任何单个变量的变化都可能解释跳变。→ **修复**：改为控因 1D 扫描（固定 $c=0.10$, `reflect`, `lognormal`），消除混杂。

**迭代二（会议建议 → 引入经济学行为模式，发现幂律崩溃）**  
会议建议："把经济学书上的定性行为模式给 AI，看跑出来什么结果。"→ 实现了凯恩斯 MPC $c(W) \sim W^{-\alpha}$（富人耗散率随财富上升而降低）和前景理论损失厌恶 $p(W) \sim 1/W$（穷人冲击概率更高），与线性基准对比：

| 行为模式 | $\beta_{\text{left}}$ | 贫困率 |
|---------|----------------------|-------|
| 线性 Kesten（基准） | 1.476 | 0.855 |
| 凯恩斯 MPC | **0.000** ⚠️ 幂律完全崩溃 | 0.936 |
| 前景理论损失厌恶 | 1.009 | 0.832 |
| 组合（MPC + 损失厌恶） | 0.739 | 0.809 |

**发现**：凯恩斯 MPC 将 $\beta_{\text{left}}$ 打至 0——幂律完全失效。机制：富人耗散率随财富降低 → 财富向顶端加速凝聚 → 底层失去自相似尾部结构。这是**第二个跑代码前完全没预期到的发现**。

**其他两条会议建议的响应**：
- 清华 FIB-Lab AgentSociety [[E6]](https://github.com/tsinghua-fib-lab/agentsociety/)：已引用为下一步多智能体框架参考；`src/network_simulator.py` 实现了 Scale-Free + Small-World 网络拓扑的初步结构化动力学。
- "把人看成结构"：v2.0 目标为 Fokker-Planck 连续场方程 $\partial_t f = -\partial_w[\mu f] + \partial_w^2[Df]$，将离散 agent 替换为财富密度场。

完整迭代日志见 [`docs/LOGBOOK.md`](docs/LOGBOOK.md)。

---

## 六、工程架构 (v1.1.0)

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

## 七、快速复现

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

## 八、参考文献

- **[E1]** Kesten, H. (1973). *Random difference equations and Renewal theory for products of random matrices*. Acta Mathematica, 131(1), 207–248.
- **[E2]** Bouchaud, J. P., & Mezard, M. (2000). *Wealth condensation in a simple model of economy*. Physica A, 282(3–4), 536–545.
- **[E3]** Gabaix, X. (2009). *Power laws in economics and finance*. Annual Review of Economics, 1(1), 255–294.
- **[E4]** 世界银行开放数据 (2024). *基尼系数与底层 20% 收入占比数据 (SI.POV.GINI, SI.DST.FRST.20)*. World Bank Group.
- **[E5]** World Inequality Database (WID.world, 2024). *中美不平等历史数据系列 (1980–2023)*.
- **[E6]** 清华大学 FIB-Lab (2024). *AgentSociety：大规模社会经济智能体仿真框架*. https://github.com/tsinghua-fib-lab/agentsociety/

---

## 九、后续研究路线

1. **提升 AI 主动学习显著性**：增加预算至 80–100 轮，调优 UCB 参数，Bootstrap 重采样代替 Welch's t-test。
2. **精化相变边界**：密集扫描 $R^* \in [0.12, 0.22]$（步长 0.005），计算一阶导峰值置信区间。
3. **连续场动力学（v2.0）**：从离散 Agent 迁移至 Fokker-Planck 财富密度场方程。
4. **LLM 假说生成**：将 `scientific_signals.json` 喂给大语言模型，实现 AI 驱动的假说迭代闭环。
