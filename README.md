# 扩展 Kesten 财富过程与相变现象 AI 自主探索引擎 (v1.1.0)
## Extended Kesten Wealth Process & Phase Transition AI Research Engine

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/Datawhale/Kesten-AI-Research)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

> **Datawhale 夏令营 AI for Research (AI4Science / Econophysics) 开放探索赛道项目**

---

## 📌 一、 项目概述与研究背景 (Overview)

在统计物理与宏观经济学（Econophysics）中，**Kesten 随机过程 (Kesten Process)** 是描述社会财富不平等、帕累托尾部（Pareto Tail）与极端贫富分化的经典数学模型。

本项目构建了一个 **高能 Numba JIT 加速的大规模 Agent 随机动力学模拟环境**，部署了基于 **主动学习 (Active Learning / UCB Surrogate) 与 CUSUM 变异检测算法的 AI 科研智能体 (Adaptive Research Agent)**，并成功接入 **世界银行 (World Bank) 与世界不平等数据库 (WID.world) 真实 API 数据**。能够将 Kesten 物理模型与中美（US vs China）历史宏观基尼系数与基底财富占比进行对齐校准，评估重大历史事件（如美国 1981 里根减税、中国 2015-2020 精准扶贫）的政策效应与反事实推演。

---

## 🎯 二、 赛题评分标准对齐 (Evaluation Alignment)

本项目严格按照 **Datawhale AI4Research 开放探索赛题 3 大评分维度** 进行设计与实现：

### 1. 问题定义与环境设计质量 (45%)
- **真实切片**: 切中真实物理经济学中的财富集中度与贫困左尾指数 $\beta_{left}$ 求解问题。
- **固定部分 (Fixed)**: 种群规模 $N = 100,000$、模拟时间步 $T = 500$、5 随机种子控制、评估指标（左尾指数 $\beta_{left}$、拟合优度 $R^2$、贫困率 Poverty Rate、峰度 Kurtosis）。
- **可探索部分 (Explorable)**:
  - 连续物理参数：冲击概率 $p \in [0.001, 0.1]$，耗散率 $c \in [0.01, 0.3]$。
  - 政策干预参数：累进税率 $\tau \in [0, 0.20]$，最低生活补贴 $S \in [0, 0.05]$。
  - 离散机制：边界条件（`reflect` 反射 / `absorb` 吸收 / `soft_clamp` 软截断），收入分布（`lognormal` / `exponential` / `uniform`）。
- **反馈机制 (Feedback)**: $\beta_{left}$ 的梯度变化、CUSUM 显著性变化量及 UCB 置信上界评分。

### 2. 探索过程与科学/研究信号 (35%)
- **正向发现 (Positive Discoveries)**: 准确识别相变临界阈值 $R^* = p/c$ 及 D1 状态突变信号。
- **反例与异常 (Anomalies / Counterexamples)**: 捕获吸收界 (`absorb`) 及极端税率下系统失去幂律分布 ($R^2 < 0.5$) 的物理塌陷现象。
- **负结果 (Negative Results)**: 揭示高冲击风险下 ($p > 0.04$)，单靠二次分配累进税无法遏制贫困率爆炸的政策失效边界。
- **实证数据校准与反事实政策推演**: 校准中美历史参数，推演“无精准扶贫补贴 ($S=0$)”与“维持高累进税 ($\tau=0.15$)”的反事实政策效果。

### 3. 可检查性与可延续性 (20%)
- **双参照系与等预算对比 (Dual Baseline Benchmark)**: 包含 `BaselineAgent` (无干预对照组) 和 `RandomAgent` (100轮等预算随机采样)，并通过 95% 置信区间与 Welch's t-test 进行显著性检验 ($p < 0.01$)。
- **10-种子控因重复验证**: 针对捕获的变异信号，在完全相同的控制参数下使用 10 个独立未采样验证种子进行重复测试（要求 $\ge 8/10$ 次独立复现）。
- **模型选择拟合 (Model Selection)**: 计算 Power-Law、Lognormal 与 Exponential 的 AIC/BIC 指标及 Likelihood Ratio 检验，证明左尾幂律拟合显著优于其他分布。
- **数据源透传与持久化**: 自动连接 World Bank API 并支持本地 CSV 缓存 (`empirical_us_china_data.csv`) 与分类科学信号包 `scientific_signals.json`。

---

## 🛠️ 三、 软件架构设计 (Repository Architecture v1.1.0)

```
Datawhale夏令营-AI4Research/
├── README.md                           # 项目主页与说明文档 (v1.1.0)
├── LICENSE                             # MIT 开源协议文件
├── requirements.txt                    # 依赖包列表
├── pyproject.toml                      # 标准 Python 包构建元数据
├── .gitignore                          # Git 忽略配置
├── main.py                             # 命令行运行统一入口
├── src/                                # 核心算法与仿真物理引擎包
│   ├── __init__.py                     # 导出模块列表 (v1.1.0)
│   ├── config.py                       # 仿真与探索引擎配置 (N=100,000, T=500)
│   ├── simulator.py                    # Numba JIT 加速 & 纯 NumPy 降级基础 Kesten 模拟器
│   ├── behavioral_simulator.py         # [扩展分支 1] 非线性经济行为模拟器 (凯恩斯 MPC & 前景理论损失厌恶)
│   ├── network_simulator.py            # [扩展分支 2] 空间复杂网络拓扑模拟器 (Scale-Free & Small-World 图结构)
│   ├── agent.py                        # Baseline, Random 及 Active Learning 智能体
│   ├── analysis.py                     # CUSUM 变异检验与 AIC/BIC 模型选择
│   ├── data_loader.py                # 世界银行 API 数据抓取模块
│   ├── calibration.py                # 中美宏观基尼系数校准匹配器
│   ├── policy_analysis.py            # 历史事件与反事实政策推演引擎
│   ├── logger.py                     # 实验日志与 JSON/CSV 持久化
│   └── visualizer.py                   # 高清分析图表绘制引擎
├── docs/                               # 大赛答辩文档与 PPT 资料
│   ├── Econophysics_preliminary.pptx
│   └── 开放探索赛初赛问题定义文档_filled.docx
├── outputs/                            # 自动生成的分析图表与科学日志
│   ├── phase_transition_R_star.png
│   ├── left_tail_fit.png
│   ├── us_china_historical_comparison.png
│   ├── policy_counterfactual_simulation.png
│   ├── agent_trajectory.png
│   ├── baseline_comparison.png
│   └── empirical_us_china_data.csv
└── notebooks/                          # 交互式 Jupyter Demo 笔记本
    └── exploration_demo.ipynb
```

---

## 🚀 四、 快速复现与运行说明 (Reproduction Guide)

### 1. 安装依赖 (支持纯 NumPy 降级运行)
```bash
pip install numpy scipy numba matplotlib pandas
```

### 2. 运行非线性经济行为扩展分支 (--mode behavioral)
```bash
python3 main.py --mode behavioral
```

### 3. 运行空间复杂网络拓扑扩展分支 (--mode network)
```bash
python3 main.py --mode network
```
*(注：即使未安装 numba / scipy / matplotlib / pandas，系统也会自动启用纯 NumPy 向量化降级引擎，保证全流程代码均可正常运行)*

### 2. 运行真实 API 数据校准与政策推演 (--mode empirical)
```bash
python3 main.py --mode empirical
```

### 3. 运行完整科研流水线 (--mode full)
```bash
python3 main.py --mode full
```

### 4. 运行单次实验或智能体探索
```bash
python3 main.py --mode single --p 0.02 --c 0.1 --boundary reflect
python3 main.py --mode explore
```

---

## 📈 五、 自动生成分析图表说明 (Generated Artifacts)

项目会生成以下高清分析图表：
1. `us_china_historical_comparison.png`: 1980–2023 中美基尼系数与底层 20% 财富占比历史演化对比图。
2. `policy_counterfactual_simulation.png`: 中美反事实政策推演对比图（精准扶贫补贴效应 & 累进税政策效应）。
3. `phase_transition_R_star.png`: 控因 1D 参数路径（固定 $c=0.10$, `reflect`, `lognormal`）下的 $R^* = p/c$ 相变曲线与临界跳变点图。
4. `agent_trajectory.png`: Active Learning Agent 在 100 轮探索中的参数空间采样轨迹。
5. `baseline_comparison.png`: Active Learning 智能体 vs 随机搜索 Baseline (100 vs 100 轮等预算) 累计发现收益对比图。
6. `left_tail_fit.png`: 财富分布左尾模型选择 (Power-Law vs. Lognormal vs. Exponential AIC/BIC 比对) 图。

---

## 📚 六、 参考文献 (References & Bibliography)

- **[E1] Kesten, H. (1973)**. *Random difference equations and Renewal theory for products of random matrices*. Acta Mathematica, 131(1), 207-248.
- **[E2] Bouchaud, J. P., & Mezard, M. (2000)**. *Wealth condensation in a simple model of economy*. Physica A: Statistical Mechanics and its Applications, 282(3-4), 536-545.
- **[E3] Gabaix, X. (2009)**. *Power laws in economics and finance*. Annual Review of Economics, 1(1), 255-294.
- **[E4] World Bank Open Data (2024)**. *Gini Index & Income Share Metrics (SI.POV.GINI, SI.DST.FRST.20)*. World Bank Group.
- **[E5] World Inequality Database (WID.world, 2024)**. *US and China Inequality Historical Series (1980-2023)*.
- **[E6] Tsinghua FIB-Lab (2024)**. *AgentSociety: Large-Scale Social and Economic Agent Simulation Framework*. GitHub Repository: https://github.com/tsinghua-fib-lab/agentsociety/

---

## 🔮 七、 后续研究路线图 (Future Research Roadmap)

1. **复杂网络上的 Kesten 扩散**: 将独立的 Agent 扩展至小世界网络与无标度网络。
2. **大模型引导假说与自动报告生成**: 结合 LLM Agent 读取 `scientific_signals.json` 自动生成科研报告。
