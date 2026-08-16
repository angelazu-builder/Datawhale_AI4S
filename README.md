# Phase Transitions in Wealth Inequality: an Active Learning Approach on Kesten Stochastic Dynamics
## 财富不平等中的相变现象：基于 Kesten 随机动力学的主动学习实证研究

[![Version](https://img.shields.io/badge/version-1.1.0-blue.svg)](https://github.com/angelazu-builder/Datawhale_AI4S)
[![Python](https://img.shields.io/badge/python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-orange.svg)](LICENSE)

> Datawhale 夏令营 AI for Research · 开放探索赛道

---

## 📌 一、 研究背景与问题 (Background & Research Question)

财富分布的幂律尾部（Pareto tail）是跨越物理、经济与社会科学的普遍现象。**Kesten 随机过程**是描述这一现象的极简数学模型：每个个体的财富在每步以随机乘法冲击（增长）与加法耗散（消费/损失）共同驱动。

**核心研究问题**：当冲击强度 $p$ 与耗散率 $c$ 的比值 $R^* = p/c$ 跨越某一临界值时，系统财富分布的左尾衰减指数 $\beta_{\text{left}}$ 是否会发生非连续的相变跳变（phase transition）？AI 主动学习智能体能否比随机搜索更高效地定位这一临界边界，并揭示现有再分配政策（累进税、精准补贴）的有效性上限？

本项目通过大规模 Agent 模拟（$N=100,000$，$T=500$）、UCB 代理模型主动学习与 CUSUM 变异检测，结合世界银行 / WID.world 真实宏观数据，对上述问题进行了系统性的实证探索。

---

## 🔬 二、 研究设计 (Research Design)

### 模型框架

每个 agent 的财富 $W_t$ 遵循 Kesten 方程：

$$W_{t+1} = A_t \cdot W_t + B_t$$

其中 $A_t$ 为随机乘法冲击因子（服从冲击概率 $p$），$B_t$ 为加法收入项（服从耗散率 $c$ 调节）。关键比值 $R^* = p/c$ 是系统的主控参数。**固定量**：$N=100,000$，$T=500$，5 随机种子，评估指标为 $\beta_{\text{left}}$、$R^2$、贫困率。

**探索参数空间**：
- 连续参数：冲击概率 $p \in [0.001, 0.1]$，耗散率 $c \in [0.01, 0.3]$
- 政策参数：累进税率 $\tau \in [0, 0.20]$，最低生活补贴 $S \in [0, 0.05]$
- 离散机制：边界条件（`reflect` / `absorb` / `soft_clamp`），收入分布（`lognormal` / `exponential` / `uniform`）

### 主要发现

> 🔴 **最意外的发现（跑代码之前完全没有预期到）**：  
> 将边界条件从 `reflect`（反射）切换至 `absorb`（吸收，财富触底归零）时，
> 幂律拟合优度 $R^2$ 从 $> 0.85$ 骤降至 $< 0.50$——幂律分布整体失效。
> 这不是参数变化的连续响应，而是系统在结构层面的相变坍塌。
> 它表明**边界条件不只是数值细节，而是决定系统处于哪个相的关键结构参数**。

- **政策失效存在硬边界（有价值的负结果）**：在高冲击风险 $p > 0.04$ 的区域，即使施加 $\tau = 0.15$ 的累进税，贫困率仍然无法遏制甚至反弹。纯再分配政策的有效性上限由冲击频率决定，而非政策力度。
- **相变临界区 $R^* \approx 0.15\text{–}0.20$**：控因 1D 扫描（固定 $c=0.10$, `reflect`, `lognormal`）中，CUSUM 检测到 $\beta_{\text{left}}$ 的非连续跳变，10-种子独立重复验证可复现性（$\ge 8/10$）。
- **中美参数结构的定性差异**：世界银行数据校准后，中国呈现高耗散率 $c$（强消费/税收）、低冲击风险 $p$，美国相反——与两国体制结构定性吻合，验证了模型的实证有效性。
- **主动学习智能体**（UCB + GP Surrogate）：等预算下与随机搜索对比，通过 Welch's t-test 检验探索效率差异（当前 $p \approx 0.28$，为下一步重点强化方向）。

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
