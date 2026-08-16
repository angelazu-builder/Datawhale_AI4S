# 📊 分析图表与实验数据索引 (Outputs Index)

本目录包含由统一命令行 `python3 main.py --mode full` 自动生成的全套高清分析图表与科学日志。

---

## 📈 高清分析图表 (Visualizations)

| 图表文件名 | 对应研究内容与科学发现 |
|------------|------------------------|
| `phase_transition_R_star.png` | **相变临界曲线**：控因 1D 参数扫描下，CUSUM 检测到 $\beta_{\text{left}}$ 在 $R^* \approx 0.15$–$0.20$ 的非连续跳变。 |
| `behavioral_mode_comparison.png` | **行为模式对比**：四面板对比图，显示凯恩斯 MPC 使得 $\beta_{\text{left}} \to 0$（**幂律完全崩溃**）。 |
| `baseline_comparison.png` | **主动学习 vs 随机搜索**：UCB + GP 代理模型与等预算随机搜索在 30 轮探索中的累计收益对比。 |
| `agent_trajectory.png` | **智能体探索轨迹**：Active Learning Agent 在高维参数空间中的采样轨迹分布。 |
| `left_tail_fit.png` | **左尾模型选择**：Power-Law vs. Lognormal vs. Exponential 候选分布的 AIC/BIC 比对。 |
| `us_china_historical_comparison.png` | **中美历史校准**：1980–2023 中美基尼系数与底层 20% 财富占比演化对比。 |
| `policy_counterfactual_simulation.png` | **反事实政策推演**：精准扶贫补贴（$S>0$）与累进税（$\tau>0$）的反事实推演效应对比。 |

---

## 📁 实验日志与科学数据 (Data & Artifacts)

| 数据文件名 | 内容描述 |
|------------|----------|
| `scientific_signals.json` | 自动分类的科学信号（D1 正向相变发现、D2 结构坍塌异常、D3 政策失效负结果）。 |
| `kesten_exploration_log.csv` | 60 轮主动学习与随机探索的逐轮原始数据（含 $p, c, \text{boundary}, \beta, \text{mutation\_flag}$）。 |
| `kesten_adaptive_history.json` | 智能体高斯过程代理模型（GP Surrogate）内部更新历史。 |
| `empirical_us_china_data.csv` | 来自世界银行 API (SI.POV.GINI) 的中美宏观基尼系数历史序列。 |
