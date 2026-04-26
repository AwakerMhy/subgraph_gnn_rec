# 在线推荐链路优化建议 — 凸显 GNN 性能（论文向）

> 创建时间：2026-04-26
> 最后更新：2026-04-26
>
> **目的**：在保留 2026-04-25「two_hop_random 中性召回」公平性原则的前提下，重新设计在线仿真链路与评估指标，使 GNN 相对 MLP/Random 的优势在论文中可见、可信、可复现。
>
> **范围**：涉及 `src/online/`、`src/recall/`、`src/online/evaluator.py`、`scripts/visualize_online_run.py`、`configs/online/*.yaml`，不涉及 legacy 协议。

---

## 0. 现状诊断

| # | 问题 | 影响 |
|---|---|---|
| 1 | `two_hop_random(top_k=100)` 把候选限定在 2-hop 内随机采样，所有 ranker 看到的候选局部结构高度相似 | GNN 高阶判别力被截掉，ranker 间差距被压扁 |
| 2 | `recall_rejected` 作为负样本，但 p_pos<1 意味着部分真正样本被随机拒绝而错标为负例 | 等比摊薄所有 ranker 的区分度，GNN 优势不能充分体现（**→ 已通过提高 p_pos=0.95 缓解**） |
| 3 | 评估指标全是单轮快照（prec@K / coverage / MRR@K） | 不反映 GNN 在轨迹上的优势（更快收敛、更稳定） |
| 4 | 论文叙事窄：主线"online link prediction"与经典 baseline 正面冲突 | 项目独有的"反馈驱动网络演化"命题没在指标里体现 |

**本轮已完成的全局调参**：
- `p_pos: 0.8 → 0.95`（降低真实关系被拒的噪声率）
- `p_neg: 0.02 → 0.0`（用户不再接受非真实关系的推荐，去除虚假正样本写入 G_t）
- 影响：159 个 `configs/online/*.yaml` 统一更新；该变更覆盖 DECISIONS.md `[2026-04-21]` 的旧参数

---

## 1. 召回设计：高覆盖 + 弱排序

**目标**：扩大候选池，让 ranker 真正承担打分任务，而非在局部结构高度相似的小池子里做无意义竞争。

| 字段 | 现状 | 建议 |
|---|---|---|
| `top_k_recall` | 100 | 200~300 |
| 召回成分 | `two_hop_random` 单源 | `two_hop_random(α=0.7) + ppr_nodes(α=0.3)`（PPR 仅取节点，不打分，扩到 3-hop） |
| `top_k`（精排） | 5 | 5（不变） |

**实现位置**：`src/recall/registry.py` 新增 `ppr_nodes` 类型（`PPRRecall` 子类，candidates 返回 `(v, 0.0)` 抹掉分数）；在 mixture 配额里挂载。

**前置验证**（再开大规模实验前做）：在 college_msg 上跑 `top_k_recall ∈ {100, 200, 300}` 三档，画 GNN-MLP gap vs top_k_recall 曲线，确认 gap 单调上升。

**风险**：候选池扩大后，MLP 也可能从更多正样本中受益。若 gap 不单调上升，回退到 top_k_recall=100，仅靠 §3 指标差异化。

---

## 2. 训练信号：Hard Negative Mining

**目标**：在 `recall_rejected` 中提取最有价值的负样本，放大 GNN 的判别优势区。

**方案**：在 `recall_rejected` 里只保留 ranker 给分排名前 30%（"模型觉得好但被用户拒"的样本）——这是 GNN 擅长、MLP 反而劣势的判别区。

**实现位置**：`src/online/trainer.py::update` 入参增加 `neg_scores: list[float]`，函数内按分数降序取前 30%；`src/online/loop.py` 在构造 `recall_rejected` 时附带对应分数。

**降级方案**：从前 50% 开始，若 gradient norm 稳定后再缩到 30%。

**风险**：过强的 hard negative 可能导致训练不稳定；MLP 在不取 hard negative 时是更公平的基准（可做带/不带 hard negative 的消融对比）。

---

## 3. 评估指标：从快照换成过程性指标

**改动**：扩展 `src/online/evaluator.py::RoundMetrics`，论文图表以**轨迹**和**收敛特性**为主。

### 3.1 Convergence Speed（必加）
- 记录达到 `coverage ∈ {25%, 50%, 75%}` 所需轮数
- 论文表格里一行，GNN 应全部占优
- **实现**：`evaluator.py` 新增 `coverage_milestones: dict[float, int]`

### 3.2 网络结构追真度（项目独有核心叙事）
- 每 `graph_every_n=10` 轮记录 G_t vs G\* 的：
  - degree distribution KL divergence（已有，确认写入 csv）
  - clustering coefficient（in/out/avg）
  - 3-motif count（feed-forward / cycle / mutual）
  - reciprocity ratio
- 主实验图：x=round，y=结构距离，三条 ranker 线 + G\* 参考线
- **实现位置**：`evaluator.py` 新增 `_compute_structural_metrics(G_t, G_star)`

### 3.3 用户分层 MRR（反驳 reviewer 偏置质疑）
- 按 G\* 中度数 quintile 分组（0-20%、20-40%、…、80-100%）
- 每组单独计算 MRR / Hits@K
- **预期**：GNN 在低度（冷门）用户上优势最大，MLP 度数偏置在此失效
- **实现位置**：`evaluator.py::update` 按 `g_star_degree[u]` 分桶聚合

### 3.4 Cumulative Regret
- 累积 `Σ_t (oracle_acceptance_t - actual_acceptance_t)`
- oracle = 若推荐池中存在 G\* 边则全部接受
- **实现位置**：`evaluator.py` 新增 `cumulative_regret` 字段

---

## 4. 模型表征：学习 ID Embedding + Edge Dropout

**注意**：hidden_dim 本轮不调整，保持各数据集现有配置。

### 4.1 学习的节点 ID Embedding（GNN 专属优势）
- 节点特征：`[in_deg, out_deg, total_deg]`（3-dim）→ 增加 `nn.Embedding(n_nodes, 16)` 可训练
- MLP/Random 仍用度数占位（这是 GNN 框架的天然优势，非不公平）
- 开关 `use_id_embedding: true/false`，方便消融
- **实现位置**：`src/model/model.py::LinkPredModel.__init__`；`src/online/trainer.py` forward 时拼接

### 4.2 Edge Dropout 正则化
- 在线图持续扩张，过拟合最近边的风险随轮次增加
- 在 GIN 每层 forward 前对边做随机 mask，`drop_rate=0.1`
- **实现位置**：`src/model/gin_encoder.py::forward`（利用 `dgl.DropEdge`）

---

## 5. 论文叙事重构：从"Online Link Prediction"到"Feedback-Driven Network Formation"

**核心改框**：把命题从"GNN 在 online link prediction 任务上更准"改为"GNN 作为推荐器更能驱动网络结构向真实形态演化"。

### 5.1 主实验图
- x=轮次，y=网络属性距离（KL/clustering），三条 ranker 线 + G\* 参考线
- 多数据集 grid，每行一个数据集，直观展示 GNN 收敛更快更准

### 5.2 Motif 恢复对比
- 同一份初始 E_obs，三种 ranker 各跑 100 轮
- 比较终态 G_t 的 3-motif 分布与 G\* 的 L1/L2 距离
- 雷达图展示

### 5.3 噪声鲁棒性消融
- 对照实验：`p_pos ∈ {0.95, 0.8, 0.5}`（p_neg 固定 0.0）
- 横轴噪声水平（用户犹豫度），纵轴 GNN-MLP gap
- **预期**：噪声越大，GNN 优势越显著（结构特征比度数特征更鲁棒）

---

## 实施优先级

| 阶段 | 内容 | 代码风险 | 何时做 |
|---|---|---|---|
| **A**（立刻） | §3.1 + §3.3 + §3.2 原型（evaluator 扩展 + 2 个数据集小跑） | 低 | 现在 |
| **B**（叙事确认后） | §3.2 全数据集 + §5.1 主实验图 | 低 | A 验证后 |
| **C**（B 完成后） | §1 召回扩容 + §2 Hard Negative Mining | 中 | 全数据集重跑 |
| **D**（最终 paper-ready） | §4 ID Embedding + Edge Dropout | 低 | C 完成后 |

**关键决策点（A 阶段后）**：

若 §3.2 原型验证通过（GNN 在结构追真度上显著赢 MLP/Random）：
→ 全力推进 B→C→D，走 §5 叙事

若 §3.2 原型失败：
→ 放弃 §5 叙事重构，论文回退到传统 online link prediction 框架，继续推进 §3.1/§3.3/§2/§1

**禁止做**：
- A 阶段同时改训练逻辑（§1/§2），混合改动让指标变化无法归因
- §1（召回扩容）和 §2（hard negative）合并同一次实验
- §4 与 §2 合并实验（ID embedding 对 MRR 的贡献应单独量化）

---

## 与现有决策的关系

| 决策 | 状态 | 说明 |
|---|---|---|
| `[2026-04-25] two_hop_random 中性召回` | active，§1 在其上扩配额，PPR 不打分保留中性原则 | 不推翻 |
| `[2026-04-21] p_pos=0.8 / p_neg=0.02` | **superseded**，新值 p_pos=0.95 / p_neg=0.0 | 需更新 DECISIONS.md |
| `[2026-04-21] Composite 用户选择` | active，§3.3 用户分层基于 G\* 度数与 selector 策略正交 | 不推翻 |
| `[2026-04-22] ego-graph + 公共邻居子图` | active，§4.2 edge dropout 若显著改善可能触发重评 | 待观察 |

---

## 验收标准（A 阶段）

以下任一条不达标 → 进入叙事回退决策点：

1. **Convergence speed**：GNN 在 ≥2 个数据集上达到 50% coverage 的轮数比 MLP 少 ≥20%
2. **用户分层 MRR**：GNN 在最低度 quintile 上的 MRR 比 MLP 高 ≥30%
3. **结构追真度**：在 college_msg 或 bitcoin_alpha 上，degree KL 或 clustering trajectory，GNN 终值视觉可见低于 MLP/Random
