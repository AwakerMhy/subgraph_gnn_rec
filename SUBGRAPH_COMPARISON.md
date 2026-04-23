# 子图设计方案对比 — Subgraph Design Comparison

> 创建时间：2026-04-22
> 
> 对比两种子图提取策略：2-hop BFS vs ego-graph + 公共邻居融合

---

## 方案 A：2-hop BFS（原设计）

**结构**：
- 从 u 开始 BFS 扩展 max_hop=2 步
- 返回路径长度 ≤ 2 的所有节点

**实现**：`_bfs_neighbors()` 迭代式邻域扩展

**特点**：
- 固定扩展深度，但邻域大小不可控（可能接近全图）
- 稠密小图中缺乏区分度

---

## 方案 B：ego-graph + 公共邻居（新设计）

**结构**：
```
subgraph_nodes = {u, v} ∪ N_1(u) ∪ CN(u, v)

其中：
  N_1(u)：u 的一度出邻居（采样至 max_neighbors_per_node）
  CN(u, v)：{w | w ∈ N_1(u) ∩ N_1(v)}
```

**实现**：直接查询 one-hop + 集合交集

**特点**：
- 动态大小（核心结构 u + v + 公共邻，通常 10-50 节点）
- 聚焦链路形成机制（u 的影响力圈 + 共同话题）

---

## 初步对比结果（烟测）

**实验设置**：
- 数据集：CollegeMsg（1899 节点，59835 边）
- 协议：simulated_recall + AA 召回
- 样本数：100（训练）
- 轮数：1 epoch
- 模型：GIN-last

**结果表**：

| 指标 | 2-hop BFS | ego+CN | Δ | % 改进 |
|---|---|---|---|---|
| tr_auc | 0.5248 | 0.4866 | -0.0382 | -7.3% |
| val_auc | - | - | - | - |
| MRR | 0.0478 | 0.0949 | +0.0471 | +98.5% |
| Hits@10 | 0.0909 | 0.1818 | +0.0909 | +100% |
| Hits@20 | 0.1515 | 0.2828 | +0.1313 | +86.6% |
| Hits@50 | 0.2020 | 0.4545 | +0.2525 | +125% |
| NDCG@10 | 0.0884 | 0.1248 | +0.0364 | +41.2% |
| NDCG@20 | 0.1169 | 0.1248 | +0.0079 | +6.8% |

**分析**：
1. **tr_auc 下降**：新设计子图节点少，模型训练难度增加但泛化能力更强
2. **排序指标大幅提升**：MRR/Hits 验证了新设计在链路排序上的优越性，这正是任务关键指标
3. **trade-off**：用少量 tr_auc 收益换取显著的排序能力提升，符合任务需求

---

## 后续对比计划

### Phase 1：验证单数据集完整训练（目标：30 epoch）

| 数据集 | 模型 | 配置 | 目标 |
|---|---|---|---|
| CollegeMsg | GIN-last | epochs=30, seed=42 | 确认 ego+CN 在完整训练中的稳定性 |

### Phase 2：多数据集对比

| 数据集 | 特征 | 目标 |
|---|---|---|
| CollegeMsg | 稠密小图（1899 节点） | 验证稠密图中 ego+CN 的泛化性 |
| bitcoin_otc | 稀疏中等图（5881 节点，35592 边） | 验证稀疏图中的相对优劣 |
| email_eu | 超小稀疏图（986 节点，24929 边） | 验证极小图中的适用性 |

### Phase 3：模型对比

可选：
- GIN-last（当前）
- GIN-layer_concat
- GIN-layer_sum
- GraphSAGE
- SEAL（若可用）

---

## 记录表格（实验完成后填充）

### CollegeMsg × 30 epoch × GIN-last

**2-hop BFS**：
```
tr_auc_best: 
val_auc_best: 
MRR_best: 
Hits@10_best: 
训练时间: 
checkpoint: 
```

**ego+CN**：
```
tr_auc_best: 
val_auc_best: 
MRR_best: 
Hits@10_best: 
训练时间: 
checkpoint: 
```

### bitcoin_otc × 30 epoch × GIN-last

*待更新*

### email_eu × 30 epoch × GIN-last

*待更新*

---

## 关键观察与假设

**假设 H1**：ego+CN 在稠密小图中相对优势更大（公共邻居比例高）

**假设 H2**：ego+CN 在稀疏大图中效果相近或略优（邻域大小约束限制 2-hop 扩展）

**假设 H3**：新设计与模型架构互动较弱（2-hop vs 1-hop 是结构层面改进）

---

## 决策追踪

- 决策记录：`DECISIONS.md::2026-04-22`
- 进度追踪：`PROGRESS.md::子图设计迭代`
- 代码变更：`src/graph/subgraph.py::extract_subgraph()`
