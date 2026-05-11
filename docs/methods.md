# 精排方法说明

> 创建时间：2026-05-09

本文档描述在线仿真实验中使用的 14 种精排方法的实现原理与配置参数。所有方法共享相同的在线仿真框架（召回 → 精排 → 用户反馈 → 图更新），仅精排模块不同。

---

## 公共配置

以下参数对所有方法固定，不随方法变化：

```yaml
recall:
  method: two_hop_bidir_random   # 双向二跳邻居随机采样召回
  top_k_recall: 100              # 每用户召回候选数

recommend:
  top_k: 10                      # 精排后向用户推荐的边数
  cold_start_random_fill: true   # 冷启动（无候选）时随机补齐

feedback:
  p_pos: 1.0                     # 用户接受推荐边的概率
  p_neg: 0.0                     # 用户拒绝后仍接受的概率
  cooldown_mode: decay           # 冷却机制：指数衰减
  cooldown_rounds: 5             # 冷却基准轮数

trainer:                         # 仅对可训练模型生效
  lr: 5.0e-05
  scheduler: constant            # 恒定学习率，不衰减
  grad_clip: 1.0
  update_every_n_rounds: 1       # 每轮更新一次模型
  min_batch_size: 4
  max_neighbors: 30              # 子图提取时邻居截断数
  batch_subgraph_max_hop: 2      # 子图最大跳数

user_selector:
  strategy: uniform              # 每轮均匀随机抽取活跃用户
  sample_ratio: 0.1              # 每轮活跃用户比例

total_rounds: 30
seed: [0, 1, 2, 3, 42]          # 5 个随机种子
```

---

## 一、启发式方法（无训练）

启发式方法不依赖模型参数，直接用当前观测图 $G_t$ 中的拓扑结构对候选边评分。邻居定义均采用**无向邻居**（出边邻居 ∪ 入边邻居），以保证有向图上的对称性。

### 1. random

**原理**：对召回候选集随机均匀打分，作为无信息基线。

**配置**：
```yaml
model:
  type: random
```

**实现**：`src/recall/heuristic.py` — 每个候选节点 $v$ 分配 $\text{Uniform}(0,1)$ 随机分，不访问图结构。

---

### 2. cn（Common Neighbors）

**原理**：共同邻居数量，衡量 $u$ 和 $v$ 在局部结构上的相似度。

$$\text{score}(u, v) = |N(u) \cap N(v)|$$

其中 $N(\cdot)$ 为无向邻居集合。

**配置**：
```yaml
model:
  type: cn
```

**实现**：`src/recall/heuristic.py:CommonNeighborsRecall` / `src/baseline/heuristic.py:score_cn`
- 大图（$n > 10^4$）：稀疏矩阵乘法 $A_\text{sym}[U] \cdot A_\text{sym}$ 批量计算，避免逐对 set intersection
- 小图：直接 set 交集

---

### 3. aa（Adamic-Adar）

**原理**：共同邻居计数的加权变体，对高度数中间节点降权，避免 Hub 节点支配评分。

$$\text{score}(u, v) = \sum_{z \in N(u) \cap N(v)} \frac{1}{\log(|N(z)| + 2)}$$

**配置**：
```yaml
model:
  type: aa
```

**实现**：`src/recall/heuristic.py:AdamicAdarRecall` / `src/baseline/heuristic.py:score_aa`
- 权重计算：`1 / log(deg(z) + 2)`，分母加 2 防止度为 0/1 时数值不稳定

---

### 4. jaccard（Jaccard 系数）

**原理**：共同邻居数占两者邻居并集的比例，对邻居集合大小归一化。

$$\text{score}(u, v) = \frac{|N(u) \cap N(v)|}{|N(u) \cup N(v)|}$$

两者均无邻居时得分为 0。

**配置**：
```yaml
model:
  type: jaccard
```

**实现**：`src/baseline/heuristic.py:score_jaccard`

---

### 5. pa（Preferential Attachment）

**原理**：优先依附，评分等于两节点度数之积，模拟"富者愈富"现象。

$$\text{score}(u, v) = |N(u)| \cdot |N(v)|$$

无需计算邻居交集，计算量最低。

**配置**：
```yaml
model:
  type: pa
```

**实现**：`src/baseline/heuristic.py:score_pa`

---

## 二、GIN 子图精排模型

基于 **GIN（Graph Isomorphism Network）** 的子图链接预测框架。对每对候选边 $(u, v)$，提取二度邻居子图 $S_{uv}$，在子图上运行 GNN 得到图级表示，再通过 MLP Scorer 输出预测分数。

**子图特征**：每个节点赋 2 维 one-hot 特征：$u \to [1,0]$，$v \to [0,1]$，其他节点 $\to [0,0]$。不使用节点属性（`node_feat_dim: 0`）。

**GIN 层**：每层使用 `GINConv(sum 聚合) + Tanh` 激活，MLP 为 `Linear → Tanh → Linear`。

**Scorer**：`Linear(in_dim, hidden_dim) → ReLU → Linear(hidden_dim, 1) → Sigmoid`，输出 $(0,1)$ 分数。

六种 GNN 变体的区别仅在于 `hidden_dim` 和 `encoder_type`（图级 readout 方式）：

| 变体 | `hidden_dim` | `encoder_type` | Scorer 输入维度 |
|------|-------------|----------------|----------------|
| gnn | 8 | last | 8 |
| gnn_h32 | 32 | last | 32 |
| gnn_concat | 32 | layer_concat | $3 \times 3 \times 32 = 288$ |
| gnn_concat_h8 | 8 | layer_concat | $3 \times 3 \times 8 = 72$ |
| gnn_sum | 32 | layer_sum | $3 \times 32 = 96$ |
| gnn_sum_h8 | 8 | layer_sum | $3 \times 8 = 24$ |

### encoder_type 说明

- **`last`**：取最后一层所有节点表示做全图 mean pooling，输出 $(H,)$。结构最简单，不区分 $u$/$v$ 角色。

- **`layer_concat`**：每层分别对 $u$ 节点、$v$ 节点、其他节点各做 mean pooling 得到三段 $(H,)$，拼接后再拼接所有层，输出 $(L \times 3H,)$。保留每层的层次结构信息，参数量最大。

- **`layer_sum`**：与 `layer_concat` 相同的每层三段拼接 $(3H,)$，但对所有层的结果相加（而非 concat），输出 $(3H,)$。在保留 $u$/$v$ 角色区分的同时压缩表示维度。

### 6. gnn

```yaml
model:
  type: gnn
  hidden_dim: 8
  num_layers: 3
  encoder_type: last
  node_feat_dim: 0
```

### 7. gnn_h32

与 `gnn` 相同，仅 `hidden_dim` 提升至 32：

```yaml
model:
  type: gnn
  hidden_dim: 32
  num_layers: 3
  encoder_type: last
  node_feat_dim: 0
```

### 8. gnn_concat

```yaml
model:
  type: gnn
  hidden_dim: 32
  num_layers: 3
  encoder_type: layer_concat
  node_feat_dim: 0
```

### 9. gnn_concat_h8

```yaml
model:
  type: gnn
  hidden_dim: 8
  num_layers: 3
  encoder_type: layer_concat
  node_feat_dim: 0
```

### 10. gnn_sum

```yaml
model:
  type: gnn
  hidden_dim: 32
  num_layers: 3
  encoder_type: layer_sum
  node_feat_dim: 0
```

### 11. gnn_sum_h8

```yaml
model:
  type: gnn
  hidden_dim: 8
  num_layers: 3
  encoder_type: layer_sum
  node_feat_dim: 0
```

---

## 三、节点嵌入 + 全图 GNN 模型

与子图 GIN 不同，这两种方法为**每个节点**维护一个全局可学习嵌入（`nn.Embedding`），以全局节点 ID 查表获取初始特征，再通过图卷积聚合邻域信息。子图提取后，节点特征不再是 one-hot，而是来自嵌入表。

### 12. graphsage_emb

**原理**：GraphSAGE（mean 聚合）+ 可学习节点嵌入。节点初始特征来自 `nn.Embedding(n_nodes, emb_dim)`，经 $L$ 层 `SAGEConv(mean)` 聚合，全图 mean pooling 后输出图级表示，送入 Scorer。

**架构**：
```
Embedding(n_nodes, 32) → SAGEConv×3(mean, hidden=32) → mean_pool → Scorer
```

**配置**：
```yaml
model:
  type: graphsage_emb
  emb_dim: 32        # 节点嵌入维度
  hidden_dim: 32     # SAGEConv 隐层维度
  num_layers: 3
```

**实现**：`src/baseline/graphsage_emb.py:GraphSAGEEmbModel`

---

### 13. gat_emb

**原理**：GAT（图注意力网络）+ 可学习节点嵌入。节点初始特征来自嵌入表，通过多头 `GATConv` 聚合，全图 mean pooling 后送 Scorer。注意力机制使模型能对不同邻居动态赋权。

**架构**：
```
Embedding(n_nodes, 32) → GATConv×3(4 heads, head_dim=8, hidden=32) → mean_pool → Scorer
```

**约束**：`hidden_dim` 须能被 `num_heads` 整除（`head_dim = hidden_dim // num_heads = 8`）。

**配置**：
```yaml
model:
  type: gat_emb
  emb_dim: 32        # 节点嵌入维度
  hidden_dim: 32     # GATConv 总输出维度（= num_heads × head_dim）
  num_layers: 3
  num_heads: 4       # 注意力头数
```

**实现**：`src/baseline/gat_emb.py:GATEmbModel`

---

## 四、SEAL

**原理**：Zhang & Chen（2018）提出的子图链接预测方法，与 GIN 子图框架的核心区别在于节点特征：使用 **DRNL（Double-Radius Node Labeling）** 代替 one-hot。

DRNL 为子图中每个节点赋予一个离散结构标签，标签由该节点到 $u$、$v$ 的最短路径距离 $(d_u, d_v)$ 决定：

$$\ell(x) = 1 + \min(d_u, d_v) + \left\lfloor \frac{d_u - d_v}{2} \right\rfloor \cdot \left(\left\lfloor \frac{d_u - d_v}{2} \right\rfloor + (d_u - d_v) \bmod 2\right)$$

不可达节点标签为 0。标签经 `nn.Embedding(max_label+1, label_dim)` 嵌入为向量，作为节点初始特征输入 GIN。

**与原文差异**：Readout 使用 mean pooling（原文用 SortPooling + DGCNN）；BFS 使用 numpy 稠密矩阵乘法加速（适配小子图 $n \leq 50$）。

**架构**：
```
DRNL标签 → Embedding(max_label+1, 16) → GINConv×3(hidden=32) → mean_pool → Scorer
```

**配置**：
```yaml
model:
  type: seal
  hidden_dim: 32    # GIN 隐层维度
  num_layers: 3
  label_dim: 16     # DRNL 标签嵌入维度
```

**实现**：`src/baseline/seal.py:SEALModel`

---

## 方法对比一览

| 方法 | 类型 | 是否训练 | 节点特征 | 图级表示 | 参数量估计 |
|------|------|---------|---------|---------|-----------|
| random | 启发式 | 否 | — | — | 0 |
| cn | 启发式 | 否 | — | — | 0 |
| aa | 启发式 | 否 | — | — | 0 |
| jaccard | 启发式 | 否 | — | — | 0 |
| pa | 启发式 | 否 | — | — | 0 |
| gnn | GIN(last, h8) | 是 | 2-dim one-hot | mean pool(last) | ~300 |
| gnn_h32 | GIN(last, h32) | 是 | 2-dim one-hot | mean pool(last) | ~4K |
| gnn_concat | GIN(concat, h32) | 是 | 2-dim one-hot | layerwise concat | ~16K |
| gnn_concat_h8 | GIN(concat, h8) | 是 | 2-dim one-hot | layerwise concat | ~1K |
| gnn_sum | GIN(sum, h32) | 是 | 2-dim one-hot | layerwise sum | ~7K |
| gnn_sum_h8 | GIN(sum, h8) | 是 | 2-dim one-hot | layerwise sum | ~500 |
| graphsage_emb | GraphSAGE+emb | 是 | 全局嵌入 | mean pool(last) | ~$32n$+4K |
| gat_emb | GAT+emb | 是 | 全局嵌入 | mean pool(last) | ~$32n$+5K |
| seal | SEAL+GIN | 是 | DRNL 嵌入 | mean pool(last) | ~5K |

> graphsage_emb / gat_emb 的参数量随数据集节点数 $n$ 线性增长（嵌入表 $32n$ 参数）。
