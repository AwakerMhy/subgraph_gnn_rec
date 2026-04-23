# Temporal Social Link Prediction via 2-Hop Subgraph GNN

> **项目目标**：研究基于二度邻居子图的GNN链接预测方法，面向同质有向社交图，预测候选节点在未来一段时间内向推荐节点建立有向边的概率。边的时间戳仅用于数据集划分，不作为模型输入特征。

---

## 目录

1. [问题定义](#1-问题定义)
2. [项目结构](#2-项目结构)
3. [数据模块](#3-数据模块)
4. [模型架构](#4-模型架构)
5. [训练与评估](#5-训练与评估)
6. [实验计划](#6-实验计划)
7. [实现顺序与任务清单](#7-实现顺序与任务清单)
8. [技术栈与约束](#8-技术栈与约束)

---

## 1. 问题定义

### 1.1 输入输出

- **图类型**：同质有向图，节点为用户，边 $(u, v, t)$ 表示用户 $u$ 在时刻 $t$ 向用户 $v$ 建立有向连接
- **查询**：给定候选节点 $u$、推荐节点 $v$、查询时刻 $t_q$
- **截断图**：$\mathcal{G}_{t_q}$ 仅包含 $t < t_q$ 的历史边（严格防止时间泄露）
- **子图**：提取 $u$ 和 $v$ 各自在 $\mathcal{G}_{t_q}$ 中的二度邻居的并集所诱导的子图 $\mathcal{S}_{uv}$
- **输出**：标量评分 $s_{uv} \in (0,1)$，越高表示 $u$ 在 $[t_q,\ t_q+\Delta t]$ 内向 $v$ 建边的概率越大

### 1.2 关键约束

- 边的时间戳**仅用于**：数据集时间切分、截断图构建、防止数据泄露
- 边的时间戳**不用于**：模型输入特征、边嵌入计算
- 子图提取时保留边的方向性
- 节点属性统一接口处理：真实数据集无原生属性时，以**度特征**（入度、出度、总度，共3维）作为占位属性；合成数据集生成器主动产生属性向量。`use_node_attr` 开关控制属性编码器（`encoder_attr`）是否激活，但节点特征向量始终存在（至少为度特征），用于拓扑编码器的初始特征拼接

---

## 2. 项目结构

```
project/
├── PLAN.md                     # 本文件
├── README.md                   # 简要说明与快速开始
├── requirements.txt
├── configs/
│   ├── default.yaml            # 默认超参数
│   ├── dataset/
│   │   ├── college_msg.yaml
│   │   ├── bitcoin_otc.yaml
│   │   ├── email_eu.yaml
│   │   ├── synth_sbm.yaml
│   │   ├── synth_hawkes.yaml
│   │   └── synth_triadic.yaml
│   └── model/
│       ├── ours_full.yaml
│       ├── ours_no_attr.yaml   # 消融：去掉属性
│       ├── ours_1hop.yaml      # 消融：改为1度子图
│       └── baselines/
│           ├── seal.yaml
│           ├── graphsage.yaml
│           └── tgat.yaml
├── data/
│   ├── raw/                    # 原始下载文件（不进git）
│   ├── processed/              # 预处理后的标准格式
│   └── synthetic/              # 合成数据集输出
├── src/
│   ├── dataset/
│   │   ├── __init__.py
│   │   ├── base.py             # 统一数据集基类
│   │   ├── real/
│   │   │   ├── college_msg.py
│   │   │   ├── bitcoin_otc.py
│   │   │   └── email_eu.py
│   │   └── synthetic/
│   │       ├── generator_base.py
│   │       ├── sbm.py
│   │       ├── hawkes.py
│   │       └── triadic.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── subgraph.py         # 子图提取核心逻辑
│   │   ├── labeling.py         # DRNL节点标记
│   │   └── negative_sampling.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── encoder_topo.py     # 子图拓扑编码器（GNN）
│   │   ├── encoder_attr.py     # 节点属性编码器（MLP）
│   │   ├── fusion.py           # 融合模块
│   │   ├── scorer.py           # 评分头
│   │   └── model.py            # 顶层模型组装
│   ├── baseline/
│   │   ├── heuristic.py        # CN, AA, Jaccard, Katz
│   │   ├── seal.py
│   │   ├── graphsage.py
│   │   └── tgat.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils/
│       ├── metrics.py          # AUC, AP, Hits@K
│       ├── split.py            # 时间感知数据划分
│       ├── logger.py
│       └── seed.py
├── experiments/
│   ├── run_main.py             # 主实验入口
│   ├── run_ablation.py         # 消融实验入口
│   ├── run_negative.py         # 负样本策略对比
│   └── run_synthetic.py        # 合成数据集专项实验
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── attention_visualization.ipynb
│   └── case_study.ipynb
└── results/
    ├── logs/
    ├── checkpoints/
    └── tables/
```

---

## 3. 数据模块

### 3.1 统一数据格式

所有数据集处理后输出标准格式，存为 `processed/<dataset_name>/`:

```python
# edges.csv: 有向边列表（按时间排序）
# columns: src, dst, timestamp
# 时间戳统一归一化到 [0, 1] 的浮点数，原始值保留在 timestamp_raw

# nodes.csv: 节点特征
# columns: node_id, feat_0, feat_1, ..., feat_d
# - 真实数据集（无原生属性）：feat = [in_degree, out_degree, total_degree]，共3维
#   degree 基于训练集截止时刻的图计算，测试/验证时不更新
# - 合成数据集：feat = 生成器产生的属性向量（维度由生成器配置决定）
# 所有数据集的 feat 列均不为空，use_node_attr 开关只控制 encoder_attr 是否激活

# meta.json: 数据集元信息
{
  "n_nodes": int,
  "n_edges": int,
  "has_native_node_feature": bool,   # true=合成数据集, false=真实数据集用度特征占位
  "feat_dim": int,
  "t_min": float,
  "t_max": float,
  "is_directed": true
}
```

**节点属性的两条路径**：

```
节点特征 x (始终存在)
    ├── → [x || label_embed] → TopoEncoder 初始特征   (始终走这条路)
    └── → AttrEncoder(MLP)  → h_u^attr, h_v^attr     (仅当 use_node_attr=true 时)
```

`use_node_attr=false` 时：`AttrEncoder` 不实例化，融合向量中不含 `h^attr` 项，其余不变。

### 3.2 真实数据集

| 数据集 | 下载来源 | 预处理要点 |
|--------|----------|------------|
| CollegeMsg | SNAP | 无节点属性，使用度特征（入度、出度、总度） |
| Bitcoin-OTC | SNAP | 无节点属性，边权重丢弃（本项目不用），同上 |
| Email-EU-core-temporal | SNAP | 无节点属性，高密度，注意重复边处理（保留最早时间戳） |

预处理步骤（每个数据集实现 `process()` 方法）：
1. 读取原始文件，统一列名
2. 去除自环（src == dst）
3. 按时间戳升序排序
4. 节点ID重新映射为连续整数
5. 若无节点属性，计算截止训练集末尾的度特征
6. 输出标准格式

### 3.3 合成数据集生成器

三种生成器共享 `GeneratorBase` 接口：

```python
class GeneratorBase:
    def generate(self) -> pd.DataFrame:
        """返回 (src, dst, timestamp) 格式的边列表"""
        raise NotImplementedError

    def get_node_features(self) -> np.ndarray:
        """返回 (n_nodes, feat_dim) 的节点属性矩阵"""
        raise NotImplementedError
```

**SBM生成器（`sbm.py`）**
- 参数：`n_nodes`, `n_communities`, `p_in`, `p_out`, `T`, `edges_per_step`, `seed`
- 规则：同社区建边概率 `p_in` >> 跨社区 `p_out`，共同邻居加成，活跃度加权
- 节点属性：社区one-hot + 随机噪声向量

**Hawkes生成器（`hawkes.py`）**
- 参数：`n_nodes`, `mu`, `alpha`, `beta`, `T`, `seed`
- 规则：基于多元Hawkes过程，历史边对未来边产生时序激励
- 节点属性：随机正态向量

**Triadic生成器（`triadic.py`）**
- 参数：`n_nodes`, `base_p`, `triadic_bonus`, `T`, `seed`
- 规则：三角闭合为主导规则，若 A→B、B→C，则 A→C 以高概率在短时间内出现
- 节点属性：随机正态向量

### 3.4 时间感知数据划分

```
train | val | test
 70%  | 15% | 15%   （按时间戳分位数切分，严禁随机划分）
```

负样本生成（每个正样本对应若干负样本）：

| 策略名 | 描述 | 用途 |
|--------|------|------|
| `random` | 随机采样不存在的有向边 | 默认训练策略 |
| `degree` | 按节点出度分布加权采样目标节点 | 对比实验 |
| `hard_2hop` | 从候选节点的二跳可达节点中采，且当前无边 | 难负样本对比 |

---

## 4. 模型架构

### 4.1 整体流程

```
输入: 子图 S_uv（DGLGraph，有向边集）+ 节点特征矩阵 X（始终存在）

Step 1: 节点标记
  每个节点 w ∈ S_uv 赋予 DRNL 标签 l(w)
  将标签嵌入为向量，与节点特征拼接：init_feat = [x_w || label_embed(l(w))]
  挂载到 g.ndata['feat']

Step 2: 子图拓扑编码（encoder_topo）
  输入: DGLGraph g，g.ndata['feat'] 为初始节点特征
  经过 L 层 GAT（保留有向边方向，入边/出边分别聚合后拼接）
  输出:
    h_u^topo  —— 节点 u 的表示（按局部节点id索引）
    h_v^topo  —— 节点 v 的表示
    h_G^pool  —— dgl.mean_nodes 全图 readout

Step 3: 节点属性编码（encoder_attr）[仅当 use_node_attr=true]
  输入: x_u, x_v（原始节点特征向量，从全局 nodes.csv 中按全局节点id取）
  经过共享权重 MLP
  输出: h_u^attr, h_v^attr

Step 4: 融合（fusion）
  h = Concat(h_u^topo, h_v^topo, h_G^pool,
             h_u^attr, h_v^attr,           # use_node_attr=false 时省略
             h_u^topo ⊙ h_v^topo)          # element-wise 交互项

Step 5: 评分头（scorer）
  s_uv = MLP(h) → sigmoid → (0, 1)
```

### 4.2 DRNL 节点标记（`labeling.py`）

对子图中每个节点 $w$，计算其到 $u$ 和 $v$ 的最短路径距离 $d_u, d_v$（使用截断图的无时间戳拓扑），然后：

$$\ell(w) = 1 + \min(d_u, d_v) + \frac{d}{2}\left\lfloor\frac{d-1}{2}\right\rfloor, \quad d = d_u + d_v$$

特殊情况：
- $w = u$：$\ell = 1$（固定标签）
- $w = v$：$\ell = 1$（固定标签，与 $u$ 相同，由位置区分）
- 某方向不可达：$d = \infty$，此时 $\ell = 0$（不可达标签）

标签嵌入：将离散标签通过 `nn.Embedding` 映射为 `label_dim` 维向量。

### 4.3 子图拓扑编码器（`encoder_topo.py`）

- **GNN类型**：GAT（默认），支持通过 config 切换为 GIN / GraphSAGE
- **层数**：`L=2`（默认），可配置
- **有向边处理**：分别聚合入边和出边，拼接后更新节点表示（Dir-GNN 方式）；使用 DGL 的 `dgl.graph` 表示有向子图，入边/出边通过 `reverse()` 或双向消息传递实现
- **Readout**：对所有节点做 mean pooling 得到 $h_G^{\text{pool}}$，使用 `dgl.mean_nodes`

```python
# 伪接口（DGL风格）
class TopoEncoder(nn.Module):
    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> tuple:
        # g: DGLGraph，有向子图，节点特征已挂载为 g.ndata['feat']
        # feat: (N, feat_dim) 初始节点特征（属性 + 标签嵌入）
        # 返回: (h_all, h_u, h_v, h_graph)
        # h_all: (N, hidden_dim)
        # h_u, h_v: (hidden_dim,)  通过节点在子图中的局部id索引
        # h_graph: (hidden_dim,)   mean pooling over all nodes
```

### 4.4 节点属性编码器（`encoder_attr.py`）

- 两个节点（$u$ 和 $v$）**共享权重**的 MLP
- 层数：2层，带 ReLU 和 LayerNorm
- 无节点属性时，该模块不实例化（由 config 中 `use_node_attr: false` 控制）

### 4.5 超参数默认值

```yaml
# configs/default.yaml
model:
  label_dim: 32
  hidden_dim: 64
  gnn_layers: 2
  gnn_type: GAT        # GAT | GIN | SAGE
  gnn_heads: 4         # GAT专用
  attr_mlp_layers: 2
  scorer_mlp_layers: 2
  dropout: 0.1
  use_node_attr: true  # 无属性数据集设为false

subgraph:
  max_hop: 2
  max_neighbors_per_node: 30   # 防止高度节点子图爆炸

training:
  lr: 0.001
  weight_decay: 1e-5
  epochs: 100
  batch_size: 64
  neg_sample_ratio: 1          # 每个正样本对应负样本数
  neg_strategy: random         # random | degree | hard_2hop
  loss: bce                    # bce | focal

evaluation:
  metrics: [auc, ap, hits@10, hits@20, hits@50]
```

---

## 5. 训练与评估

### 5.1 训练流程（`train.py`）

```
1. 加载数据集，执行时间切分
2. 构建训练集正样本列表（来自 train 时间段的边）
3. 对每个 epoch：
   a. 对每个正样本 (u, v, t_q)：
      - 以 t_q 为截断时刻，从历史图中提取子图 S_uv
      - 采样 neg_sample_ratio 个负样本 (u, v', t_q)，提取 S_uv'
   b. 批量前向传播，计算 BCE Loss（或 Focal Loss）
   c. 反向传播，梯度裁剪（max_norm=1.0）
   d. 验证集评估，保存最佳 checkpoint（以 AUC 为准）
4. 早停：val AUC 连续 10 epoch 不提升则停止
```

**注意**：子图提取是瓶颈，优先实现离线预计算并缓存到磁盘（`data/processed/<dataset>/subgraphs/`）。

### 5.2 评估流程（`evaluate.py`）

- 加载最佳 checkpoint，在测试集上推理
- 计算全部指标：AUC-ROC, Average Precision, Hits@K（K=10,20,50）
- Hits@K 计算方式：对每个正样本，从全部节点中随机采 999 个负样本，正样本排名在前 K 即命中

### 5.3 指标实现（`utils/metrics.py`）

```python
def compute_auc(y_true, y_score) -> float: ...
def compute_ap(y_true, y_score) -> float: ...
def compute_hits_at_k(pos_scores, neg_scores, k) -> float: ...
    # pos_scores: (N,)，neg_scores: (N, 999)
    # 对每个样本判断 pos_score 是否在 top-k 中
```

---

## 6. 实验计划

### 6.1 主实验

所有数据集上跑完整模型与所有 baseline，报告 AUC / AP / Hits@20。

**Baseline 列表**：
- 启发式：Common Neighbors, Adamic-Adar, Jaccard, Katz（静态图上计算，作为下界参考）
- 静态GNN：SEAL, GraphSAGE, GAT（节点级嵌入版）
- 时序GNN：TGAT, TGN（使用时序边信息，作为上界参考）
- **本方法**：Ours-Full

### 6.2 消融实验

在 CollegeMsg + Synth-SBM 上进行（规模适中，结果稳定）：

| 实验组 | 变体 | 修改点 |
|--------|------|--------|
| A: 子图范围 | Ours-1hop | 仅用1度邻居 |
| A: 子图范围 | Ours-2hop | 默认 |
| A: 子图范围 | Ours-3hop | 3度邻居（验证收益是否递减） |
| B: 节点标记 | Ours-NoLabel | 去掉 DRNL 标记 |
| B: 节点标记 | Ours-DistLabel | 改用简单距离标记 |
| B: 节点标记 | Ours-DRNL | 默认 |
| C: 属性融合 | Ours-NoAttr | 去掉属性编码器 |
| C: 属性融合 | Ours-TopoOnly | 等同 NoAttr |
| C: 属性融合 | Ours-AttrOnly | 去掉拓扑编码器，只用属性 |
| C: 属性融合 | Ours-Full | 默认 |
| D: 有向性 | Ours-Undirected | 边视为无向 |
| D: 有向性 | Ours-Directed | 默认 |
| E: GNN类型 | Ours-GIN | 替换 GAT |
| E: GNN类型 | Ours-SAGE | 替换 GAT |
| E: GNN类型 | Ours-GAT | 默认 |

### 6.3 负样本策略对比实验

固定模型（Ours-Full），在三个真实数据集上分别用三种负样本策略训练和测试，报告各指标变化，分析评估结论的稳健性。

### 6.4 合成数据集专项分析

目的：验证模型是否真正捕获了特定结构规律。

- **Synth-Triadic**：检验模型对三角闭合结构的评分是否显著高于随机节点对
- **Synth-Hawkes**：检验模型能否感知时序爆发节点的活跃度
- **Synth-SBM**：检验 GAT 注意力权重是否集中于社区内部节点

### 6.5 规模与效率分析

- 子图提取时间 vs. 节点度分布
- 训练速度（样本/秒）vs. 数据集规模
- `max_neighbors_per_node` 采样阈值的影响

---

## 7. 实现顺序与任务清单

严格按以下顺序实现，每步完成后运行对应的单元测试再进入下一步。

### Phase 1：数据基础设施

- [ ] `src/utils/split.py`：时间切分函数，含断言检查（验证无未来边泄露）
- [ ] `src/dataset/base.py`：数据集基类
- [ ] `src/dataset/real/college_msg.py`：CollegeMsg 预处理
- [ ] `src/dataset/synthetic/generator_base.py`：合成生成器基类
- [ ] `src/dataset/synthetic/sbm.py`：SBM 生成器
- [ ] 单元测试：验证时间切分、格式、自环过滤

### Phase 2：子图提取

- [ ] `src/graph/subgraph.py`：二度邻居子图提取，支持有向图，严格 $t < t_q$ 截断，返回 `dgl.DGLGraph`
- [ ] `src/graph/labeling.py`：DRNL 标记实现
- [ ] `src/graph/negative_sampling.py`：三种负样本策略
- [ ] 单元测试：小图上手动验证子图内容和标签正确性
- [ ] 离线子图缓存逻辑（生成后用 `dgl.save_graphs` 存 `.bin` 文件，加载用 `dgl.load_graphs`）

### Phase 3：模型实现

- [ ] `src/model/encoder_topo.py`：基于 `dgl.nn.GATConv` 的有向图编码器，入/出边分别聚合
- [ ] `src/model/encoder_attr.py`：共享权重 MLP
- [ ] `src/model/fusion.py`：拼接 + 交互项
- [ ] `src/model/scorer.py`：MLP 评分头
- [ ] `src/model/model.py`：组装，支持 `use_node_attr` 开关
- [ ] 前向传播 smoke test（随机数据，验证维度正确）

### Phase 4：训练与评估

- [ ] `src/utils/metrics.py`：AUC, AP, Hits@K
- [ ] `src/train.py`：完整训练循环，含早停和 checkpoint
- [ ] `src/evaluate.py`：测试集评估
- [ ] `src/baseline/heuristic.py`：CN, AA, Jaccard, Katz
- [ ] CollegeMsg 上端到端跑通

### Phase 5：扩展数据集与 Baseline

- [ ] `src/dataset/real/bitcoin_otc.py`
- [ ] `src/dataset/real/email_eu.py`
- [ ] `src/dataset/synthetic/hawkes.py`
- [ ] `src/dataset/synthetic/triadic.py`
- [ ] `src/baseline/seal.py`（可复用官方实现，封装接口）
- [ ] `src/baseline/graphsage.py`
- [ ] `src/baseline/tgat.py`

### Phase 6：实验脚本

- [ ] `experiments/run_main.py`：主实验，遍历所有数据集 × 所有模型
- [ ] `experiments/run_ablation.py`：消融实验
- [ ] `experiments/run_negative.py`：负样本策略对比
- [ ] `experiments/run_synthetic.py`：合成数据专项
- [ ] `results/tables/` 自动生成 LaTeX 表格

### Phase 7：分析与可视化

- [ ] `notebooks/attention_visualization.ipynb`：GAT 注意力权重热图
- [ ] `notebooks/case_study.ipynb`：合成数据集上的结构案例分析
- [ ] 按节点度分层的 Hits@K 分析

---

## 8. 技术栈与约束

### 依赖

```
torch >= 2.0
dgl >= 2.1                  # 图神经网络框架，替代 PyG
numpy
pandas
scipy
scikit-learn
pyyaml
tqdm
matplotlib
jupyter
```

**DGL 关键 API 使用约定**：
- 子图用 `dgl.graph((src, dst))` 构建，节点特征挂载为 `g.ndata['feat']`
- 有向图消息传递：入边聚合用 `g.in_edges`，出边聚合用 `g.out_edges` 或 `dgl.reverse(g)`
- 图级 readout：`dgl.mean_nodes(g, 'feat')`
- 批量子图：`dgl.batch([g1, g2, ...])` 用于 mini-batch 训练
- GAT 使用 `dgl.nn.GATConv`，GIN 使用 `dgl.nn.GINConv`，GraphSAGE 使用 `dgl.nn.SAGEConv`

### 硬件假设

- GPU：RTX 3090（24GB VRAM）
- 子图缓存存储：至少 20GB 空余磁盘空间

### 代码约束

1. **时间泄露防护**：所有子图提取调用必须传入 `cutoff_time`，函数内部断言过滤
2. **可复现性**：所有随机操作使用 `src/utils/seed.py` 中的全局 seed 设置
3. **配置驱动**：超参数全部通过 yaml config 传入，禁止硬编码在模型文件中
4. **接口统一**：所有数据集实现 `base.py` 中的接口，所有 baseline 实现统一的 `predict(u, v, graph) -> float` 接口

### 实验记录

- 每次实验自动保存 config snapshot 到 `results/logs/<run_id>/config.yaml`
- 训练曲线保存为 `results/logs/<run_id>/metrics.json`
- 最终结果汇总到 `results/tables/main_results.csv`
