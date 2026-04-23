# TODO — 中长期任务与研究计划

> 创建时间：2026-04-08 15:30
> 最后更新：2026-04-20

---

## Phase 1：数据基础设施 ✅ 已完成

- [x] `src/utils/split.py`：时间切分函数，含断言检查
- [x] `src/dataset/base.py`：数据集基类（含 `first_time_only` 开关）
- [x] `src/dataset/real/college_msg.py`：CollegeMsg 预处理
- [x] `src/dataset/synthetic/generator_base.py`：合成生成器基类
- [x] `src/dataset/synthetic/sbm.py`：SBM 生成器
- [x] 单元测试：`tests/test_split.py`、格式验证、自环过滤

## Phase 2：子图提取 ✅ 已完成

- [x] `src/graph/subgraph.py`：二度邻居子图提取（有向图，严格 $t < t_q$ 截断）+ `TimeAdjacency` 快路径
- [x] `src/graph/labeling.py`：DRNL 标记实现
- [x] `src/graph/negative_sampling.py`：random / hard_2hop / historical / degree / 混合策略
- [x] `tests/test_subgraph.py`、`tests/test_labeling.py`
- [x] 离线子图缓存逻辑（`cache_subgraphs` / `load_cached_subgraphs`，待接入训练流程）

## Phase 3：模型实现 ✅ 已完成

- [x] `src/model/gin_encoder.py`：GINEncoder / GINEncoderLayerConcat / GINEncoderLayerSum
- [x] `src/model/encoder_attr.py`：节点属性 MLP encoder
- [x] `src/model/scorer.py`：MLP 评分头 → sigmoid
- [x] `src/model/model.py`：`LinkPredModel` 组装，`encoder_type` / `use_node_attr` 开关
- [x] 前向传播 smoke test（`tests/smoke_model.py`）

## Phase 4：训练与评估 ✅ 已完成

- [x] `src/utils/metrics.py`：AUC / AP / Hits@K / MRR / NDCG@K
- [x] `src/train.py`：完整训练循环（legacy + simulated_recall 双协议），含早停和 checkpoint
- [x] `src/evaluate.py`：测试集评估
- [x] `src/baseline/heuristic.py`：CN / AA / Jaccard
- [x] CollegeMsg 上端到端跑通（AUC=1.0 smoke test）

## Phase 5：扩展数据集与 Baseline ✅ 已完成

- [x] `src/dataset/real/bitcoin_otc.py`
- [x] `src/dataset/real/email_eu.py`
- [x] `src/dataset/real/bitcoin_alpha.py` / `dnc_email.py` / `sx_mathoverflow.py` / `sx_askubuntu.py` / `sx_superuser.py`
- [x] `src/dataset/real/gowalla.py` / `epinions.py`（loader 实现，数据待下载）
- [x] `src/dataset/synthetic/hawkes.py` / `triadic.py`
- [x] `src/baseline/seal.py`
- [x] `src/baseline/graphsage.py`
- [x] `src/baseline/tgat.py`（已实现但后续不投入，见 DECISIONS.md）

## Phase 5.5：模拟召回框架 ✅ 已完成（2026-04-17）

- [x] `src/graph/edge_split.py`：`filter_first_time_edges` / `TwoLayerEdgeSet` / `temporal_mask_split` / `random_mask_split` / `build_two_layer`
- [x] `src/recall/`：`RecallBase` / `CommonNeighborsRecall` / `AdamicAdarRecall` / `build_recall` / `CurriculumScheduler`
- [x] `src/train.py`：`RecallDataset` / `eval_mrr_epoch` / `_run_simulated_recall` / `--protocol` 分支
- [x] 端到端测试：CollegeMsg simulated_recall，MRR=0.0996 >> 1/100，97 例测试全通过

## Phase 6：实验脚本（部分完成）

- [x] `scripts/run_comparison.py`：多模型多数据集对比实验（并行）
- [x] `scripts/run_encoder_ablation.py`：encoder_type 消融
- [x] `scripts/run_protocol_comparison.py`：legacy vs simulated_recall 协议对比
- [x] `scripts/preprocess_datasets.py`：批量预处理
- [ ] `results/tables/` LaTeX 表格自动生成
- [ ] 消融实验：`--first_time_only` on/off 对比
- [ ] 消融实验：召回策略 CN vs AA vs union 对比
- [ ] `simulated_recall` 协议全数据集正式实验（bitcoin_otc / email_eu / dnc_email…）
- [ ] **encoder_type 消融**：GIN-last vs GIN-layer_concat vs GIN-layer_sum 效果对比
  - 固定：CollegeMsg × ego_cn × simulated_recall × hidden_dim=64 × seed=42
  - 参数：`--encoder_type last | layer_concat | layer_sum`
  - 指标：val MRR / Hits@10

## Phase 7：分析与可视化

- [ ] `notebooks/attention_visualization.ipynb`：GAT 注意力权重热图
- [ ] `notebooks/case_study.ipynb`：合成数据结构案例分析
- [ ] 按节点度分层的 Hits@K 分析

---

## 性能优化任务

### 离线子图缓存（训练提速，优先级：高）

**背景**：当前训练慢的根因是每个样本在 `collate_fn` 里都做一次 O(E) 的 DataFrame 全表扫描 + 邻接表重建（`subgraph.py` 慢路径），且没有跨 epoch 缓存，30 epoch 重复计算 30 次。

**方案**：在训练开始前一次性提取所有 `(u, v, cutoff_time)` 对应子图并存到磁盘，之后每个 epoch 直接读缓存。`cache_subgraphs()` / `load_cached_subgraphs()` 工具已在 `subgraph.py:241` 实现，只需接入训练流程。

**磁盘估算**：
- 当前 `--max_samples 2000` 实验：~25 MB
- 全量三数据集（含 val）：~1.1 GB

**预计加速**：10~30×（消除 N×E 重复计算）

**实现要点**（约 30 行改动）：
1. `train.py`：`build_model` 之前检查缓存是否存在，不存在则调 `cache_subgraphs()`
2. `LinkPredDataset.__init__`：改为加载缓存图列表，`__getitem__` 返回 `(DGLGraph, label)`
3. `collate_fn`：改为只做 `dgl.batch`，不再调 `extract_subgraph`
4. 缓存键：`dataset_name + split + max_hop + max_neighbors + neg_ratio`

**其他可选优化**（次优先级）：
- 把子图提取移入 `__getitem__` + `num_workers≥4`（Windows 上有多进程坑，暂缓）
- `--max_hop 1` 消融：子图缩减约 10×，先验证 1-hop AUC 是否足够

---

## ✅ 子图设计对比表（已完成，2026-04-23）

| 数据集 | bfs_2hop MRR | ego_cn MRR | 胜者 |
|---|---|---|---|
| CollegeMsg | ~0.048（烟测） | **0.1934** | ego_cn |
| bitcoin_otc | **0.2849** | 0.2328 | bfs_2hop |
| email_eu | **0.1958** | 0.1714 | bfs_2hop |

结论：ego_cn 在稠密小图占优；bfs_2hop 在稀疏大图占优。

---

## ✅ 消融实验：hidden_dim 敏感性分析（已完成，2026-04-23）

固定配置：CollegeMsg × ego_cn × simulated_recall × seed=42

| hidden_dim | val MRR | vs 64 |
|---|---|---|
| 4  | 0.1384 | -28.4% |
| 8  | 0.1738 | -10.1% |
| 16 | 0.1853 | -4.2%  |
| 32 | 0.1865 | -3.6%  |
| **64** | **0.1934** | 基线 |

结论：帕累托点在 32，模型未过参数化，后续保持 hidden_dim=64。

---

## 研究问题备忘

- 2-hop vs 1-hop vs 3-hop 的收益边界在哪里？
- DRNL 标记对有向图是否需要修改（原论文针对无向图）？
- 合成数据集上 GAT 注意力权重能否可视化出社区结构？
- `max_neighbors_per_node=30` 的阈值对高度节点的影响？

---

## 远期研究方向：在线学习 + 社交网络生成过程评估

> 记录时间：2026-04-17

### 核心想法

当前框架是**离线批训练**：图结构固定，模型学完再评估精度。

未来考虑切换到**在线学习模式**：

- 模型从零开始，在一张空图（或极稀疏初始图）上持续学习
- 每轮：模型推荐 → 用户接受/拒绝 → 新边形成 → 图结构更新 → 模型用新数据继续训练
- 形成闭合的"推荐-反馈-图演化"回路

### 评估范式转变

不再只看最终的 AUC / MRR，而是评估**整个过程**：

- **收敛速度**：网络从稀疏到接近真实社交结构需要多少轮推荐
- **结构质量轨迹**：每轮后计算图的聚类系数、平均路径长度、度分布与真实网络的 KL 散度
- **推荐多样性 vs 回音壁效应**：算法是否导致社交圈固化（filter bubble）
- **最终形态对比**：模拟生成的网络与真实熟人网络在宏观结构上的相似程度

### 与现有框架的关系

- 现有"两层图 + 模拟召回"框架是此方向的**离线近似版本**：
  edge dropout 模拟稀疏初始图，模拟召回模拟推荐曝光，E_hidden 模拟用户反馈
- 在线版本需要引入**用户行为模拟器**（接受概率模型）和**图动态更新机制**
- 与 memory 中记录的"未来研究：社交网络反馈模拟"方向完全一致

### 关键技术挑战

1. 用户接受概率模型：如何模拟"推荐后用户是否关注"（非确定性）
2. 非平稳分布：图结构变化导致模型输入分布持续漂移，需要处理 distribution shift
3. 评估指标设计：如何定义"好的社交网络形成过程"的量化标准
4. 与真实数据对齐：如何用公开图数据集验证模拟出的网络是否合理

### 前置条件

- 完成当前 Step 1-8 离线框架，验证"模拟召回 + GNN 精排"在离线场景下有效
- 在合成数据集（SBM / Triadic）上先做在线模拟实验（可控环境）
- 再迁移到真实数据集（CollegeMsg 等）
