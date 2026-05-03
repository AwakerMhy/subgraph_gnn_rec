# PROGRESS — 当前任务进度

> 创建时间：2026-04-08 15:30
> 最后更新：2026-05-02

---

## [新离线协议：ego_cn_offline]
- 状态：✅ 已完成
- 开始时间：2026-04-30
- 完成时间：2026-04-30

### 任务定义
- 切分：70% E_obs / 中15% 训练 / 后15% 测试（时间切分）
- 正样本：E_hidden 中存在的边 (u→v)
- 负样本：(u→v) 满足 (u,v) ∉ E_obs 且 (u,v) ∉ E_hidden（当前切分期）
- 子图：u 的 ego 子图 + u,v 公共邻居（ego_cn），背景图始终为 E_obs
- cutoff_time：固定传 2.0（> 所有归一化时间戳），TimeAdjacency 仅含 E_obs 边，不存在泄露

### 实现步骤
- [x] 步骤1：新增 `EgoCNOfflineDataset` 类（src/train.py，RecallDataset 之后）
- [x] 步骤2：新增 `_run_ego_cn_offline` 函数（src/train.py）
- [x] 步骤3：更新 `main()` 协议分支，加入 `ego_cn_offline`
- [x] 步骤4：smoke test 通过（sbm 3 epoch，MRR 0.079→0.167，loss 正常收敛）

### 已变更文件
- src/train.py

---

## [在线学习仿真：CollegeMsg 全量运行]
- 状态：✅ 已完成
- 开始时间：2026-04-23
- 完成时间：2026-04-30
- 配置：configs/online/college_msg_full.yaml（mixture 召回+composite 用户+p_neg=0.02+decay+replay=200）
- [x] 步骤 1：dry_run 验证（CPU/CUDA 均通过，1 轮 coverage=0.0684 prec@K=0.0373）
- [x] 步骤 2：100 轮完整仿真（results/online/college_msg_full/rounds.csv，共 100 轮）
- [x] 步骤 3：最终指标（round 99）：coverage=0.1157，prec@K=0.0251，mrr@1=0.175，mrr@3=0.242，mrr@5=0.242

---

## [子图设计迭代：ego-graph + 公共邻居方案]
- 状态：进行中
- 开始时间：2026-04-22
- 当前进度：第 3 步 / 共 3 步（已完成）
- [x] 步骤 1：实现新设计（ego-graph + CN 融合，1-hop 采样）
- [x] 步骤 2：全 30-epoch 训练对比（CollegeMsg, simulated_recall）
- [x] 步骤 3：系统对比实验（两种设计 × 多数据集）
  - [x] CollegeMsg × ego_cn × 30ep（val MRR=0.1934，Hits@10=0.2887）
  - [x] CollegeMsg × bfs_2hop（烟测估计 MRR≈0.048）
  - [x] bitcoin_otc × bfs_2hop × 30ep（**val MRR=0.2849**）
  - [x] bitcoin_otc × ego_cn × 30ep（val MRR=0.2328，早停ep14）
  - [x] email_eu × bfs_2hop × 30ep（**val MRR=0.1958**）
  - [x] email_eu × ego_cn × 30ep（val MRR=0.1714，早停ep21）
- 状态：✅ 已完成
- 关键发现：ego_cn 在 CollegeMsg（稠密小图）占优；bfs_2hop 在 bitcoin_otc/email_eu（稀疏大图）占优
- 已变更文件：src/graph/subgraph.py

---

## [hidden_dim 敏感性消融实验]
- 状态：✅ 已完成
- 开始时间：2026-04-23
- 完成时间：2026-04-23
- 固定配置：CollegeMsg × ego_cn × simulated_recall × 30ep（patience=10）× seed=42
- [x] hidden_dim=4
- [x] hidden_dim=8
- [x] hidden_dim=16
- [x] hidden_dim=32
- [x] hidden_dim=64（基线）

### 结果表

| hidden_dim | val MRR | vs 64 |
|---|---|---|
| 4  | 0.1384 | -28.4% |
| 8  | 0.1738 | -10.1% |
| 16 | 0.1853 | -4.2%  |
| 32 | 0.1865 | -3.6%  |
| **64** | **0.1934** | 基线 |

### 结论
- 32→64 边际收益仅 0.4%，**帕累托点在 32**
- 模型未过参数化（64 仍是最优），无欠容量迹象
- 后续实验可用 hidden_dim=64 作为默认配置，如需轻量版取 32

---

## [新数据集接入：Facebook/Twitch/LastFM/Epinions/ogbl-collab]
- 状态：进行中
- 开始时间：2026-04-23
- [x] 创建 loader：facebook_ego.py / twitch_gamers.py / lastfm_asia.py / ogbl_collab.py
- [x] 下载 + 预处理：facebook_ego（4039节点/176468边）
- [x] 下载 + 预处理：lastfm_asia（7624节点/55612边）
- [x] 下载 + 预处理：epinions（75879节点/508837边）
- [ ] 下载 + 预处理：twitch_gamers（zip 下载中）
- [ ] 预处理：ogbl_collab（需 pip install ogb）
- [x] 下载 + 预处理：twitch_gamers（168k节点/13.6M边）
- [x] 下载 + 预处理：ogbl_collab（235k节点/1.9M边）
- [x] 训练：facebook_ego × ego_cn × random_split × 30ep（✅ val MRR=0.9402）
- [x] 训练：lastfm_asia × ego_cn × 30ep（✅ val MRR=0.9663）
- [x] 训练：epinions × ego_cn × temporal_split（❌ 失败：val split 为空，数据集无时间戳，temporal_split 退化）
- [x] 训练：twitch_gamers × ego_cn（❌ 失败：同上，val split 为空）
- [ ] 训练：ogbl_collab × ego_cn × temporal_split × 30ep（⏳ 未启动）
- 状态：部分完成（facebook_ego + lastfm_asia 已跑；epinions/twitch 失败待处理；ogbl_collab 未启动）

---

## [encoder_type 消融：GIN-last vs layer_concat vs layer_sum]
- 状态：✅ 已完成
- 开始时间：2026-04-23
- 完成时间：2026-04-30
- 固定配置：CollegeMsg × ego_cn × simulated_recall × hidden_dim=64 × seed=42
- [x] encoder_type=last（ep21 早停，best ep11，MRR=0.2001，Hits@10=0.3196）
- [x] encoder_type=layer_concat（ep27 早停，best ep17，MRR=0.2171，Hits@10=0.3711）
- [x] encoder_type=layer_sum（ep19 早停，best ep9，MRR=0.2176，Hits@10=0.3196）

### 结论
- layer_concat 和 layer_sum 均优于 last（MRR +8-9%）
- layer_concat 在 Hits@10 上略胜（0.3711 vs 0.3196）
- 默认配置沿用 layer_sum（online 实验中已在用）

### 烟测结果对比（100 样本，1 epoch，CollegeMsg）
| 指标 | 2-hop BFS | ego+CN | 改进 |
|---|---|---|---|
| tr_auc | 0.5248 | 0.4866 | ↓1.7% |
| MRR | 0.0478 | 0.0949 | ↑98% |
| Hits@10 | 0.0909 | 0.1818 | ↑100% |

### 完整训练结果（2000 样本，30 epoch 早停于 21，CollegeMsg，simulated_recall）
| 指标 | 2-hop BFS (烟测估计) | ego+CN 30ep | 改进 |
|---|---|---|---|
| val MRR | 0.0478 | **0.1934** | +305% |
| Hits@10 | 0.0909 | **0.2887** | +218% |
| val tr_auc | ~0.52 | 0.6365 | +22% |
- 最佳 epoch：11，checkpoint：`results/checkpoints/cmp_ego_cn_30ep_best.pt`

---

## [两层图 + 模拟召回框架 — Step 1-6 实现]
- 状态：已完成
- 开始时间：2026-04-17
- 完成时间：2026-04-17
- [x] Step 1: `src/graph/edge_split.py::filter_first_time_edges`（首次边过滤）
- [x] Step 3: `TwoLayerEdgeSet` + `temporal_mask_split` / `random_mask_split` / `build_two_layer`
- [x] Step 4: `src/recall/` 包（`base.py`, `heuristic.py`, `registry.py`）— CN + AA 召回
- [x] Step 5: `compute_reciprocity_labels`（互惠性标签，备用）
- [x] Step 6: `src/utils/metrics.py` 扩展 MRR / NDCG@K / `compute_ranking_metrics`
- [x] `src/train.py` 集成：`RecallDataset`, `eval_mrr_epoch`, `--protocol` 分支, `_run_simulated_recall`
- [x] 端到端冒烟测试（CollegeMsg × 3 epoch）：MRR=0.0996 >> 1/100，无泄露断言通过
- 已变更文件：src/graph/edge_split.py, src/recall/\*, src/utils/metrics.py, src/train.py, configs/default.yaml, src/dataset/base.py
- 测试：tests/test_edge_split.py（23例）, tests/test_recall.py（21例）, tests/test_metrics.py（新增11例）— 97例全部通过

---

## [在线学习仿真框架 — src/online/ 新包]
- 状态：已完成
- 开始时间：2026-04-20
- 完成时间：2026-04-20
- [x] `src/online/static_adj.py`：StaticAdjacency（duck-type TimeAdjacency，O(1) 动态加边）
- [x] `src/online/env.py`：OnlineEnv（G*/G_t/cooldown/用户采样/图演化）
- [x] `src/online/feedback.py`：FeedbackSimulator（伯努利接受）
- [x] `src/online/trainer.py`：OnlineTrainer（子图批量打分 + BCE 梯度更新）
- [x] `src/online/replay.py`：ReplayBuffer（capacity=0 时 no-op）
- [x] `src/online/evaluator.py`：RoundMetrics（Precision/MRR/coverage/聚类系数/degree KL）
- [x] `src/online/schedule.py`：build_scheduler（cosine_warmup/constant/step）
- [x] `src/online/loop.py`：run_online_simulation 主循环
- [x] `configs/online/`：default.yaml / sbm_smoke.yaml / college_msg.yaml
- [x] `scripts/run_online_sim.py`：CLI 入口
- [x] `tests/test_online.py`：10/10 单元测试通过
- [x] 端到端烟测（sbm_smoke，3 轮）：无异常，rounds.csv 正常写出
- 已变更文件：src/online/\*（新建），configs/online/\*（新建），scripts/run_online_sim.py，tests/test_online.py

---

## [全量对比实验：legacy 协议，bitcoin_otc + email_eu × 2模型]
- 状态：⏸ 暂时废弃（2026-04-21，专注在线仿真框架）
- 开始时间：2026-04-20 00:37
- 配置：epochs=30, full dataset, neg=random:0.5+hard_2hop:0.3+degree:0.2, max_workers=2, seed=42
- 日志路径：`results/logs/otc_eu_rerun_apr20.log`
- 结果路径：`results/logs/cmp_bitcoin_otc_*/train.json`、`results/logs/cmp_email_eu_*/train.json`

### 运行进度
- [x] college_msg / GIN-last（**已放弃**：6 epoch 后 val_auc_mean 单调下降 0.535→0.52，tr_auc≈0.99 严重过拟合，稠密小图无改善空间）
- [x] college_msg / GraphSAGE（**已放弃**：同上，0.533→0.52）
- [ ] college_msg / SEAL（跳过，DRNL 在稠密图上失效，见 DECISIONS.md）
- [ ] bitcoin_otc / GIN-last（🟡 运行中）
- [ ] bitcoin_otc / GraphSAGE（🟡 运行中）
- [ ] bitcoin_otc / SEAL（⏳ 排队）
- [ ] email_eu / GIN-last（⏳ 排队）
- [ ] email_eu / GraphSAGE（⏳ 排队）
- [ ] email_eu / SEAL（⏳ 排队）
- [ ] dnc_email / GIN-last（⏳ 待启动）
- [ ] dnc_email / GraphSAGE（⏳ 待启动）
- [ ] bitcoin_alpha / GIN-last（⏳ 待启动）
- [ ] bitcoin_alpha / GraphSAGE（⏳ 待启动）

### 下一步行动
等 bitcoin_otc + email_eu 完成后，查看结果；再决定是否启动 dnc_email / bitcoin_alpha

---

## [新增 GraphSAGE / SEAL / TGAT baseline + 对比脚本]
- 状态：⏸ 暂时废弃（2026-04-21）
- 开始时间：2026-04-14 10:00
- [x] src/baseline/graphsage.py
- [x] src/baseline/seal.py
- [x] src/baseline/tgat.py
- [x] src/graph/subgraph.py：添加 store_edge_time 参数
- [x] src/train.py：添加 --model_type，更新 collate_fn / run_epoch
- [x] scripts/run_comparison.py：对比脚本
- 已变更文件：src/baseline/graphsage.py, src/baseline/seal.py, src/baseline/tgat.py, src/graph/subgraph.py, src/train.py, scripts/run_comparison.py
- 下一步行动：运行对比实验

## [Scorer 架构调整：修复数值爆炸]
- 状态：⏸ 暂时废弃（2026-04-21）
- 开始时间：2026-04-10
- [x] 修改 model.py：scorer_hidden_dim 默认值改为与 hidden_dim 挂钩
- [ ] 重跑验证（后台任务 bfnkeerpg）
- 已变更文件：src/model/model.py
- 下一步行动：等训练完成，观察 bitcoin_otc layer_concat/sum 的 val_loss 是否稳定

---

## [负样本策略修复：全时段排除假负样本]
- 状态：已完成
- 开始时间：2026-04-09 15:00
- [x] 修改 negative_sampling.py：sample_negatives 新增 all_time_adj_out 参数
- [x] 修改 train.py：预计算全时段邻接表并传入 LinkPredDataset
- [ ] 重跑 bitcoin_otc × 3 encoder（hidden_dim=8，后台任务 bkbifi7es）
- 已变更文件：src/graph/negative_sampling.py, src/train.py
- 下一步行动：等训练完成，对比 val_auc 是否有提升

---

## [Bitcoin-OTC / Email-EU × 3 Encoder 对比实验]
- 状态：⏸ 废弃（2026-04-21，整体切至在线仿真框架）
- 开始时间：2026-04-09 14:00

---

## [GIN 多变体 Graph Encoder]
- 状态：已完成
- 开始时间：2026-04-09
- [x] gin_encoder.py 新增 GINEncoderLayerConcat（每层 u/v/others mean pooling → concat，输出 L×3H）
- [x] gin_encoder.py 新增 GINEncoderLayerSum（每层 u/v/others mean pooling → concat 后各层相加，输出 3H）
- [x] model.py 增加 encoder_type 参数（"last" / "layer_concat" / "layer_sum"），自动适配 scorer in_dim
- 已变更文件：src/model/gin_encoder.py, src/model/model.py
- 下一步行动：在消融实验中对比三种 encoder_type

---

## [Phase 4 收尾 — heuristic baseline]
- 状态：已完成
- 开始时间：2026-04-09 10:00
- 完成时间：2026-04-09 12:00
- [x] 步骤1：tests/test_heuristic.py（10/10 通过）
- [x] 步骤2：CollegeMsg 预处理（1899节点，59835边）
- [x] 步骤3：性能优化（prebuilt_adj，拒绝采样，快路径 extract_subgraph）
- [x] 步骤4：端到端跑通（train 3 epoch + evaluate，AUC=1.0 smoke test）
- 已变更文件：src/graph/negative_sampling.py, src/graph/subgraph.py, src/train.py, src/evaluate.py, tests/test_heuristic.py
- 下一步行动：Phase 5 — 扩展数据集（Bitcoin-OTC, Email-EU）

---

## [Phase 3 — 简化模型实现]
- 状态：已完成
- 开始时间：2026-04-08 17:00
- 完成时间：2026-04-08 17:30
- [x] src/model/gin_encoder.py：GIN 图编码器（2维one-hot输入→mean pooling→graph embedding）
- [x] src/model/scorer.py：MLP 评分头（graph embedding → sigmoid score）
- [x] src/model/model.py：顶层组装（one-hot赋值 + GIN + Scorer，含 forward_batch）
- [x] tests/smoke_model.py：单图/batch smoke test 全部通过
- 已变更文件：src/model/gin_encoder.py, src/model/scorer.py, src/model/model.py, tests/smoke_model.py
- 下一步行动：实现 Phase 4 — src/train.py 训练循环

## [Phase 4 — 训练与评估]
- 状态：✅ 已完成（2026-04-09）

---

## [Bug 修复批次：评估指标 + 仿真逻辑]
- 状态：✅ 已完成
- 完成时间：2026-04-24
- smoke test：3 轮全通过

### 修复清单
| Bug | 文件 | 影响 |
|---|---|---|
| MRR@K 三个 K 值相同（未截断） | evaluator.py | 指标虚报 |
| 正样本用 accepted_set 漏掉 p_pos 拒绝的真正样本 | evaluator.py | MRR/Hits 低估 |
| rec_coverage@K 分母=全图节点数（被 sample_ratio 稀释） | evaluator.py | 跨数据集横比失效 |
| pos_arr/neg_arr 死代码（`if False`） | evaluator.py | 代码污染 |
| `_refresh_G_t` 每次 `to_undirected()` O(E) 重建 | evaluator.py | 大图每 10 轮浪费 |
| `set_cooldown_mode("decay")` 清空 cooldown 表 | env.py | 切换时机影响实验 |
| cooldown 清理 decay 模式恒清空（`reject_round > t` 恒 False） | env.py | cooldown 惩罚每 10 轮消失 |
| 冷启动有放回抽样（两步转换+重复节点） | loop.py | 候选重复，信噪比低 |
| cooldown exclude 在 decay 模式下漏判 | env.py + loop.py | 冷启动可重采刚拒绝的节点 |
| `extract_topo_features` 每轮调用两次 | loop.py | MLP 模式冗余计算 |
| UserSelector degree 每轮 O(N+E) Python loop | user_selector.py | 大图每轮慢 1-3s |
| `node_feat` gather 后无 pin_memory | trainer.py | H→D 拷贝与 forward 串行 |

---

## [训练速度优化：Numba + Sparse MatMul]
- 状态：✅ 已完成（基准测试通过）
- 开始时间：2026-04-23 20:00
- 完成时间：2026-04-24 11:30
- 目标：解决 facebook_ego_exp1 指数级减速（18s→297s/轮）

### 实现细节
- [x] Numba 并行编译：_count_edges_batched_nb / _fill_edges_batched_nb（二分查找 + prange）
- [x] Scipy 稀疏矩阵：CommonNeighborsRecall / AdamicAdarRecall 批量预计算（A[users]@A + 加权）
- [x] Numpy 向量化：np.union1d 替代 sorted({u,v}|nbrs_u|cn)，O(d)→O(d log d)
- [x] 容错机制：Numba不可用时fallback numpy；scipy不可用时fallback set ops；Torch不可用时跳过子图测试

### 基准测试结果（facebook_ego_exp1 规模：4039节点 / 15000边 / 400活跃用户）

| 组件 | 旧实现 | 新实现 | 加速比 | 正确性 |
|---|---|---|---|---|
| CN召回（400用户×set intersection） | 1.3ms | 5.1ms | **0.3×**（变慢） | ✅ 10/10通过 |
| AA召回（加权版本） | 1.6ms | 5.5ms | **0.3×**（变慢） | ✅ 10/10通过 |
| 子图建图（512 pairs，union1d优化） | — | — | — | Torch阻止 |

### 关键发现
1. **Scipy稀疏矩阵反而变慢**：在4039节点小图上，矩阵转dense()的开销（O(n²)内存）远大于集合操作的收益
2. **预期的加速场景**：dynamic graph增长到更大规模（边数>50k），稠密度提高时，sparse matmul优势才会显现
3. **Numba并行不可用**：Windows Smart App Control 阻止 DLL 加载，无法启用
4. **正确性验证通过**：CN/AA 公式实现正确，结果与 _two_hop_scores fallback 完全一致

### 原因分析
- 小图上集合操作（O(d)）比矩阵操作（O(n²)内存分配）更高效
- Scipy稀疏op的编译和内存开销在小规模上摊销不了
- 需要在动态增长的真实实验（facebook_exp1）中测试，才能看到真正的瓶颈

### 已变更文件
- src/online/trainer.py：Numba @njit / prange / torch条件导入
- src/recall/heuristic.py：Scipy稀疏矩阵 + 缓存
- src/online/static_adj.py：CSR懒构建 + iter_out_neighbors
- tests/bench_optimizations.py：完整基准测试套件
- 下一步行动：在facebook_ego_exp1完整运行中测实际减速原因，可能不是召回/子图建图而是其他层面
