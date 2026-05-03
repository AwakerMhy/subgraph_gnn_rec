# BLUEPRINT — Social Network Online Learning Simulation

> 创建时间：2026-04-08 15:30
> 最后更新：2026-05-02

**本文件是项目地图，每次任务开始前必读。目的是减少 grep/read 开销。**

---

## 目录树与模块职责

```
gnn/
├── CLAUDE.md              # CC 行为规范（每次会话自动加载）
├── BLUEPRINT.md           # 本文件：项目地图（每次任务前必读）
├── DECISIONS.md           # 架构决策记录
├── MISTAKES.md            # 错误与教训库
├── PROGRESS.md            # 当前任务进度
├── TODO.md                # 中长期研究计划
├── CHECKPOINT.md          # 会话恢复点
├── META_REFLECTION.md     # 周期性元反思
├── PLAN.md                # 完整项目计划（只读参考）
├── requirements.txt
│
├── configs/
│   ├── default.yaml       # 默认超参（含 protocol/recall/curriculum 新字段）
│   ├── dataset/           # 各数据集 yaml（college_msg/bitcoin_otc/email_eu/gowalla/epinions…）
│   ├── model/             # 模型变体 yaml（含消融实验）
│   └── online/            # 在线仿真配置（algo_sweep_*/bitcoin_alpha/college_msg_*/sbm*…，共 30+ yaml）
│
├── data/
│   ├── raw/               # 原始下载文件（.gitignore 排除）
│   ├── processed/         # 标准格式：edges.csv / nodes.csv / meta.json
│   └── synthetic/         # 合成数据集输出
│
├── src/
│   ├── dataset/
│   │   ├── base.py        # TemporalDataset 基类（含 first_time_only 开关）
│   │   ├── real/
│   │   │   ├── college_msg.py
│   │   │   ├── bitcoin_otc.py
│   │   │   ├── bitcoin_alpha.py
│   │   │   ├── email_eu.py
│   │   │   ├── dnc_email.py
│   │   │   ├── sx_mathoverflow.py
│   │   │   ├── sx_askubuntu.py
│   │   │   ├── sx_superuser.py
│   │   │   ├── gowalla.py
│   │   │   ├── epinions.py
│   │   │   ├── facebook_ego.py  # Facebook ego-network（4039 节点，176468 边）
│   │   │   ├── lastfm_asia.py   # LastFM Asia（7624 节点，55612 边）
│   │   │   ├── ogbl_collab.py   # ogbl-collab（235k 节点，1.9M 边）
│   │   │   ├── twitch_gamers.py # Twitch Gamers（168k 节点，13.6M 边）
│   │   │   ├── wiki_vote.py   # Wikipedia 投票网络（7115 节点，103689 边，recip=0.056）
│   │   │   ├── slashdot.py    # Slashdot Zoo friend/foe（77350 节点，516575 边，recip=0.186）
│   │   │   └── email_euall.py # Email-EuAll 欧洲邮件网络（265009 节点，418956 边，recip=0.260）
│   │   └── synthetic/
│   │       ├── generator_base.py
│   │       ├── sbm.py           # 标准 SBM：均匀活跃度 + 共同邻居加成
│   │       ├── dcsbm.py         # DC-SBM：θ_out/θ_in Pareto 幂律度分布
│   │       ├── hawkes.py
│   │       ├── triadic.py
│   │       └── synth_dataset.py # SyntheticDataset 包装类（供 loop.py 加载合成图）
│   │
│   ├── graph/
│   │   ├── subgraph.py          # extract_subgraph + TimeAdjacency + 子图缓存工具
│   │   ├── edge_split.py        # TwoLayerEdgeSet + temporal/random_mask_split + 互惠性标签
│   │   ├── labeling.py          # DRNL 标记
│   │   └── negative_sampling.py # legacy 负采样（random/hard_2hop/historical/degree）
│   │
│   ├── recall/                  # 模拟召回子系统
│   │   ├── __init__.py          # 导出 RecallBase/CN/AA/build_recall/CurriculumScheduler
│   │   ├── base.py              # RecallBase ABC（含 update_graph hook）
│   │   ├── heuristic.py         # CommonNeighborsRecall / TwoHopRandomRecall / AdamicAdarRecall；自适应阈值（n≤10k用set ops，>10k用sparse matmul）
│   │   ├── ppr.py               # PPRRecall：个性化 PageRank power iteration
│   │   ├── community.py         # CommunityRandomRecall：社区内随机采样
│   │   ├── mixture.py           # MixtureRecall：配额合并多子召回器
│   │   ├── registry.py          # build_recall：支持 adamic_adar/ppr/community_random/mixture/two_hop_random
│   │   └── curriculum.py        # CurriculumScheduler
│   │
│   ├── model/
│   │   ├── gin_encoder.py       # GINEncoder / GINEncoderLayerConcat / GINEncoderLayerSum
│   │   ├── scorer.py            # MLP 评分头 → sigmoid
│   │   ├── model.py             # LinkPredModel（组装 GIN + Scorer，use_node_attr 开关）
│   │   ├── encoder_attr.py      # 节点属性 MLP encoder
│   │   └── node_emb_model.py    # NodeEmbModel（纯 embedding lookup ranker）
│   │
│   │
│   ├── online/                      # 在线学习仿真包（2026-04-20）
│   │   ├── static_adj.py            # StaticAdjacency：duck-type TimeAdjacency，O(1) 动态加边，懒构建 CSR
│   │   ├── env.py                   # OnlineEnv：G*/G_t/cooldown(hard+decay双模式)/cooldown_excluded_nodes()
│   │   ├── feedback.py              # FeedbackSimulator：p_pos/p_neg 伯努利接受模拟
│   │   ├── user_selector.py         # UserSelector：uniform/composite；degree 用 np.diff(CSR indptr) O(N)
│   │   ├── trainer.py               # OnlineTrainer：批量子图打分+BCE在线更新；numba并行边提取；pin_memory
│   │   ├── replay.py                # ReplayBuffer：capacity=0 时 no-op
│   │   ├── evaluator.py             # RoundMetrics：截断MRR@K/G*正样本/候选池coverage/延迟undirected重建
│   │   ├── schedule.py              # build_scheduler：cosine_warmup/constant/step
│   │   └── loop.py                  # run_online_simulation：主循环（mlp/gnn）；冷启动无放回采样；feat缓存
│   │
│   ├── baseline/
│   │   ├── heuristic.py         # CN / AA / Jaccard
│   │   ├── graphsage.py
│   │   ├── seal.py
│   │   ├── tgat.py
│   │   └── mlp_link.py          # MLPLinkScorer + extract_topo_features
│   │
│   ├── train.py                 # 训练主循环（含 protocol 分支 + RecallDataset）
│   ├── evaluate.py              # 测试集评估
│   └── utils/
│       ├── metrics.py           # AUC/AP/Hits@K/MRR/NDCG@K（Step 6 扩展）
│       ├── split.py             # temporal_split
│       ├── logger.py
│       └── seed.py
│
├── scripts/
│   ├── preprocess_datasets.py        # 批量预处理所有数据集
│   ├── preprocess_new_datasets.py    # 新数据集预处理（facebook/lastfm/ogbl/twitch）
│   ├── download_new_datasets.py      # 新数据集下载脚本
│   ├── run_online_sim.py             # 在线仿真 CLI 入口
│   ├── run_online_sim_win.py         # Windows 版本（规避多进程问题）
│   ├── run_algo_sweep.py             # 算法横向对比实验
│   ├── run_ablation_grid.py          # 消融网格调度（center_plus_flips / full）
│   ├── run_comparison.py             # 多模型对比实验
│   ├── run_encoder_ablation.py       # encoder_type 消融
│   ├── run_protocol_comparison.py    # legacy vs simulated_recall 协议对比
│   ├── run_eps_sweep.py              # epsilon-greedy 探索率扫描
│   ├── run_hidden_dim_ablation.sh    # hidden_dim 消融 shell 脚本
│   ├── run_gt_init_sweep.py          # ground_truth init 扫描
│   ├── gen_thr_grid_configs.py       # 阈值网格 config 生成
│   ├── precompute_subgraphs.py       # 离线子图预计算
│   ├── plot_algo_sweep.py            # algo_sweep 结果出图
│   ├── plot_coverage.py              # coverage 曲线可视化
│   ├── visualize_online_run.py       # 生成 coverage/KL/diversity 图表
│   └── compare_cooldown.py           # cooldown 策略对比
│
├── tests/
│   ├── test_split.py
│   ├── test_subgraph.py
│   ├── test_labeling.py
│   ├── test_heuristic.py
│   ├── test_edge_split.py            # 23 例
│   ├── test_recall.py                # 21 例
│   ├── test_metrics.py               # 扩展（含 MRR/NDCG）
│   ├── test_curriculum.py            # 16 例
│   ├── test_train.py                 # 7 例
│   ├── test_datasets.py              # 数据集格式验证
│   ├── test_online.py                # 在线仿真集成测试（10 例）
│   ├── test_feedback_probabilistic.py # 6 例
│   ├── test_recall_ppr.py             # 5 例
│   ├── test_recall_mixture.py         # 5 例
│   ├── test_user_selector.py          # 6 例
│   ├── test_env_init_sampling.py      # 7 例
│   ├── test_cooldown_decay.py         # 6 例
│   ├── test_evaluator_metrics.py      # 6 例
│   ├── smoke_model.py                 # 模型前向传播 smoke test
│   ├── bench_optimizations.py         # 优化基准测试套件
│   ├── inspect_subgraphs.py           # 子图调试工具
│   ├── gen_sbm_data.py                # SBM 数据生成辅助
│   └── process_college_msg.py         # CollegeMsg 处理辅助
│
├── results/
│   ├── logs/
│   ├── checkpoints/
│   └── tables/
│
└── docs/
    ├── progress.md
    └── reproducibility.md
```

---

## 两大训练协议（Protocol Dispatch）

`--protocol legacy | simulated_recall`，在 `src/train.py:main()` 中单点门控。

### legacy 协议（保留，用于 baseline 对比）
```
edges → temporal_split → train/val/test edges
  → TimeAdjacency(all edges)
  → LinkPredDataset（random/hard_2hop/historical 负采样）
  → run_epoch → val AUC(hard2hop + historical 均值) → checkpoint
```

### simulated_recall 协议（新，Step 1-8）
```
edges → [filter_first_time_edges] → TwoLayerEdgeSet(E_obs, E_hidden_train, E_hidden_val)
  → TimeAdjacency(E_obs ONLY)  ← 关键：禁止用全量 edges
  → build_recall(CN | AA | union)
  → RecallDataset(E_hidden, recall, cutoff_time, [reciprocity_weights], [difficulty_range])
  → run_epoch（加权 BCE）+ eval_mrr_epoch → val MRR → checkpoint
  → [CurriculumScheduler] 每 epoch 重建 RecallDataset（难度渐进）
```

---

## 模块依赖关系

```
src/train.py
  ├── src/graph/edge_split.py    (TwoLayerEdgeSet, filter_first_time_edges)
  ├── src/graph/subgraph.py      (extract_subgraph, TimeAdjacency)
  ├── src/graph/negative_sampling.py  (legacy 路径)
  ├── src/recall/                (RecallBase, CN/AA, build_recall, CurriculumScheduler)
  ├── src/model/model.py
  │       └── src/model/gin_encoder.py + scorer.py
  ├── src/utils/metrics.py       (AUC/AP/MRR/NDCG/Hits@K)
  ├── src/utils/split.py
  └── src/utils/seed.py

src/dataset/base.py
  └── src/graph/edge_split.py    (filter_first_time_edges，first_time_only 开关)

src/recall/heuristic.py
  └── src/graph/subgraph.py      (TimeAdjacency.out_neighbors — O(log d))
```

---

## 关键函数/类位置索引

| 功能 | 文件:行 |
|------|---------|
| `set_seed(seed)` | `src/utils/seed.py:7` |
| `temporal_split(edges)` | `src/utils/split.py:8` |
| `TemporalDataset`（基类） | `src/dataset/base.py:15` |
| `TimeAdjacency`（时序邻接表） | `src/graph/subgraph.py:25` |
| `TimeAdjacency.out_neighbors(u, t)` | `src/graph/subgraph.py:58` |
| `extract_subgraph(u, v, t, ...)` | `src/graph/subgraph.py:150` |
| `cache_subgraphs(...)` | `src/graph/subgraph.py:326` |
| `load_cached_subgraphs(...)` | `src/graph/subgraph.py:361` |
| `build_adj_out(edges, cutoff)` | `src/graph/negative_sampling.py:30` |
| `sample_negatives(...)` | `src/graph/negative_sampling.py:54` |
| `sample_negatives_mixed(...)` | `src/graph/negative_sampling.py:230` |
| `filter_first_time_edges(edges)` | `src/graph/edge_split.py:32` |
| `TwoLayerEdgeSet`（dataclass） | `src/graph/edge_split.py:49` |
| `temporal_mask_split(edges, ...)` | `src/graph/edge_split.py:66` |
| `random_mask_split(edges, ...)` | `src/graph/edge_split.py:110` |
| `build_two_layer(edges, cfg)` | `src/graph/edge_split.py:164` |
| `compute_reciprocity_labels(edges)` | `src/graph/edge_split.py:189` |
| `RecallBase`（ABC） | `src/recall/base.py:7` |
| `RecallBase.update_graph(round_idx)` | `src/recall/base.py:20` |
| `_two_hop_scores(u, t, time_adj, ...)` | `src/recall/heuristic.py:35` |
| `CommonNeighborsRecall` | `src/recall/heuristic.py:81` |
| `TwoHopRandomRecall` | `src/recall/heuristic.py:142` |
| `AdamicAdarRecall` | `src/recall/heuristic.py:197` |
| `PPRRecall` | `src/recall/ppr.py:1` |
| `PPRRecall.candidates(u, cutoff_time, top_k)` | `src/recall/ppr.py:40` |
| `CommunityRandomRecall` | `src/recall/community.py:1` |
| `MixtureRecall` | `src/recall/mixture.py:1` |
| `MixtureRecall.candidates(u, cutoff_time, top_k)` | `src/recall/mixture.py:30` |
| `build_recall(cfg, adj, n_nodes)` | `src/recall/registry.py:13` |
| `CurriculumScheduler` | `src/recall/curriculum.py:7` |
| `CurriculumScheduler.difficulty(epoch)` | `src/recall/curriculum.py:41` |
| `CurriculumScheduler.top_k_range(epoch, k)` | `src/recall/curriculum.py:53` |
| `_ranks(pos, neg)` | `src/utils/metrics.py:20` |
| `compute_hits_at_k(pos, neg, k)` | `src/utils/metrics.py:37` |
| `compute_mrr(pos, neg)` | `src/utils/metrics.py:56` |
| `compute_ndcg_at_k(pos, neg, k)` | `src/utils/metrics.py:73` |
| `compute_ranking_metrics(scores_by_query, k_list)` | `src/utils/metrics.py:93` |
| `compute_all_metrics(...)` | `src/utils/metrics.py:121` |
| `parse_neg_strategy(s)` | `src/train.py:69` |
| `LinkPredDataset`（legacy） | `src/train.py:87` |
| `RecallDataset`（simulated_recall） | `src/train.py:153` |
| `collate_fn(...) → (bg, labels, qids, weights)` | `src/train.py:239` |
| `run_epoch(...)` | `src/train.py:282` |
| `eval_mrr_epoch(...)` | `src/train.py:355` |
| `_run_simulated_recall(...)` | `src/train.py:419` |
| `main()` | `src/train.py:603` |
| `StaticAdjacency` | `src/online/static_adj.py:14` |
| `OnlineEnv` | `src/online/env.py:16` |
| `OnlineEnv.cooldown_excluded_nodes(u, t)` | `src/online/env.py:242` |
| `OnlineEnv.set_cooldown_mode(mode)` | `src/online/env.py:259` |
| `OnlineEnv.mask_cooldown(u, cands, t)` | `src/online/env.py:274` |
| `OnlineEnv.step(recs, round_idx)` | `src/online/env.py:294` |
| `FeedbackSimulator.simulate(recs)` | `src/online/feedback.py:32` |
| `UserSelector` | `src/online/user_selector.py:1` |
| `UserSelector.select(t, adj)` | `src/online/user_selector.py:50` |
| `UserSelector.update_after_round(t, accepted)` | `src/online/user_selector.py:80` |
| `OnlineTrainer` | `src/online/trainer.py:130` |
| `OnlineTrainer.score(u, candidates, adj)` | `src/online/trainer.py:343` |
| `OnlineTrainer.update(pos, neg, adj)` | `src/online/trainer.py:422` |
| `RoundMetrics` | `src/online/evaluator.py:17` |
| `RoundMetrics.update(...)` | `src/online/evaluator.py:57` |
| `build_scheduler(optimizer, ...)` | `src/online/schedule.py:14` |
| `run_online_simulation(cfg)` | `src/online/loop.py:124` |
| `MLPLinkScorer` | `src/baseline/mlp_link.py:18` |
| `extract_topo_features(adj, n_nodes, ...)` | `src/baseline/mlp_link.py:37` |
| `LinkPredModel(n_nodes, node_emb_dim)` | `src/model/model.py:19` |
| `NodeEmbModel` | `src/model/node_emb_model.py:11` |

---

## 核心数据结构 Schema

### edges.csv
| 列 | 类型 | 含义 |
|----|------|------|
| src | int | 源节点 ID（连续整数，0-based） |
| dst | int | 目标节点 ID |
| timestamp | float | 归一化至 [0,1] |

### nodes.csv
| 列 | 类型 | 含义 |
|----|------|------|
| node_id | int | 节点 ID |
| feat_0..feat_2 | float | 度特征（in_degree, out_degree, total_degree）或原生特征 |

### meta.json
```json
{ "n_nodes": int, "n_edges": int, "has_native_node_feature": bool,
  "feat_dim": int, "t_min": float, "t_max": float, "is_directed": true }
```

### TwoLayerEdgeSet（edge_split.py:49）
```python
@dataclass
class TwoLayerEdgeSet:
    E_obs:          pd.DataFrame  # 观测图（70% 时间或 mask 后剩余）
    E_hidden_val:   pd.DataFrame  # 训练目标正样本（15%）
    E_hidden_test:  pd.DataFrame  # 测试目标正样本（15%）
    cutoff_val:     float         # E_obs 最大时间戳
    cutoff_test:    float         # E_obs ∪ E_hidden_val 最大时间戳
```

### RecallDataset 样本格式
- 无互惠性权重：`(u, v, cutoff_time, label, query_id)` — 5-tuple
- 有互惠性权重：`(u, v, cutoff_time, label, query_id, weight)` — 6-tuple

### DGLGraph 子图
- `g.ndata['feat']`：`(N, feat_dim + label_dim)` — 节点特征 ‖ DRNL 嵌入
- `g.ndata['label']`：`(N,)` int — DRNL 离散标签
- `g.ndata['u_mask']`、`g.ndata['v_mask']`：标识目标节点对

---

## 重要约束与风险点

1. **TimeAdjacency 必须只基于 E_obs 构建**（`_run_simulated_recall:460`）。用全量 edges 是 silent data leakage，训练仍可跑但结果无效。断言在 `train.py:461`。
2. **子图提取必须传 `cutoff_time`**（函数内部断言过滤未来边）。
3. **collate_fn 返回 4-tuple**：`(bg, labels, query_ids_or_None, sample_weights_or_None)`。所有调用方必须解包 4 个值。
4. **Epinions 无原生时间戳**：用行序号作代理时间，`temporal_split` 语义为"前 70% 条边 → E_obs"。

---

## 已知扩展点

1. **召回策略**：`src/recall/registry.py` 新增 `node2vec` / `gcn_single` 只需实现 `RecallBase` 子类
2. **GNN 类型**：`--encoder_type last | layer_concat | layer_sum`
3. **负样本策略**（legacy）：`--neg_strategy random:0.5,hard_2hop:0.3,degree:0.2`
4. **课程调度**：`--curriculum --curriculum_schedule linear|cosine|step`
5. **子图缓存**：`cache_subgraphs()` / `load_cached_subgraphs()` 已实现（`subgraph.py:326`）。小图（n<5000）禁用磁盘缓存（每份 ~12GB），已改用 TimeAdjacency 内存方案；大图（n>50k）条件性启用待评估
6. **GNN + 节点嵌入混合**：`model.node_emb_dim > 0` 时在 GIN 图嵌入后 concat `emb(u) ‖ emb(v)` 再送 Scorer。config 中加 `node_emb_dim: <int>` 即可启用；`_build_flat_batched_graph` 写入 `g.ndata["_node_id"]` 供 lookup
