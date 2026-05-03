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
- [x] **encoder_type 消融**：已完成（见 PROGRESS.md）— last=0.2001, concat=0.2171, sum=0.2176；结论：concat/sum 均优于 last，默认沿用 layer_sum

## Phase 7：分析与可视化

- [ ] `notebooks/attention_visualization.ipynb`：GAT 注意力权重热图
- [ ] `notebooks/case_study.ipynb`：合成数据结构案例分析
- [ ] 按节点度分层的 Hits@K 分析

---

## 性能优化任务

### 离线子图缓存（训练提速）

> ⚠️ superseded（2026-05-02）：小图（n<5000）磁盘缓存已证明不可用（college_msg 单份 12GB，见 MISTAKES.md）。当前方案为 TimeAdjacency 内存方案，性能已足够。大图（n>50k）条件性启用仍可探索，但非当前优先级。

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

---

## 代码/文档审计待办（新增 2026-04-25）

> 来源：本次 review 对 `src/online/*`、`src/recall/*`、`configs/online/*`、`tests/*`、`BLUEPRINT.md`、`PROGRESS.md` 的全面比对。条目按优先级分组。

### A. 代码逻辑错误（必须修，影响实验正确性）✅ 已完成 2026-04-25

- [x] **A1. `loop.py` 未透传 `recall.components`** — 改为透传整个 recall_cfg；registry 加 schema 校验
- [x] **A2. `evaluator.py::Hits@K` 仅生成 `hits@ks[0]`** — 改为循环 self._ks 生成所有 K
- [x] **A3. `tests/test_evaluator_metrics.py::test_rec_coverage_distinct_targets` 断言错误** — 改为 1.0，加 unique_recs@3 辅助断言
- [x] **A4. `registry.py::build_recall("union")` 语义错误** — 改为报 ValueError，提示用 mixture
- [x] **A5. `evaluator.py::MRR` 多正样本求 mean** — 改为标准最佳 rank 倒数
- [x] **A6. `OnlineTrainer._precompute_u_nbrs` 读私有字段** — StaticAdjacency 暴露 out_degree/in_degree/out_neighbors_set/in_neighbors_set

### B. 性能/设计可改进（非阻塞）

- [ ] **B1. `heuristic.py::CN/AA::update_graph` 每轮无条件重建 sparse A**
  - 改：仿 PPR，缓存 `_last_n_edges`，只在边数变化时调 `_build_sparse_adj`

- [ ] **B2. `evaluator._refresh_G_t` 每次重添加全部 `adj.iter_edges()`**
  - 改：维护一个 `_known_edges` 集合，只 add 新增；或缓存 nx.DiGraph 在 `OnlineEnv.add_edge` 钩入增量

- [ ] **B3. `StaticAdjacency.get_csr` 全量 Python loop 重建 CSR**
  - 改：按 `_out` 增量维护排序数组；或在 dirty 重建时用 numpy `np.fromiter` 替 `sorted(set)` 提速

- [ ] **B4. `OnlineEnv.cooldown_excluded_nodes` 对每用户全量扫 `_cooldown` 表**
  - 改：再维护一个 `dict[int, set[int]]`（src→{dst}），cold_start_users 多时省一次 O(|cooldown|)

- [ ] **B5. `loop.py` 中 `extract_topo_features` 在 247、282 行重复 import**
  - 改：放到模块顶部 `try: ... except ImportError`，避免每轮 import overhead

- [ ] **B6. `evaluator.py` 死导入 `compute_hits_at_k / compute_mrr / compute_ndcg_at_k`**
  - 改：删除（实际指标已 inline 实现）；或重构为复用 `src/utils/metrics.py`

### C. 配置一致性（影响实验复现）✅ 已完成 2026-04-25

- [x] **C1. `default.yaml` 与 `college_msg.yaml` feedback.p_accept** — 改为 p_pos/p_neg；加 user_selector、cooldown_mode: decay
- [x] **C2. `college_msg.yaml` 缺字段** — 补全 init_strategy/user_selector/cooldown_mode/cold_start_random_fill
- [x] **C3. hidden_dim 矛盾** — 在 default.yaml/college_msg.yaml 加注释"在线小图用 8/离线大图用 64"
- [x] **C4. FeedbackSimulator p_accept 接口** — 加 DeprecationWarning，p_pos 为主接口

### D. BLUEPRINT.md 大量过期（影响 LLM 检索成本）

- [ ] **D1. 文件:行 索引整体过期**
  - 实测对比：`run_online_simulation` 文档写 `loop.py:58` 实际 `:78`；`OnlineTrainer.score` 写 `:64` 实际 `:342`；`OnlineTrainer.update` 写 `:81` 实际 `:421`；`OnlineEnv.step` 写 `:80` 实际 `:294`
  - 改：跑一次 `/blueprint-update`，重新抓取所有函数定义行号；建议添加一个 hook 脚本周期性校验

- [ ] **D2. 目录树中 `configs/` 重复出现两次（行 25、103）**
  - 改：合并为单棵子树，online 作为 configs/ 子目录列出

- [ ] **D3. `configs/online/` 实际 18 个 yaml，BLUEPRINT 仅列 3 个**
  - 改：补齐 `default.yaml / sbm_smoke.yaml / college_msg_*.yaml(全部) / facebook_ego_exp1.yaml / college_msg_init*.yaml` 等

- [ ] **D4. `src/dataset/real/` 缺新 loader**
  - 改：补 `facebook_ego.py / lastfm_asia.py / ogbl_collab.py / twitch_gamers.py`（PROGRESS 已记录添加）

- [ ] **D5. `scripts/` 列举不全**
  - 改：补 `download_new_datasets.py / preprocess_new_datasets.py / precompute_subgraphs.py / run_encoder_ablation.py / run_hidden_dim_ablation.sh / run_online_sim_win.py / _estimate.py / _profile_round.py`

- [ ] **D6. `src/dataset/synthetic/` 缺 `synth_dataset.py`；`tests/` 缺 `bench_optimizations.py / smoke_model.py / inspect_subgraphs.py / process_college_msg.py / gen_sbm_data.py`**
  - 改：完整列举或在 BLUEPRINT 写"以下文件不进 index：辅助/旧脚本"白名单

- [ ] **D7. BLUEPRINT 注释 `# ← 新（P2-3）` `# ← 新（Step 7）` 等编号引用已废**
  - 改：清除所有 P0/P1/P2/Step N 注释，改为日期标签或删除

- [ ] **D8. "已知扩展点 5. 子图缓存（TODO）" 与 MISTAKES.md [2026-04-15] 子图缓存磁盘爆炸 教训冲突**
  - 改：注明"小图（n<5000）禁用磁盘缓存，已改为 TimeAdjacency 内存方案"；TODO 改为"评估大图启用条件"

### E. PROGRESS.md / TODO.md 状态同步

- [ ] **E1. `PROGRESS.md` 顶部任务"CollegeMsg 全量运行"标 🟡 运行中**
  - 实际：`results/online/college_msg_full/rounds.csv` 已存在且 git status 显示已修改；应改为"已完成 + 结果摘要"

- [ ] **E2. `PROGRESS.md` 早期任务（Phase 4 进行中、Bitcoin-OTC × 3 Encoder 进行中、Scorer 数值爆炸）日期是 2026-04-09 ~ 04-10 状态长期"进行中"**
  - 改：归档至 `docs/progress.md` 历史日志，PROGRESS.md 只留近 2 周活跃任务

- [ ] **E3. `PROGRESS.md` 新数据集接入：epinions / twitch / ogbl_collab 仍标"运行中"或"⏳"**
  - 改：根据 `results/logs/` 的 train.json 更新；缺失就明确写"未启动"

- [ ] **E4. `PROGRESS.md` encoder_type 消融"运行中"无最终结果**
  - 改：补充 last/layer_concat/layer_sum 的 val MRR/Hits@10 三行表

- [ ] **E5. `TODO.md` Phase 6 中 "encoder_type 消融"既在 [ ] 列表又在已完成结果表**
  - 改：删 [ ] 条目，结果汇总放 PROGRESS.md/docs/progress.md

- [ ] **E6. `TODO.md` 性能优化"离线子图缓存（高优先级）"与"训练速度优化策略"DECISIONS / MISTAKES 冲突**
  - 改：标 superseded，改为"按数据集规模条件性启用（n>50k 才缓存）"

### F. MISTAKES / DECISIONS 整理

- [ ] **F1. `MISTAKES.md` 已解决 ✓ 条目（在线精排崩溃 / init_edge_ratio 消融失效）可移到 `MISTAKES_archive.md`**
  - 改：每 ~20 条触发 `/meta-reflect` 后归档，主文件保留"复发可能"警觉信号

- [ ] **F2. `DECISIONS.md` [2026-04-20] 放弃 college_msg legacy 状态写 active，但 [2026-04-21] 整体切到在线仿真**
  - 改：标 `superseded by [2026-04-21] 在线仿真切换`

### G. 其他细节

- [ ] **G1. `recall/heuristic.py` `_two_hop_scores` 命名误导**：函数仅作小图 fallback 路径，但名字暗示通用 2-hop。改名 `_two_hop_scores_fallback` 或在 docstring 注明"仅作 fallback"

- [ ] **G2. `community.py` greedy_modularity_communities 在 75k 节点 Epinions 上 O(E log V)，每 20 轮全量重算开销大**
  - 改：写明"建议在 n>50k 数据集禁用 community_random，或将 recompute_every_n 调大到 100"

- [ ] **G3. `loop.py::_load_dataset` 加载 `college_msg` 后丢弃 timestamp，online 仿真时初始化策略基于随机/分层而非时序**
  - 改：在 docstring 显式注明"在线仿真不消费时间戳，时序顺序仅在 'init_strategy=temporal_prefix'（待加）时使用"

- [ ] **G4. `register.py` 注释列出召回方法少 `common_neighbors` 与 `union`**
  - 改：docstring 与 ValueError 信息保持完整列表

