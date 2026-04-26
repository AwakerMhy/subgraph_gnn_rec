# 项目审查报告:`AwakerMhy/subgraph_gnn_rec`

> 审查日期:2026-04-26
> 审查范围:仓库 `main` 分支全量代码 + 所有 `*.md` 计划文档
> 审查目标:研究目的、进展核对、正确性风险、实验效率问题

---

## 一、研究目的与进展概览

### 核心研究问题

构建一个闭环在线仿真系统,研究:

> **"GNN 链接预测 → Top-K 推荐 → 概率化用户反馈 → 图动态演化 → 模型在线微调"** 这一反馈循环能否推动观测图 $G_t$ 收敛到真实熟人网络 $G^*$ 的统计特性(度分布、聚类系数、覆盖率)。

GNN 精排模型结构:基于 ego-graph + 公共邻居的子图($\{u\} \cup N(u) \cup CN(u,v) \cup \{v\}$)+ GIN 编码器 + MLP 评分头。

### 当前进展(截至 2026-04-25)

- **已完成**:Phase 1–5.5(数据、子图、模型、离线训练、模拟召回框架)
- **已完成**:在线仿真框架 `src/online/*` 打通,并完成关键修复(反馈概率化、cooldown 双模式、Composite 用户选择、Mixture 召回、A 级指标 bug)
- **已完成**:跨数据集对比(9 个真实稀疏有向图)统一采用 `two_hop_random` 召回 + 三 ranker(GNN/MLP/Random)横评
- **已完成**:`hidden_dim` 敏感性、`bfs_2hop` vs `ego_cn` 子图设计对比
- **在跑 / 未完**:
  - CollegeMsg 100 轮仿真(PROGRESS 标"运行中"但 `rounds.csv` 已产出,状态需对齐)
  - epinions / twitch / ogbl-collab 训练
  - `encoder_type` 消融的最终对比表

---

## 二、影响研究正确性的问题(按严重程度排序)

### P0 — 必须修复(正确性 / 可信度)

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| 1 | `src/online/evaluator.py::_refresh_G_t` | `add_edges_from(adj.iter_edges())` 每次把**全量**边再加一次。虽然 networkx 对已存在边幂等,但在 75k 节点 Epinions / 235k ogbl-collab 上每 10 轮 O(E),且通过 `cur_edges == self._G_t_edge_count` 判断时**增量并未真正增量**(整张图被重新 `add_edges_from`,networkx 内部仍做 O(E) 次哈希查询)。TODO.md B2 已记录。 | 维护 `self._known_edges: set`,或在 `OnlineEnv.add_edge` 钩入回调;或 `_refresh_G_t` 内部通过 `adj._n_edges - self._G_t_edge_count` 取**新增**边(需 `StaticAdjacency` 暴露"自上次快照后新边迭代"接口)。 |
| 2 | `src/recall/heuristic.py::CN/AA.update_graph` | 每轮无条件 `_build_sparse_adj`,即使无新边;`PPR.update_graph` 已有 `_last_n_edges` 短路,CN/AA 未同步改造。 | 仿 PPR 加 `if cur_edges == self._last_n_edges: self._cache.clear(); return`。 |
| 3 | `src/recall/community.py::_recompute_communities` | `greedy_modularity_communities` 在 75k Epinions / 235k ogbl-collab 上 O(E log V);**且每次全量 `nx.Graph(); add_edges_from(adj.iter_edges())`**,与大图严重冲突。TODO.md G2 有记录但未改。 | 硬加护栏:`if n > 50_000: recompute_every_n *= 5`;同时 `nx.Graph` 改为从 CSR 直接构造:`G = nx.from_scipy_sparse_array(A + A.T)`。 |
| 4 | `src/online/env.py::_sample_init_edges` (forest_fire 分支) | `self._rng.shuffle(nbrs)` 原地 shuffle **的是 `adj_map.get(node, [])` 引用**——修改了原邻接表顺序。同一初始化调用内部无影响,但 `adj_map` 被外部复用会有副作用。 | `nbrs = list(adj_map.get(node, [])); self._rng.shuffle(nbrs)`。 |
| 5 | `src/online/loop.py:82-83` | `device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")`——用户配 `device: cuda` 但环境无 CUDA 时,**静默 fallback 到 CPU**,导致实验日志与 config 不一致。PROGRESS 中"CPU/CUDA 均通过"可能是 CPU 运行被误标为 CUDA。 | 改为 `if device_str.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("config 要求 CUDA 但不可用")`;若要 fallback,必须打印显著警告并在 `config.json` 中记录实际 device。 |
| 6 | `src/online/loop.py::_load_dataset` (college_msg 分支) | 读取 `edges.csv` 后 `drop_duplicates` **丢弃时间戳**,`init_strategy=stratified/all_covered` 使用随机样本,等价于随机切分初始图。PLAN.md 的实验合法性基础是"观测图随时间推进",若要"从空图演化",时序顺序应保留供 `init_strategy=temporal_prefix` 使用。TODO.md G3。 | 至少在 `_load_dataset` docstring 明确注明"仿真丢弃时间戳";或增加 `init_strategy=temporal_prefix`(按 timestamp 取前 init_n 条作为 G_0)。 |
| 7 | `src/online/env.py::step` cooldown 清理窗口 | 清理窗口 `cutoff = round_idx - 10 * self._cooldown_rounds` 意味着条目需活 50 轮才丢(`cooldown_rounds=5`)。`total_rounds=100` 影响不大,但 ogbl-collab 长跑(1000+ 轮)+ decay 模式下内存膨胀显著。 | `cutoff` 系数从 10 降到 3(e⁻³≈0.05 已足够衰减),并按字典大小自适应触发。 |

### P1 — 显著影响效率 / 结果可信度

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| 8 | `src/recall/heuristic.py::CN/AA.precompute_for_users` (大图路径) | `AA_dense = np.asarray(AA.todense())` 分配 `len(users) × n` 稠密矩阵——n=235k, users=400 时 = 94M × float32 = **376 MB**,且每轮触发。 | 保持稀疏:按行取 top-k(`AA.getrow(i).toarray()` 或直接 `AA[i].data / AA[i].indices` argpartition),避免 dense 化。 |
| 9 | `src/recall/ppr.py::precompute_for_users` | `P = np.zeros((n, m))` 在 n=235k, m=400 时 = 376 MB;且每轮 `max_iter=20` 次矩阵乘法 `T @ P`。 | 分块(每批 50 用户)做 PPR;或换 push-based Bear / FORA(稀疏 PPR)。`np.abs(P_new - P).sum(axis=0).max()` 是 L1 收敛判据,GPU 迁移应换成 Frobenius norm。 |
| 10 | `src/online/evaluator.py::_graph_similarity` | 每 10 轮 `nx.average_clustering(self._G_t)` 是 **O(Σ d²)**,大图瓶颈。 | 对 n>50k 改为抽样聚类系数:`nx.average_clustering(G_t, trials=1000)`(nx 内置抽样参数)。 |
| 11 | `src/online/env.py::cooldown_excluded_nodes` | 对每个冷启动用户 u 全量扫 `self._cooldown.items()`(O(\|cooldown\|))。TODO.md B4。 | 维护 `self._cooldown_by_src: dict[int, dict[int, int]]`,冷启动变为 O(\|cooldown[u]\|)。 |
| 12 | `src/online/static_adj.py::get_csr` | Python 循环 + `sorted(self._out[u])` 每次 dirty 重建——**每 add_edge 都置 dirty**,每轮 `env.step` 后第一次 CSR 访问都要 O(N + E log d) 重建(无增量)。 | 改为增量更新:`add_edge(u, v)` 时维护 per-node 排序数组(`bisect.insort`),或把 dirty 粒度细化为"哪些行脏了",CSR 重建只重建脏行。 |
| 13 | `src/online/loop.py` MLP 路径 `_feat_edge_count` | 赋值仅在 `elif model_type == "mlp":` 分支发生,**下一轮 `_feat_edge_count` 可能引用上轮值**。若训练不走该分支,会与 feat 状态不一致。 | 把 `_feat_edge_count = -1` 在 `for t in range(total_rounds)` **外**初始化,确保每轮一致性。 |
| 14 | `src/online/trainer.py::_build_flat_batched_graph` Phase 3 | `g.ndata["node_feat"] = gathered.pin_memory()` 配 `g.to(self.device)` 触发 async copy,但随后 `forward_batch` 在同一 stream 立即 wait,**pin_memory 收益几乎为零**。 | 要真正重叠需显式 `non_blocking=True` + 独立 stream;或直接把 `node_feat` 常驻 GPU(n_nodes × feat_dim 很小),避免每轮 gather+transfer。 |
| 15 | `src/online/feedback.py::FeedbackSimulator.__init__` | `self._p_pos = p_accept if p_accept is not None else p_pos`——**若同时传 `p_pos=0.8` 和向后兼容的 `p_accept=1.0`,`p_accept` 会覆盖 `p_pos`**,与意图相反。 | 交换优先级:`self._p_pos = p_pos if p_pos != 1.0 or p_accept is None else p_accept`;或直接报错不允许同时传。 |

### P2 — 代码质量 / 维护性

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| 16 | `src/online/evaluator.py:12` | `from src.utils.metrics import compute_hits_at_k, compute_mrr, compute_ndcg_at_k`——**死导入**(实际 inline 实现了 MRR/Hits)。TODO.md B6。 | 直接删除。 |
| 17 | `src/graph/subgraph.py::cache_subgraphs` | MISTAKES.md [2026-04-15] 记录了"小图 cache 磁盘爆炸"教训,但 `cache_subgraphs` 仍暴露原样接口且无警告。 | 函数开头加 `warnings.warn("小图(n<5000)或 college_msg 级数据集请不要启用磁盘缓存,改用 TimeAdjacency 内存方案")` + 自动拒绝。 |
| 18 | BLUEPRINT.md 大量过期 | 文件:行号全部漂移(`OnlineTrainer.update` 写 `:81` 实际 `:421` 等)。TODO D1–D8。 | 跑一次 `blueprint-update` 重新同步,或 CI 加脚本校验。 |
| 19 | `PROGRESS.md` 顶部"CollegeMsg 全量运行"标 🟡 运行中,但 `results/online/college_msg_full/rounds.csv` 已存在 | 状态与事实脱节。TODO E1。 | 对齐状态到"已完成"并附结果摘要。 |
| 20 | `tmp_heuristic.py` / `tmp_process.py` | 临时脚本留根目录(`.gitignore` 未排除),污染仓库。 | 移到 `scripts/` 或删除。 |

---

## 三、实验设计层面需关注的方法学问题

### 1. 召回与精排的信息泄露边界

`DECISIONS.md [2026-04-25]` 统一用 `two_hop_random` 做召回是正确方向(让 ranker 差异完全由 ranker 决定),但论文需要补充分析:

- 在 `top_k_recall=100` 下,正样本在候选池中的命中率是多少?
- 若低于 10%,ranker 之间差距可能被**召回天花板**压制。

**建议**:在 `rounds.csv` 中加记录 `recall_positive_rate`(候选池中 G* 边的比例)。

### 2. 概率反馈 p_pos=0.8 / p_neg=0.02 的敏感性

DECISIONS 记录了这组取值但没有做消融。

**建议**:在 SBM / Triadic 合成数据上跑至少一组网格:
- `p_pos ∈ {0.5, 0.8, 1.0}` × `p_neg ∈ {0.0, 0.02, 0.1}`
- 证明"网络收敛"结论对反馈噪声的稳健性。

### 3. Cooldown 模式切换的实验污染

DECISIONS [2026-04-24] 修了 `hard↔decay` 的数学转换,但从"hard 跑 N 轮→切 decay 再跑 N 轮"流程上,**同一 run 内切换模式本身就改变了探索-利用权衡**。

**建议**:论文明确"单 run 不切换模式",或单独做切换消融。

### 4. G* ground truth 的时间定义

PLAN.md 强调"边时间戳仅用于初始截断图构建",但在线仿真完全丢弃时间戳,把所有边展平成 `star_set`。意味着 G* 是**最终稳态网络**而非某一时刻的观测。对"网络是否收敛到熟人结构"这一研究问题是**合理假设**,但必须在论文 setup 明确。

**建议**:在 PLAN.md / 论文 setup 补充一段 "G* Definition" 明确这个选择。

### 5. 评估指标 `coverage` 和 `precision_k` 的混淆

- `coverage` = accepted ∩ star
- `precision_k` = accepted / recommended

在 `p_neg>0` 模式下,**非 star 边也会被接受**(探索性连接写入 G_t),因此 `precision_k` 不等于"推荐质量"也不等于"最终图质量"。

**建议**:新增两个指标分别记录:
- `precision_k_star` — 仅 star 内接受
- `precision_k_any` — 所有接受

---

## 四、建议的修复优先级

### 立即(本周)

- [ ] **P0 #2** heuristic.update_graph 跳过无变化重建 — 10 行改动,大图性能 5-10×
- [ ] **P0 #5** device fallback 静默问题 — 避免未来实验日志出错
- [ ] **P1 #8** AA todense — 解锁 ogbl-collab 训练
- [ ] **P1 #12** CSR dirty 粒度 — 大图单轮 5-15% 提升
- [ ] **P2 #16, #19, #20** — 清洁工作

### 短期(2-3 周)

- [ ] **P0 #1** `_refresh_G_t` 真增量
- [ ] **P0 #3** community 大图护栏
- [ ] **P0 #6** `init_strategy=temporal_prefix` 或文档明确
- [ ] **P1 #9** PPR 分块
- [ ] **P1 #10** 聚类系数抽样
- [ ] **P1 #13** `_feat_edge_count` 作用域修复
- [ ] 方法学 #1(召回天花板分析)+ #2(反馈敏感性消融)

### 中期

- [ ] BLUEPRINT / PROGRESS 全面同步(TODO.md D/E 分组)
- [ ] 子图缓存条件化启用机制
- [ ] 评估指标 `precision_k_star` / `precision_k_any` 分拆

---

## 五、附:审查覆盖的文件

- 计划文档:`PLAN.md`、`PROGRESS.md`、`BLUEPRINT.md`、`DECISIONS.md`、`MISTAKES.md`、`TODO.md`、`META_REFLECTION.md`、`CHECKPOINT.md`、`SUBGRAPH_COMPARISON.md`、`CLAUDE.md`
- 核心源码:`src/online/{env,loop,trainer,evaluator,feedback,static_adj,user_selector,replay,schedule}.py`
- 召回模块:`src/recall/{heuristic,ppr,community,mixture,registry,curriculum,base}.py`
- 图模块:`src/graph/{subgraph,labeling,edge_split,negative_sampling}.py`

---

> 如需进一步展开某一项(例如直接写修复 patch、或帮忙设计召回天花板 / 反馈敏感性实验),请告知。
