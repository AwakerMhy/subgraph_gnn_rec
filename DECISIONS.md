# DECISIONS — 架构决策记录

> 创建时间：2026-04-08 15:30
> 最后更新：2026-04-24
>
> **做新决策前必读，遇到看似"奇怪"的设计先查这里。**

---

## [2026-04-24] 评估指标设计：G* 正样本 + 截断 MRR@K + 候选池 coverage

- **背景**：发现三个评估逻辑错误：①MRR@K 实为同一值重复 K 次；②正样本用 accepted_set 受 p_pos 噪声污染；③rec_coverage 分母用全图节点数被 sample_ratio 稀释
- **决定**：
  1. **MRR@K**：改为截断 MRR，`mean(1/rank if rank≤k else 0)`，不同 K 有意义差异
  2. **正样本**：排序指标（MRR/Hits）用 `self._star`（G*）做 ground truth，消除 p_pos 采样噪声；precision_k / hit_rate 仍用 accepted_set（衡量真实行为）
  3. **rec_coverage@K**：分母改为候选池去重大小，新增 `unique_recs@K` 原始计数，横比数据集时不被 sample_ratio 稀释
- **原因**：原实现会系统性低估 MRR/Hits（20% 真正样本被当负例），且不同 K 的 MRR@K 完全相同没有区分度
- **状态**：active

---

## [2026-04-24] Cooldown 双模式语义修复

- **背景**：`env._cooldown` 在 hard 模式存 unlock_round，在 decay 模式存 reject_round，同一张表语义不同；三处逻辑均假设 hard 模式导致 decay 模式下冷启动不排除、定期清理恒清空、模式切换丢失历史
- **决定**：
  1. **冷启动排除**：新增 `env.cooldown_excluded_nodes(u, t)` 公开方法，decay 用 `dt < cooldown_rounds`，不再暴露私有属性
  2. **定期清理**：decay 保留 `reject_round > round_idx - 10*cooldown_rounds` 的条目（衰减到 e⁻¹⁰ 后丢弃）
  3. **模式切换**：`hard→decay` 精确转换（`reject = unlock - N`），`decay→hard` 反向（`unlock = reject + N`），不清空表
- **状态**：active

---

## [2026-04-24] 训练速度优化策略

- **背景**：在线仿真每轮时间从 1s 线性增长到 16s（100 轮），主要瓶颈为子图边提取（O(P×d) Python loop）和 CN/AA 召回（190 次 set intersection）
- **决定**：
  1. **边提取**：实现 Numba `@njit parallel` 的 `_count/_fill_edges_batched_nb`；Windows SAC 阻止 DLL 时自动 fallback 到 numpy searchsorted
  2. **CN/AA 召回**：自适应阈值——n≤10k 用 set intersection（小图 scipy overhead 更大），n>10k 用 `A[users]@A` sparse matmul（大图批量摊销）
  3. **N(u) 构建**：`np.union1d` 替代 `sorted(set|set)`，O(d) 归并
  4. **UserSelector degree**：`np.diff(CSR indptr)` 替代 Python list comprehension，避免每轮 O(E) 转换
- **实测（CollegeMsg, 100 轮）**：30 轮后稳定在 ~15s/轮（此前指数增长到 297s/轮）
- **状态**：active；Numba 启用后预计再快 2-3×

---

## [2026-04-22] 子图设计方案：ego-graph + 公共邻居 替代 2-hop BFS

- **背景**：原始设计使用 max_hop=2 BFS 扩展，但 2-hop 邻域在稠密小图中接近全图，无法有效聚焦链路形成的局部结构
- **备选方案**：
  - 方案 A：保持 max_hop=2 BFS（原设计）
  - 方案 B：ego-graph + 公共邻居融合设计（本项目选择）
- **决定**：方案 B，子图组成 = {u} ∪ N(u) ∪ CN(u,v) ∪ {v}，其中：
  - N(u)：请求用户的一度出邻居（取样至 max_neighbors_per_node 个）
  - CN(u,v)：u 和 v 的共同一度出邻居
  - 所有节点和边仅从可观测图 E_obs（t < cutoff_time）中取
  - v 始终包含于子图
- **原因**：
  1. 聚焦假设：2-hop 邻域结构过于复杂，模型难以从中提取有意义的子图编码；ego-graph 只保留 u 的直接影响力圈，公共邻居聚焦 u,v 的共同话题
  2. 经验验证：CollegeMsg 100-sample 1-epoch 烟测结果（simulated_recall 协议）对比：
     | 指标 | 2-hop BFS | ego+CN | 改进 |
     |---|---|---|---|
     | tr_auc | 0.5248 | 0.4866 | ↓1.7% |
     | MRR | 0.0478 | 0.0949 | ↑98% |
     | Hits@10 | 0.0909 | 0.1818 | ↑100% |
  3. 任务适配：链路预测重点在排序（哪个候选最有可能），而非绝对分类；MRR/Hits@K 优先于 tr_auc
- **后果**：`extract_subgraph()` 实现变化（保留 max_hop 参数向后兼容但仅在 1-hop 处理）；后续需对两种设计做系统对比
- **状态**：active，待全 30-epoch 训练验证

---

## [2026-04-21] 概率化反馈模型：p_pos=0.8 / p_neg=0.02

- **背景**：原始实现用 Oracle 反馈（若 (u,v)∈G* 则 100% 接受），相当于把标签直接暴露给模型，reviewer 必定质疑实验合法性
- **备选方案**：
  - 方案 A：继续 Oracle 反馈，仅在论文中注明"理想上界"
  - 方案 B：引入 p_pos/p_neg 伯努利采样，模拟真实用户不确定性（本项目选择）
- **决定**：方案 B，`p_pos=0.8, p_neg=0.02`
- **原因**：p_neg>0 引入的探索性接受（非 G\* 边写入 G_t）更贴近真实系统；p_pos<1 模拟用户决策延迟/犹豫。coverage 统计仍仅计 G\* 命中，不被探索边污染
- **后果**：`FeedbackSimulator` 接受 `p_pos/p_neg` 两参，`p_accept` 作为 `p_pos` 的向后兼容别名；`p_neg=0.0` 时退化为旧行为
- **状态**：active

---

## [2026-04-21] Composite 用户选择策略作为默认

- **背景**：均匀随机抽取活跃用户忽略真实社交平台的幂律活跃度分布；高度节点贡献更多连接机会但被等权对待
- **备选方案**：
  - 方案 A：保持均匀采样（均匀 baseline 更简单但不真实）
  - 方案 B：Composite 策略：Pareto 活跃度 × 度数因子 × 时间衰减 × 事件触发（本项目选择）
- **决定**：方案 B，`strategy=composite`，默认参数 `alpha=0.5, beta=2.0, lam=0.1, gamma=2.0, w=3`
- **原因**：三个因子分别对应不同真实机制：幂律分布捕捉"大用户效应"，时间衰减防止死板的"轮换制"，事件触发模拟"被推荐后活跃度提升"的反馈回路
- **后果**：`UserSelector` 在 `strategy='uniform'` 时完全退化为旧行为（向后兼容）；`OnlineEnv` 集成 selector，不再直接调用 `np.random.choice`
- **状态**：active

---

## [2026-04-21] Mixture 召回 = AA(30) + PPR(10) + Community(10)

- **背景**：纯 AdamicAdar 在孤立节点（G_0 中 2-hop 为空）时永远召回空集；早期轮次 73% 节点 out_degree=0 导致精排无候选
- **备选方案**：
  - 方案 A：只用 AA，依赖 snowball init 降低孤立节点比例
  - 方案 B：Mixture 召回混合 AA + PPR + 社区随机（本项目选择）
- **决定**：方案 B，配额 AA:PPR:Community = 30:10:10
- **原因**：PPR 依赖 random walk 不需要 2-hop 邻居存在；Community 从同社区随机采样绕开度数约束；三者互补保证每个用户至少有候选集
- **后果**：`registry.py` 新增 ppr/community_random/mixture；`loop.py` 在每轮开始调用 `recall.update_graph(t)` 保持图状态同步
- **状态**：active

---

## [2026-04-21] 衰减式 Cooldown 替代硬 Mask

- **背景**：硬 mask 在 cooldown_rounds 内完全屏蔽被拒推荐，导致高活跃用户的候选池枯竭（拒绝记录累积后可选节点数 → 0）
- **备选方案**：
  - 方案 A：保持硬 mask，增加 cooldown_rounds 时长
  - 方案 B：衰减权重：`score × (1 - exp(-Δt / N))`，Δt 越大权重越接近 1（本项目选择）
- **决定**：方案 B，`cooldown_mode=decay`，N=cooldown_rounds=5
- **原因**：软惩罚允许被拒节点逐渐"重新进入"候选，防止候选枯竭同时仍压制短期重复推荐
- **后果**：`_cooldown` 存储拒绝时刻而非解锁时刻；`mask_cooldown` 接受 mode 参数；`hard` 模式仍可通过配置启用（向后兼容）
- **状态**：active

---

## [2026-04-20] 放弃 college_msg 的 legacy 协议实验，专注 bitcoin_otc / email_eu

- **背景**：college_msg 上 GIN-last 和 GraphSAGE 在 8 epoch 后 val_auc_mean 从 ep1 的 0.535 单调下降至 0.52，tr_auc ep1 即达 0.99+，无任何泛化迹象
- **备选方案**：
  - 方案 A：继续跑满 30 epoch，观察是否有后期突破（成本：~2 小时计算）
  - 方案 B：立即停止，将资源转移到 bitcoin_otc / email_eu（本项目选择）
- **决定**：方案 B，放弃 college_msg legacy 协议实验
- **原因**：稠密小图（1899 节点）的结构同质性是根本限制，与训练轮数无关；继续跑只是浪费 GPU 时间。bitcoin_otc（5881 节点，稀疏）和 email_eu（986 节点，但首次边过滤后结构更有判别力）更值得观察
- **后果**：college_msg 在 legacy 协议下无结果；若需要 college_msg 数据点，应换用 simulated_recall 协议
- **状态**：active

---

## [2026-04-17] first_time_only 过滤为可选开关

- **背景**：Step 1 要求对每个 (u,v) 对只保留最早一条边，将任务转为"新链接预测"。但部分数据集（Email-EU 332k→25k）去重后训练信号剧烈萎缩。
- **备选方案**：
  - 方案 A：强制去重（简单，但稀疏数据集可能崩溃）
  - 方案 B：可配置开关 `--first_time_only`，默认关闭（本项目选择）
- **决定**：方案 B，`--first_time_only` flag 默认 False
- **原因**：不同数据集边分布差异大；先验证召回框架有效性，再按数据集按需开启
- **后果**：关闭时 E_hidden 可能包含重复边，正样本含义略宽松（"再次连接"而非"首次连接"）
- **状态**：active

---

## [2026-04-17] src/recall/ 作为独立包，不合入 src/graph/

- **背景**：CN / AA 召回在逻辑上属于"检索"而非"子图提取"，需要决定放在哪个模块。
- **备选方案**：
  - 方案 A：合入 `src/graph/negative_sampling.py`（已有 2-hop 扩展逻辑可复用）
  - 方案 B：独立包 `src/recall/`（本项目选择）
- **决定**：方案 B
- **原因**：召回是独立检索子系统，有自己的 `RecallBase` 抽象；后续切换 Node2Vec / GCN 召回时 `src/graph/` 零改动；`_two_hop_scores` helper 复用了 `TimeAdjacency.out_neighbors` 而非 `negative_sampling` 的内部数据结构，耦合点干净
- **后果**：`src/recall/` 依赖 `TimeAdjacency`（`src/graph/subgraph.py`），不得反向依赖 `src/graph/negative_sampling.py`
- **状态**：active

---

## [2026-04-17] protocol 开关与 legacy 保留策略

- **背景**：引入 simulated_recall 协议，需决定如何与旧 legacy 代码共存。
- **备选方案**：
  - 方案 A：直接替换 legacy 代码路径（代码简洁，但丢失对照基准）
  - 方案 B：`--protocol legacy | simulated_recall` 单点门控，旧路径完整保留（本项目选择）
- **决定**：方案 B，`train.py` 中 `_run_simulated_recall()` 函数封装新协议，`main()` 在进入 TimeAdjacency 构建前分叉
- **原因**：legacy 协议下的 AUC 两极化现象是论文核心对照组，必须可复现；两路协议不共享 TimeAdjacency 实例（关键：新协议的 TimeAdjacency 仅基于 E_obs）
- **后果**：`train.py` 有两套独立训练循环，维护成本上升；但共享 `run_epoch`、`eval_mrr_epoch`、`collate_fn` 等函数
- **状态**：active

---

## [2026-04-08] 使用 DGL 而非 PyG 作为 GNN 框架

- **背景**：需要选择 GNN 框架支持有向子图和批量子图处理
- **备选方案**：
  - PyTorch Geometric (PyG)：社区更大，文档丰富，但有向图支持较复杂
  - DGL：原生支持有向图消息传递，`dgl.reverse()` + `dgl.batch()` 接口简洁
- **决定**：选 DGL >= 2.1
- **原因**：项目核心是有向子图 GNN，DGL 的有向图原生支持和 `dgl.mean_nodes` readout 更适合；且 `dgl.save_graphs` / `dgl.load_graphs` 天然支持离线子图缓存
- **后果**：所有模型代码依赖 DGL API，不可随意切换到 PyG；子图缓存格式绑定 `.bin`
- **状态**：active

---

## [2026-04-08] 时间戳仅用于切分，不作为模型输入

- **背景**：有向边带时间戳 $(u, v, t)$，需决定是否将时间信息引入模型
- **备选方案**：
  - 方案 A：仅用时间戳做数据集切分和截断图构建（本项目选择）
  - 方案 B：TGAT 风格，将时间编码作为边特征输入模型
- **决定**：方案 A
- **原因**：研究重点是子图结构信息的表达能力；时序 GNN（TGAT/TGN）作为上界参考 baseline 单独实现
- **后果**：模型输入中没有边时间戳；时序 baseline 需单独封装
- **状态**：active

---

## [2026-04-08] 节点属性统一接口：真实数据集用度特征占位

- **背景**：CollegeMsg / Bitcoin-OTC / Email-EU 均无原生节点属性
- **备选方案**：
  - 方案 A：无属性时不传节点特征（需要特殊分支代码）
  - 方案 B：统一用度特征 [in_deg, out_deg, total_deg] 占位，`use_node_attr=false` 关闭 AttrEncoder
- **决定**：方案 B
- **原因**：节点特征向量始终存在（用于 TopoEncoder 初始特征拼接），`use_node_attr` 开关只控制 AttrEncoder 是否激活；代码路径统一，无特殊分支
- **后果**：`nodes.csv` 的 feat 列永远非空；`use_node_attr=false` 时 AttrEncoder 不实例化，融合向量中不含 `h^attr`
- **状态**：active

---

## [2026-04-08] 子图提取优先离线预计算缓存

- **背景**：2-hop 子图提取是训练瓶颈，在线提取会严重拖慢训练速度
- **备选方案**：
  - 方案 A：在线提取（每个 batch 实时计算）
  - 方案 B：离线预计算，用 `dgl.save_graphs` 缓存到 `data/processed/<dataset>/subgraphs/`
- **决定**：方案 B，但保留在线提取接口作为 fallback
- **原因**：RTX 3090 显存充足但 CPU 子图提取慢；离线缓存可复用，磁盘空间 20GB 足够
- **后果**：需要预计算脚本；缓存键 = 数据集名 + 截断时刻 + max_hop + max_neighbors_per_node；缓存丢失后需重建
- **状态**：active

---

## [2026-04-09] 负样本排除集改为全时段出边

- **背景**：原策略只排除 `t < t_q` 的出边，导致 `t ≥ t_q` 的未来真实边被标为 label=0（假负样本）
- **备选方案**：
  - 方案 A：保持现状，接受假负样本噪声
  - 方案 B：排除集改为全时段出边（u 在整个数据集中曾连接过的所有节点）
- **决定**：方案 B
- **原因**：假负样本直接污染训练信号，是模型无法泛化的根因之一；全时段排除实现简单，只需预计算一次
- **后果**：`sample_negatives` 新增 `all_time_adj_out` 参数；`LinkPredDataset` 需传入全时段邻接表；负样本候选池略缩小（高出度节点影响有限）
- **状态**：active

---

## [2026-04-15] sample_negatives 中 rng seed 固定为调用参数

- **背景**：`sample_negatives` 内部每次调用都以传入的 `seed` 参数重建 `np.random.default_rng(seed)`；默认值为 42，调用方通常不改变
- **备选方案**：
  - 方案 A：seed 固定（当前实现）——每次调用产生相同随机序列，批量调用时不同样本的负样本高度相关
  - 方案 B：调用方传入随机状态（`rng` 对象）并跨调用共享——无相关性，但接口改动较大
- **决定**：保持方案 A，已知权衡
- **原因**：当前阶段批量负样本数量有限（k=1~5），相关性影响可忽略；接口简单；`run_comparison.py` 已在外层设 global seed 保证可复现性
- **后果**：若未来 k 很大或采用重要性采样，需换为方案 B；此问题已记录，不视为 bug
- **状态**：active

---

## [2026-04-15] TGAT baseline 后续不投入

- **背景**：对比实验中运行 TGAT（时序 GNN baseline），耗时是 GIN/GraphSAGE 的 20-30 倍，收敛困难
- **实验结果**（college_msg smoke test，3 epoch）：
  - GIN-last：val_auc 0.90，13s/epoch
  - SEAL：val_auc 0.48，190s/epoch
  - TGAT：未完成（kill），预估 300-500s/epoch
- **根因分析**：
  - TGAT 的核心优势是时序编码，但在**固定截断子图**框架下无优势（每个样本已经是静态切片）
  - DGL 消息传递在每条边计算注意力，而 GIN 用优化的内置聚合函数
  - 原论文针对完整动态图 node classification，不适合子图 link prediction
- **决定**：后续实验**排除 TGAT**
- **原因**：投入产出比太低，不如改进 GIN 的结构判别力（hard_2hop 对抗能力差）
- **后果**：baseline 从 4 个模型降为 3 个（GIN / GraphSAGE / SEAL）；如后续需要时序信息，应设计新方案而非依赖 TGAT
- **状态**：active

---

## [2026-04-16] SEAL 在稠密小图上 DRNL 失效——后续不作为主要 baseline

- **背景**：对比实验中 SEAL 在 college_msg 上 val_auc ≈ 0.47（接近随机），而 GIN/GraphSAGE 均 > 0.99
- **诊断**：
  - college_msg 2-hop 子图规模约 200 节点（占全图 10%+），子图极为稠密（5000+ 条边）
  - DRNL 标签分布：正负样本子图中 **0% 的节点 label=0**，正负子图 DRNL 分布几乎相同
  - 根因：DRNL 依赖"u 和 v 距离远时大量节点不可达 → label=0"来区分正负样本；稠密图中所有节点均可到达 u/v，该信号消失
- **备选方案**：
  - 方案 A：保留 SEAL，接受其在稠密图上的失效作为对比数据点
  - 方案 B：移除 SEAL，改用其他 baseline
- **决定**：方案 A——保留 SEAL 参与对比实验，但**在论文分析中明确标注其失效原因**，不作为主要竞争对手
- **原因**：失效本身是有意义的研究发现，说明基于 DRNL 的结构标签方法在稠密社交图上存在根本性局限；GIN/GraphSAGE 的相对优势得以凸显
- **后果**：最终结果表格中 SEAL 数据保留但加注解；TGAT 已因时间成本排除，SEAL 因稠密图失效记录；两者均作为对比反例
- **状态**：active

---

## [2026-04-16] 训练负样本：混合策略 random:0.5 + hard_2hop:0.3 + degree:0.2

- **背景**：单一 random 策略训练信号太弱；纯 hard_2hop 候选池有限且噪声大
- **决定**：混合策略，各策略权重 random:0.5 / hard_2hop:0.3 / degree:0.2
- **原因**：
  - random 提供稳定基础梯度；hard_2hop 提供结构判别压力；degree 引入高度节点偏置对抗
  - 混合避免单策略的极端性（纯 hard_2hop 在稠密图上退化，纯 random 信号太弱）
- **后果**：`--neg_strategy` 参数默认值 = `random:0.5,hard_2hop:0.3,degree:0.2`；hard_2hop 使用 time_adj 精确截断图（已修复全时段 fallback bug）
- **状态**：active

---

## [2026-04-16] 验证评估策略：去掉 random，只保留 hard_2hop + historical

- **背景**：实验发现 random AUC ≈ 0.999（负样本大多不在 2-hop 子图内，模型只需判断 v 是否出现在子图中，无实质挑战）；historical 退化为 random（候选池为空）
- **根因分析**：
  - random：负样本 ~60% 不在 2-hop 内，正样本在 2-hop 内 → 区分trivial
  - historical：候选 = 历史出边 - all_time_adj_out = 空集（历史出边 ⊆ 全时段出边）→ fallback 为 random
- **决定**：
  1. 去掉 random 验证路，只保留 hard_2hop + historical
  2. checkpoint 指标改为 `(val_auc_hard2hop + val_auc_historical) / 2`
  3. historical 候选池修复：仅排除自环，不排除 all_time_adj_out
- **修复后语义**：
  - hard_2hop 负：在 2-hop 内无直接历史边（结构接近但不会连接）
  - historical 负：有直接历史出边（曾连接但不会续期）
  - 正样本：无直接历史边，可能在 2-hop 内（新连接）
- **后果**：两路评估互补；hard_2hop AUC 是结构判别力下界，historical AUC 测试续期识别能力；historical 候选池大小 ~17-40（与节点活跃度相关）
- **状态**：active

---

## [2026-04-17] 负样本难度两极化问题——已知未解

- **背景**：系统性实验后发现，所有已试策略都陷入"太简单 AUC≈0.99"或"太难 AUC≈0.50"的两极，缺少有效中间区间

- **各策略实验结果汇总**（数据集：email_eu, college_msg；模型：GIN-last, GraphSAGE）：

  | 策略 | AUC 区间 | 问题 |
  |---|---|---|
  | random | ~0.999 | 太简单：负样本 ~60% 不在 2-hop 内，正样本几乎都在 2-hop 内，模型只需判断"v 是否在子图里" |
  | historical（旧/有 bug） | ~0.994 | 退化为 random：候选池被 all_time_adj_out 过滤后为空，fallback 到 random |
  | historical（修复后） | ~0.50 | 太难：负样本是 u 的历史邻居，17% 会在未来重连（固有假负），模型无法区分"新连接"与"历史续期" |
  | hard_2hop（修复后） | ~0.51–0.55 | 太难：2-hop 邻居与正样本在结构上几乎无区别，模型缺乏足够的判别特征 |
  | 混合（random:0.5 + hard_2hop:0.3 + degree:0.2） | — | 训练策略，验证路径仍依赖上述策略 |

- **根因分析**：
  - **random 太简单**：2-hop 子图覆盖率低（college_msg ~10%，email_eu 更低），正负样本子图覆盖状态本身就是强信号
  - **hard_2hop 太难**：在密集社交图中，2-hop 内几乎所有节点都有相似的结构邻域；子图 GNN 无法区分"2-hop 内会连接的节点"与"2-hop 内不会连接的节点"
  - **historical 太难**：正样本语义 = "新连接"，负样本语义 = "历史邻居（不续期）"；二者子图结构可能完全相同，区分依赖时序/社交动态而非结构

- **潜在改进方向（未实施）**：
  1. **1-hop 负样本**：u 的1-hop邻居中不形成正样本的节点，比 2-hop 难但比 random 简单——候选池极小
  2. **度数分段**：对高度节点单独采样，避免模型只学"度数大的节点是负样本"偏置
  3. **子图相似度过滤**：预计算候选节点与正样本的子图 Jaccard 相似度，只保留相似度在区间 [0.3, 0.7] 的节点作为负样本——计算代价高
  4. **重新审视任务定义**：AUC≈0.50 可能揭示子图 GNN 在稠密社交图上的根本限制，而非负样本策略问题

- **当前状态**：问题已知，暂无可行的"中等难度"策略；继续用 hard_2hop + historical 作为评估路径，接受 AUC 偏低，关注模型间相对差异
- **状态**：active（待解决）

---

## [2026-04-08] 时间感知切分比例 70/15/15

- **背景**：需要按时间顺序切分数据集，防止数据泄露
- **备选方案**：随机切分（会泄露未来信息）vs 时间分位数切分
- **决定**：按时间戳分位数切分，train 70% / val 15% / test 15%
- **原因**：链接预测任务天然是时序预测，随机切分会造成标签泄露；分位数切分最常见也最公平
- **后果**：`split.py` 必须含断言检查，确保 val/test 中无训练集时间范围内的边
- **状态**：active
