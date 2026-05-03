# docs/progress.md — 历史变更日志

> 创建时间：2026-04-08 15:30
> 最后更新：2026-04-29 (2)

[2026-04-27 17:30] [算法对比实验 + ε-greedy 扫描（4数据集，init=0.25/0.30）] [scripts/run_algo_sweep.py, scripts/run_eps_sweep.py, results/online/algo_sweep*, results/online/eps_sweep] [完成]
- 11种算法（ground_truth/gnn/gnn_sum/gnn_concat/node_emb/mlp/cn/aa/jaccard/pa/random）× 4数据集对比
- 核心发现1：无探索的GNN在coverage上输给random，原因是集中推荐高分节点触发cooldown
- 核心发现2：GNN-sum排序质量（hits@1/MRR@1）远优于random（email_eu高出72%），但需要ε-greedy才能转化为coverage优势
- 核心发现3：ε-greedy GNN-sum 在3/4数据集超过random（email_eu +0.033），ε=0.1~0.3为推荐配置
- 核心发现4：瓶颈在召回而非精排——ground_truth与random差距在多数数据集仅0.01~0.06
- init=0.30比init=0.25全面提升约0.04~0.10，仍在有效区间内
- gnn_sum全面优于gnn_last和gnn_concat，后续默认encoder_type: layer_sum

[2026-04-27 16:00] [新增 ground_truth 精排模型（召回上界基线）] [src/online/loop.py] [完成]
- model.type: ground_truth：从召回候选中挑出 G* 真实边推荐
- m >= k：随机选 k 条真实边；m < k：全部真实边 + 随机补非真实边至 k
- 无需模型/优化器，训练阶段自动跳过（归入 _NO_MODEL_TYPES）

[2026-04-27 15:00] [新增 wiki_vote / slashdot 数据集 loader 及预处理] [src/dataset/real/wiki_vote.py(+新建), src/dataset/real/slashdot.py(+新建), data/processed/wiki_vote/, data/processed/slashdot/, BLUEPRINT.md] [完成]
- Enron (recip=1.000) 排除，不满足有向稀疏条件
- wiki-Vote: 7115 节点, 103689 边, recip=0.056, deg_mean=17.0；行序号作代理时间
- Slashdot: 77350 节点, 516575 边, recip=0.186, deg_mean=11.9；sign 列丢弃，行序号作代理时间

[2026-04-27 15:30] [新增 email_euall 数据集 loader 及预处理] [src/dataset/real/email_euall.py(+新建), data/processed/email_euall/, BLUEPRINT.md] [完成]
- email-EuAll: 265009 节点, 418956 边, recip=0.260, deg_mean=1.9；行序号作代理时间

[2026-04-27 16:00] [更新 docs/datasets.md：补充全部数据集 recip/deg_mean 字段，新增 wiki_vote/slashdot/email_euall 条目] [docs/datasets.md] [完成]
[2026-04-27 16:30] [新增 DC-SBM 合成数据集生成器] [src/dataset/synthetic/dcsbm.py(+新建), BLUEPRINT.md] [完成]
- P(i→j) ∝ θ_out[i]·θ_in[j]·B[c_i,c_j]，θ 服从 Pareto(α) 产生幂律度分布
- 节点特征：社区 one-hot + log(θ_out) + log(θ_in)，共 n_communities+2 维

---

[2026-04-15] [节点属性 concat：GIN/GraphSAGE/SEAL 加 AttrEncoder] [src/model/encoder_attr.py(+新建), src/model/model.py(+node_feat_dim+attr_encoder), src/baseline/graphsage.py(+node_feat_dim+attr_encoder), src/baseline/seal.py(+node_feat_dim+attr_encoder), src/graph/subgraph.py(+node_feat参数), src/train.py(+加载nodes.csv+传node_feat)] [完成]
- 三个模型 Scorer 输入维度 64→128（GNN embedding 64 + attr embedding 64）
- node_feat_dim=0 时 attr_encoder=None，向后兼容

[2026-04-15] [smoke test 对比实验：GIN vs GraphSAGE vs SEAL vs TGAT] [scripts/run_comparison.py, results/logs/cmp_smoke_mixed.log] [完成]
- college_msg 500样本，10 epoch（GIN/GraphSAGE）/ 7 epoch（SEAL，早停）
- 结果：GIN-last 综合最强（val_rand 0.9041 / val_hard2h 0.4141 / val_hist 0.9041）
- 发现 TGAT 时间成本过高（预估 300-500s/epoch vs GIN 13-15s/epoch），时序优势在子图框架下无体现
- 决策：后续不投入 TGAT，focus 改进 GIN 结构判别能力

[2026-04-16] [Apr-15 新版模型全量对比实验启动] [scripts/run_comparison.py, results/logs/comparison_apr15.log] [运行中]
- 触发原因：Apr-15 21:30 更新 model.py（encoder_type/AttrEncoder）和 train.py，旧结果全部过期（生成于 21:30 之前）
- 配置：3 数据集（college_msg / bitcoin_otc / email_eu）× 3 模型（GIN-last / GraphSAGE / SEAL）= 9 runs
- 参数：epochs=30，full dataset（max_samples=0），neg=random:0.5+hard_2hop:0.3+degree:0.2，max_workers=2
- 预计耗时：7~10 小时（college_msg 每 run ~90~120 min）
- 后台 PID：16588（GIN-last）、16200（GraphSAGE）并行，主进程 12516

[2026-04-15] [混合负样本策略 + 三路验证评估] [src/graph/negative_sampling.py(+sample_negatives_mixed), src/train.py(+parse_neg_strategy, LinkPredDataset+strategy_mix, 三路val_loader)] [完成]
- 训练默认改为 random:0.5+hard_2hop:0.3+degree:0.2 混合策略
- 验证改为 random/hard_2hop/historical 三路独立评估，分别报 AUC/AP
- checkpoint 以 val_auc_random 为基准，train.json 记录全部三路指标

[2026-04-15] [TimeAdjacency 在线快路径优化] [src/graph/subgraph.py, src/train.py, scripts/run_comparison.py] [完成]
- 放弃 DGL binary 离线缓存方案（12GB/split，不可行）
- 新增 TimeAdjacency 类：预构建时序邻接表，per-sample cutoff 二分查找，O(log degree)
- extract_subgraph 新增 time_adj 参数作为推荐快路径
- smoke test：每 epoch 从 ~40s → ~2s（college_msg，500样本），加速 20×
- 正式对比实验已启动（3 datasets × 4 models，30 epochs，ID: bcwozkayu）

---

[2026-04-08 15:30] [项目初始化] [CLAUDE.md, BLUEPRINT.md, DECISIONS.md, MISTAKES.md, PROGRESS.md, TODO.md, CHECKPOINT.md, META_REFLECTION.md, docs/progress.md, docs/reproducibility.md, .claude/settings.json, .claude/commands/*, .claude/hooks/*, requirements.txt, configs/default.yaml, .gitignore, src/utils/seed.py] [完成]
[2026-04-09] [GIN 多变体 encoder] [src/model/gin_encoder.py(+GINEncoderLayerConcat+GINEncoderLayerSum), src/model/model.py(+encoder_type 参数)] [完成]
[2026-04-14 00:00] [新增方法对比分析文档] [docs/method_comparison.md(+新建)] [完成]
[2026-04-09 12:00] [Phase 4 收尾：heuristic baseline + CollegeMsg 端到端] [tests/test_heuristic.py, src/graph/negative_sampling.py(+build_adj_out+拒绝采样), src/graph/subgraph.py(+build_graph_adj+prebuilt_adj快路径), src/train.py(+prebuilt_adj+max_samples+flush日志), src/evaluate.py(+prebuilt_adj+批量Hits@K+max_test_samples)] [完成]
[2026-04-20] [在线学习仿真框架 src/online/] [src/online/__init__.py+static_adj.py+env.py+feedback.py+trainer.py+replay.py+evaluator.py+schedule.py+loop.py(全部新建), configs/online/default.yaml+sbm_smoke.yaml+college_msg.yaml(新建), scripts/run_online_sim.py(新建), tests/test_online.py(新建)] [完成 — 10/10 测试通过，sbm_smoke 端到端 3 轮无异常]
- 抛弃时间戳，全图 G* 作为 ground truth；随机采样 5% 初始化 G_0
- 每轮：用户采样→召回→精排→反馈→在线梯度更新→图演化
- StaticAdjacency duck-type TimeAdjacency，零改动 extract_subgraph
- 评估：Precision/MRR + coverage + 聚类系数 + degree KL

[2026-04-20 00:37] [重启 bitcoin_otc+email_eu 对比实验] [scripts/run_comparison.py, results/logs/otc_eu_rerun_apr20.log] [运行中]
- college_msg 实验主动放弃：6 epoch 后 val_auc_mean 单调下降（0.535→0.52），tr_auc≈0.99 严重过拟合，稠密小图结构同质性导致无改善空间
- 当前跑 bitcoin_otc × GIN-last/GraphSAGE（并行），email_eu 排队
- TODO.md Phase 1-5 全部标记为已完成，新增 Phase 5.5（模拟召回框架）

[2026-04-17 18:00] [两层图+模拟召回框架 Step1-6] [src/graph/edge_split.py(新建), src/recall/__init__.py+base.py+heuristic.py+registry.py(新建), src/utils/metrics.py(+MRR+NDCG+compute_ranking_metrics), src/train.py(+RecallDataset+eval_mrr_epoch+_run_simulated_recall+5个CLI参数), src/dataset/base.py(+first_time_only), configs/default.yaml(+protocol/recall/eval块), tests/test_edge_split.py+test_recall.py(新建), tests/test_metrics.py(扩展)] [完成 — CollegeMsg端到端MRR=0.0996>>0.01, 97例测试全通过]

[2026-04-25] [在线仿真性能优化：update() 改用批量路径，25× 提速] [src/online/trainer.py(update()改用_build_flat_batched_graph快速路径+_u_nbrs_cache跨轮缓存), configs/online/college_msg_full.yaml(score_chunk_size 128→512)] [完成 — 每轮14-16s→0.2-0.7s，100轮总耗时~25min→~1min，指标数值不变]

[2026-04-21 00:00] [综合改进 P0+P1+P2 全落地] [src/online/feedback.py, src/online/env.py, src/online/user_selector.py(新建), src/online/evaluator.py, src/online/loop.py, src/recall/base.py, src/recall/ppr.py(新建), src/recall/community.py(新建), src/recall/mixture.py(新建), src/recall/registry.py, src/baseline/mlp_link.py(新建), scripts/visualize_online_run.py(新建), scripts/run_ablation_grid.py(新建), configs/online/college_msg_full.yaml(新建), configs/online/college_msg_no_replay.yaml(新建), tests/test_feedback_probabilistic.py+test_recall_ppr.py+test_recall_mixture.py+test_user_selector.py+test_env_init_sampling.py+test_cooldown_decay.py+test_evaluator_metrics.py(新建)] [完成 — 173例测试全通过，5轮烟测 coverage 0.069→0.076，precision@K 4.2%]

[2026-04-25] [专家 review：修复 A 组代码逻辑错误 + C 组配置一致性] [src/online/loop.py(recall_cfg 整体透传), src/recall/registry.py(union→报错, mixture schema 校验), src/online/evaluator.py(Hits@K 循环所有 K, MRR 改标准最佳 rank), tests/test_evaluator_metrics.py(rec_coverage 断言更新), src/online/static_adj.py(out/in_degree 公共方法), src/online/trainer.py(改用公共方法), src/online/feedback.py(p_accept DeprecationWarning), configs/online/default.yaml+college_msg.yaml(p_pos/p_neg/cooldown_mode/user_selector 补全)] [完成 — 6项 A 组全验证通过，4项 C 组配置对齐]

[2026-04-25] [在线仿真扩展数据集：email_eu + sx_mathoverflow 100轮测试] [configs/online/email_eu_full.yaml(新建), configs/online/sx_mathoverflow_full.yaml(新建), results/online/email_eu_full/(新建), results/online/sx_mathoverflow_full/(新建)] [完成]
- email_eu (986节点/24,929边)：MRR@10 mean=0.328，coverage末轮0.184，每轮~0.2s
- sx_mathoverflow (24,759节点/390,441边)：MRR@10 mean=0.331（67/100轮有效），coverage末轮0.074，每轮~4s
- bitcoin_otc (5,881节点/35,592边，上轮已完成)：MRR@10 ~0.02（有向信任图结构较难，评估触发轮次少）
- 四数据集横向对比：email_eu/sx_mathoverflow MRR@10≈0.33，与 college_msg 接近；bitcoin_otc 明显偏低
- sx_mathoverflow coverage 100轮仅7.4%，图太大需增加 total_rounds 或 sample_ratio 才能充分覆盖

[2026-04-25] [弱项分析 + cyclic LR：bitcoin_alpha GNN vs random 对比实验] [src/online/schedule.py(+cyclic策略), src/online/loop.py(透传cycle_rounds), configs/online/bitcoin_alpha_weak_recall.yaml(新建), configs/online/bitcoin_alpha_weak_recall_random.yaml(新建), configs/online/bitcoin_alpha_cyclic_lr.yaml(新建)] [完成]
- 根因分析：高互惠率(83%)图上AA召回已隐式覆盖互惠对，GNN精排无增量信号；cosine decay后期lr→1e-5导致模型停止更新，无法适应distribution shift
- 弱召回(PPR+community，去掉AA)验证：GNN mean 0.305→0.338，early已超random，但late仍崩塌(0.234)
- cyclic LR(周期25轮)验证：late 0.234→0.332(+42%)，mean 0.338→0.365，首次在bitcoin_alpha上GNN整体超过random
- 结论：弱召回+cyclic lr 是高互惠稀疏图的推荐配置；distribution shift是核心问题，不是模型结构

[2026-04-25] [多数据集系统性实验：GNN vs MLP vs random，召回策略与LR调度消融] [configs/online/*_cyclic/*.yaml(批量新建), configs/online/*_mlp.yaml(批量新建), src/online/schedule.py(+cyclic策略), src/online/loop.py(透传cycle_rounds), scripts/plot_coverage.py(新建), results/online/coverage_trend.png+mrr_trend.png] [完成]

核心发现汇总：

1. 数据集筛选规则
   - 无向图（recip≥0.95）不适合测：AA召回已完全覆盖，GNN无增量价值（facebook_ego/lastfm_asia验证）
   - 稠密图（deg_mean≥20）不适合测：子图高度重叠，GNN判别信号消失（facebook_ego deg=43.7验证）
   - 推荐数据集：bitcoin_alpha/otc、epinions、sx_askubuntu/superuser（有向+稀疏）

2. 召回策略影响
   - 强召回（AA+PPR+community）使GNN精排无增量价值，因为AA已隐式覆盖互惠对
   - 弱召回（PPR+community，去掉AA）让GNN mean MRR: 0.305→0.338，early首次超过random
   - 结论：弱召回才能给GNN留下可学习的结构判别空间

3. Cyclic LR vs Cosine Decay
   - cosine decay后期lr→1e-5，模型后30轮基本停止更新，无法应对distribution shift
   - cyclic LR（周期25轮）：bitcoin_alpha late MRR 0.234→0.332(+42%)，mean首次超过random
   - 弱召回+cyclic LR是高互惠稀疏图的推荐配置

4. GNN vs MLP vs random 三路对比规律
   - GNN early几乎总是领先random（5/5有对照数据集，唯一例外bitcoin_otc因用了强召回）
   - MLP后期稳定性最差（late崩塌最严重），度特征随图演化漂移剧烈
   - GNN整体最稳定，epinions上mean MRR: GNN=0.439 > random=0.432 > MLP=0.391

5. Coverage趋势
   - 三种模型coverage曲线几乎重叠，覆盖率由用户选择策略决定，与模型无关
   - epinions/sx_askubuntu 100轮coverage仅10-19%，大图需要更多轮才能充分覆盖

6. 核心瓶颈：distribution shift
   - GNN在图稀疏时（early）结构判别能力最强，随图变稠密训练分布漂移，优势收窄
   - 假设：更低的init_edge_ratio可延长稀疏阶段，对GNN更有利（待验证）

[2026-04-26] [反馈参数调整：p_pos=0.95, p_neg=0.0] [configs/online/*.yaml(全部批量更新), scripts/gen_thr_grid_configs.py, DECISIONS.md] [完成]
- 将 p_pos 从 0.8 提升至 0.95，p_neg 从 0.02 降至 0.0
- 目的：消除假正例写入 G_t（p_neg=0.0），强化真实关系推荐信号（p_pos=0.95）
- DECISIONS.md 标记旧决策 superseded，新增 [2026-04-26] ADR

[2026-04-26] [PPR 补充召回：解决 coverage 停滞] [src/recall/ppr.py(+PPRNodesRecall), src/recall/registry.py(+ppr_nodes分支), configs/online/*_thr_*.yaml(27个，召回改为mixture)] [完成]
- 问题：p_neg=0.0 下 G_t 增长缓慢，pure two_hop_random 2-hop 候选很快耗尽，coverage 停滞
- 方案：MixtureRecall = two_hop_random(quota=70) + PPRNodesRecall(quota=30)
  - PPRNodesRecall 继承 PPRRecall 但将所有候选分数置 0.0，保持对 ranker 的中立性（不泄露 PPR 排序信息）
- 效果：college_msg cov_gain ~18%→~45%（对齐 random 水平），bitcoin_alpha 类似提升
- 验证：大数据集（sx_askubuntu/sx_superuser/epinions）同步测试中

[2026-04-26] [init_edge_ratio 消融：0.1 / 0.2] [configs/online/*_init1_*.yaml(6个新建), configs/online/*_init2_*.yaml(6个新建)] [完成，发现 bug]
- college_msg init=0.1/0.2 均有效区分，init=0.2 覆盖更多初始边，GNN 早期优势更明显
- **Bug**：bitcoin_alpha init=0.1 与 init=0.05 结果完全相同（stratified init 有 floor=n_unique_sources 限制，init_n < n_unique_sources 时两者等价）
- 结论：bitcoin_alpha 需用 init=0.2 才能进行有效消融；init=0.1 档在 bitcoin_alpha 上数据无效

[2026-04-26] [hidden_dim 消融：h16 / h32 / h64（init=0.2 base）] [configs/online/*_init2_gnn_h{16,32,64}.yaml(6个新建)] [完成]
- college_msg: h64 mrr3_mean=0.320 > h32=0.310 > h16=0.295，mrr 随 hidden_dim 单调提升
- bitcoin_alpha: h64 mrr3_mean=0.369 > h32=0.341 > h16=0.312，趋势一致
- **负效应**：h64 的 cov_gain 反而最低（college_msg 17.7%），大模型倾向集中推荐高分节点，探索不足
- Random 在 mrr3_mean 上仍领先所有 GNN 配置（college_msg 0.407，bitcoin_alpha 0.417）
- 假设：问题来自 cyclic LR 的周期性重置破坏已学知识，或学习率绝对值不当

[2026-04-26] [LR 消融启动：cyclic schedule, lr ∈ {0.0001, 0.0005, 0.001, 0.005} × {college_msg, bitcoin_alpha}] [configs/online/*_init2_gnn_h64_lr*.yaml(6个新建，0.001已有baseline)] [进行中]
- college_msg lr=0.0001(cyclic) 已完成：mrr3_mean=0.324（vs baseline 0.001→0.320），cov_gain=31.9%（vs 17.7%）
- 低 lr 缩小了与 Random 的 cov_gain 差距，但 mrr 仍低于 Random(0.407)
- 其余 5 个配置待运行

[2026-04-26] [constant LR schedule 对比实验启动] [configs/online/*_init2_gnn_h64_const_lr*.yaml(8个新建)] [待运行]
- 假设：cyclic schedule 的周期性重置（每 25 轮 lr 从 base 跌至 min_lr=1e-5 再重置）破坏已学排序偏好，constant schedule 更稳定
- 覆盖 lr ∈ {0.0001, 0.0005, 0.001, 0.005} × {college_msg, bitcoin_alpha}，与 cyclic 形成完整对比矩阵

[2026-04-26] [constant LR + LR 消融结果] [results/online/college_msg_init2_gnn_h64_const_lr*.yaml] [完成]
- constant schedule 整体优于 cyclic（college_msg: const lr=0.001 → mrr3=0.335 vs cyclic→0.320）
- 最优 cyclic：lr=0.0001(mrr=0.324)；最优 constant：lr=0.001(mrr=0.335)
- GNN 在所有 LR/schedule 配置下仍弱于 Random(0.407)，根因未解决

[2026-04-26] [假负例根因实验：p_pos 1.0 + top_k sweep] [configs/online/*_ppos1.yaml, *_topk*.yaml] [完成，重大发现]
- **p_pos=1.0（消除假负例）**：college_msg mrr3 0.335→0.422，首次超越 Random(0.407)
- **top_k 增大（p_pos=0.95）**：mrr 急剧下降（topk10→0.177，topk20→0.150）——top_k 越大假负例越多
- 结论：假负例是 college_msg GNN 弱于 Random 的主因；top_k 不是解法

[2026-04-26] [college_msg top_k × ppos=1.0 完整 tradeoff 曲线] [configs/online/college_msg_init2_gnn_h64_topk*_ppos1.yaml] [完成]
- top_k=3: mrr=0.665(+63% vs Random), cov=17.8%
- top_k=5: mrr=0.422(+4%), cov=24.3%
- top_k=10: mrr=0.239, cov=29.7%
- top_k=20: mrr=0.128≈Random, cov=44.7%≈Random
- 最优配置：topk=3 + ppos=1.0，mrr 远超 Random，代价是 coverage 低

[2026-04-26] [recall 精度诊断 + 跨数据集系统性实验] [configs/online/{email_eu,dnc_email}_init2_*.yaml] [完成]
- recall_prec: college_msg 3.0%, bitcoin_alpha 2.0%, dnc_email 2.9%, email_eu 17.9%
- email_eu recall_prec=17.9%：random 探索即可覆盖100% G*，GNN 加不了值（任务退化）
- dnc_email：GNN ppos1 mrr=0.435 > Random 0.397（+9.5%）✓
- bitcoin_alpha：MLP ppos1≈MLP ppos0.95（0.344 vs 0.345），消除假负例无效 → 结构不匹配

[2026-04-26] [三类失效模式完整识别] [docs/progress.md, DECISIONS.md] [完成]
- 类型1 假负例毒害（college_msg）→ ppos=1.0 修复
- 类型2 结构不匹配（bitcoin_alpha）→ 2-hop子图对trust网络无效，GNN/MLP同等失败
- 类型3 任务退化（email_eu）→ recall 精度过高，random已是最优

[2026-04-27] [GNN + 节点嵌入混合模型] [src/model/model.py, src/online/trainer.py, src/online/loop.py, configs/online/_smoke_gnn_node_emb.yaml(+新建)] [完成]
- LinkPredModel 新增 n_nodes / node_emb_dim 参数；node_emb_dim>0 时创建 nn.Embedding，将 emb(u)‖emb(v) concat 到 GIN 图嵌入后再送 Scorer
- trainer._build_flat_batched_graph 写入 g.ndata["_node_id"]（全局节点 ID），供模型 embedding lookup
- loop.py 透传 n_nodes / node_emb_dim；node_emb_dim=0 时完全向后兼容
- smoke test（college_msg, 5 轮, CPU）通过，loss 正常收敛

[2026-04-27] [SBM合成图四方法对比实验（n=500, topk=10/20）] [configs/online/sbm_random_topk{10,20}.yaml, sbm_gnn_topk{10,20}.yaml, sbm_node_emb_topk{10,20}.yaml, sbm_gnn_node_emb_topk{10,20}.yaml, results/online/sbm_*/rounds.csv] [完成]
- 数据集：SBM合成图 n_nodes=500, n_communities=5, p_in=0.3, p_out=0.05, T=500, edges_per_step=15, seed=42（约462有效节点）
- 结果汇总（峰值覆盖率 / MRR@5活跃均值 / 总接受边）：
  random    topk10: cov=0.3907 / mrr=0.645 / accepted=183 / active_rounds=37
  random    topk20: cov=0.2953 / mrr=0.551 / accepted=132 / active_rounds=23
  node_emb  topk10: cov=0.1944 / mrr=0.573 / accepted=78  / active_rounds=13
  node_emb  topk20: cov=0.2991 / mrr=0.708 / accepted=134 / active_rounds=12
  gnn       topk10: cov=0.3850 / mrr=0.767 / accepted=180 / active_rounds=26
  gnn       topk20: cov=0.2953 / mrr=0.435 / accepted=132 / active_rounds=19
  gnn+emb   topk10: cov=0.4206 / mrr=0.764 / accepted=199 / active_rounds=27  ← 峰值覆盖最高
  gnn+emb   topk20: cov=0.1533 / mrr=0.316 / accepted=56  / active_rounds=19
- 关键发现：n=500图太小，约第30~40轮后全部模型n_accepted降为0（图饱和）；topk=20普遍弱于topk=10；结论可信度不足，需用n_nodes≥5000重跑

[2026-04-27] [SBM合成图四方法对比实验v2（n=5000, two_hop_random召回, p_pos=1.0）] [configs/online/sbm5k_v2_*.yaml, results/online/sbm5k_v2_*/rounds.csv] [完成]
- 数据集：SBM n_nodes=5000, n_communities=5, p_in=0.3, p_out=0.05, T=2000, edges_per_step=30, seed=42
- 配置变更（相比v1）：p_pos=1.0（之前0.95），召回改为纯 two_hop_random top_k_recall=100（之前mixture+ppr_nodes）
- 结果汇总（峰值覆盖 / MRR@5活跃均值 / Hits@5 / 接受边 / 活跃轮数）：
  random       topk10: cov=0.2659 / mrr=0.940 / hits=0.965 / acc=982  / active=100
  random       topk20: cov=0.4348 / mrr=0.854 / hits=0.888 / acc=1750 / active=94
  node_emb     topk10: cov=0.2595 / mrr=0.958 / hits=0.985 / acc=953  / active=100
  node_emb     topk20: cov=0.0710 / mrr=0.516 / hits=0.547 / acc=96   / active=32   ← 崩溃
  gnn          topk10: cov=0.2633 / mrr=0.980 / hits=0.985 / acc=970  / active=100
  gnn          topk20: cov=0.4750 / mrr=0.913 / hits=0.930 / acc=1933 / active=100  ← 覆盖最高
  gnn+node_emb topk10: cov=0.1781 / mrr=0.992 / hits=0.992 / acc=583  / active=61
  gnn+node_emb topk20: cov=0.3004 / mrr=0.746 / hits=0.779 / acc=1139 / active=70
- 关键发现：
  1. gnn topk=20 覆盖最高(0.475)且跑满100轮，综合最强
  2. node_emb topk=20 在纯2-hop召回下冷启动失败，去掉PPR后弱点更暴露
  3. gnn+node_emb 弱于纯gnn，混合模型稀疏冷启动有负向干扰
  4. 去掉PPR后gnn topk=20覆盖从0.449→0.475，召回更纯净有利于gnn排序发挥

[2026-04-27] [分析：random/node_emb/gnn 互补性] [无代码变更] [分析记录]
- 三种方法的优势区域不同：
  random：冷启动鲁棒、探索最广，无排序能力
  node_emb：全局节点亲和力，图稠密后稳定，冷启动嵌入未收敛
  gnn：局部结构精准，图成熟后最强，冷启动子图稀疏时失效
- 互补证据：
  bitcoin上 gnn+node_emb MRR(0.677) > gnn(0.530) 且 > node_emb(0.608)
  SBM上 node_emb topk=20 仅活跃32轮崩溃，random/gnn跑满100轮，失败模式不同
  epinions上四种方法MRR均>0.90，图够大时互补性消失
- 主要互补形式：时序互补（冷启动用random/node_emb兜底→图成熟后切gnn主导），
  而非简单混合推荐列表（会稀释精度）
- 潜在方向：动态切换策略，按图密度或轮次自动调整主推模型

[2026-04-29] [新探索策略：exploit_ratio + hidden_dim 消融（wiki_vote）] [src/online/loop.py, configs/online/algo_sweep_wiki_vote/*_explore02.yaml, *_h4.yaml] [完成]

- **新探索策略**：将 user-level ε-greedy 替换为 slot-level 固定比例探索
  - `exploit_ratio=0.8`：top-10 中 8 条来自 GNN top-8，2 条从 GNN 未选候选随机抽
  - 配置字段：`trainer.exploit_ratio`（默认 1.0 = 纯利用，向后兼容）
- wiki_vote 消融结果（round 100）：

| 配置 | coverage | prec@K | MRR@10 | hits@5 |
|---|---|---|---|---|
| h8 纯利用（基准） | 0.532 | 0.0261 | 0.302 | 0.585 |
| random | 0.510 | 0.0252 | 0.328 | 0.624 |
| h8 + explore=0.2 | 0.528 | 0.0234 | 0.328 | 0.595 |
| h32 + explore=0.2 | 0.509 | 0.0196 | 0.278 | 0.461 |
| h4 + explore=0.2 | 0.499 | 0.0141 | 0.191 | 0.281 |
| h4 纯利用 | 0.482 | 0.0096 | 0.228 | 0.413 |

- 核心发现：
  1. h8 是最优 hidden_dim，h4 容量不足，h32 引入探索后过拟合
  2. h8 + explore=0.2 的 MRR@10 追平 random（0.328），但 coverage/hits 略低于纯利用
  3. 探索策略对负样本空间有扩充作用，但在小图（wiki_vote 7k节点）收益有限

[2026-04-28] [slashdot + epinions algo_sweep（11方法，100轮）] [configs/online/algo_sweep_slashdot/*(11个新建), configs/online/algo_sweep_epinions/*(11个新建), results/online/algo_sweep_slashdot/*, results/online/algo_sweep_epinions/*] [完成]

slashdot（77k节点，517k边，recip=0.186，sample_ratio=0.01）round 100结果：
- coverage：ground_truth=0.293 > cn=0.270 > gnn_sum=0.269 > aa=0.269 > gnn=0.263 > random=0.263 > pa=0.265 > jaccard=0.265 > gnn_concat=0.258 > node_emb=0.252 > mlp=0.251
- prec@K：random=0.0094 > gnn=0.0035 > gnn_sum=0.0031 > gnn_concat=0.0012（mlp/node_emb≈0）
- MRR@10：random=0.317 > gnn_concat=0.319 > cn=0.290 > gnn=0.286 > gnn_sum=0.226 > aa=0.208（mlp/node_emb缺失）
- 特点：slashdot上随机方法MRR竟高于GNN系列，可能与大图冷启动+子图稀疏有关

epinions（76k节点，509k边，recip未知，sample_ratio=0.01）round 100结果：
- coverage：ground_truth=0.357 > aa=0.295 > gnn_sum=0.294 > cn=0.292 > jaccard=0.284 > random=0.282 > gnn=0.278 > pa=0.275 > gnn_concat=0.272 > node_emb=0.259 > mlp=0.255
- prec@K：gnn_sum=0.0109 > random=0.0182 > gnn=0.0085（mlp/node_emb≈0）
- MRR@10：ground_truth=1.0 > random=0.388 > gnn_sum=0.320 > gnn=0.296 > gnn_concat=0.283 > cn=0.207 > jaccard=0.202 > pa=0.194 > aa=0.172 > mlp=0.111 > node_emb=0.000
- 亮点：epinions上gnn_sum MRR@10(0.320)明显优于aa/cn启发式(0.172/0.207)，与slashdot规律不同

[2026-04-28] [cooldown模式对比 + wiki_vote多方法评测] [scripts/run_algo_sweep.py, results/online/algo_sweep_init025_cd_hard5, results/online/algo_sweep_wiki_vote_init025_cd_hard5] [完成]
- cooldown三种模式对比（decay5/hard0/hard5），4真实数据集×11方法：hard5全面最优，均值coverage提升+0.03~+0.14；hard0≈decay5
- 核心发现：硬排除（hard）而非衰减（decay）才是关键，5轮窗口足够；后续实验默认改用hard5
- SBM5k（hard5）：启发式方法coverage达0.97+，结构过于规律；DC-SBM5k覆盖率仅0.27~0.30，召回瓶颈突出
- wiki_vote（100轮，hard5，8方法）：gnn_sum最强（coverage=0.453，mrr@10=0.376），高于random（0.408/0.356）；pa垫底

[2026-04-29] [SGD vs Adam 优化器对比实验] [src/online/loop.py, configs/online/optim_compare/] [完成]
- 将在线仿真优化器从硬编码 Adam 改为可配置（trainer.optimizer: adam|sgd），SGD 默认 momentum=0.9
- 4 数据集同配置公平对比（hidden_dim=8, init_ratio=0.25, 100轮）：Adam lr=0.001，SGD lr=0.01
- coverage 终点：college_msg Adam=0.3311 > SGD=0.3249；dnc_email SGD=0.4635 > Adam=0.4473；bitcoin_alpha SGD=0.4118 > Adam=0.4049；wiki_vote SGD=0.3098 > Adam=0.2949
- 结论：SGD 在 3/4 数据集上 coverage 略高（+0.007~+0.016），差距在 1-2% 量级，整体两者相当
[2026-04-29] [SGD vs Adam 优化器对比实验] [src/online/loop.py, configs/online/optim_compare/] [完成]
- 将在线仿真优化器从硬编码 Adam 改为可配置（trainer.optimizer: adam|sgd），SGD 默认 momentum=0.9
- 4 数据集同配置公平对比（hidden_dim=8, init_ratio=0.25, 100轮）：Adam lr=0.001，SGD lr=0.01
- coverage 终点：college_msg Adam=0.3311 > SGD=0.3249；dnc_email SGD=0.4635 > Adam=0.4473；bitcoin_alpha SGD=0.4118 > Adam=0.4049；wiki_vote SGD=0.3098 > Adam=0.2949
- 结论：SGD 在 3/4 数据集上 coverage 略高（+0.007~+0.016），差距在 1-2% 量级，整体两者相当

[2026-04-30] [p_neg 扫描实验：0 / 0.02 / 0.1 三组对比] [configs/online/algo_sweep_sgd_hard5*, results/online/algo_sweep_sgd_hard5*] [完成]
- 公共配置：cooldown_mode=hard, cooldown_rounds=5, total_rounds=100, GNN=gnn_sum(layer_sum,h8,SGD lr=0.01), 5数据集×7方法
- p_neg=0.00：random coverage 最强，gnn_sum 几乎垫底；mrr@5 上 random 也不差，CN/AA/Jaccard 反而更弱
- p_neg=0.02：coverage 排序基本不变；precision_k 从 ~0.001 升至 ~0.02，信号稍有增强但区分度仍弱
- p_neg=0.10：gnn_sum 的 mrr@5 开始超越 CN/AA/Jaccard/PA（dnc_email: gnn_sum=0.302 vs aa=0.019）；random mrr@5 在部分数据集急跌（dnc_email: 0.23→0.07）；coverage 上 random 与 gnn_sum 差距从 ~0.05 缩至 ~0.01
- 核心结论：p_neg=0 时 coverage 偏向探索、掩盖排序质量；p_neg=0.1 是更合理设置，模型排序能力开始显现
- random 强≠问题定义有误：random MRR 高是因为 CN/AA 在稀疏演化图中与真实正样本反相关；p_neg>0 引入错误边代价后 random 优势消失
- 后续方向：加 ε-greedy 探索，在 p_neg=0.1 设置下对比 gnn_sum vs random
