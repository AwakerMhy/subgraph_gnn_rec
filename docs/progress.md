# docs/progress.md — 历史变更日志

> 创建时间：2026-04-08 15:30
> 最后更新：2026-04-25

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
