# MISTAKES — 错误与教训库

> 创建时间：2026-04-08 15:30
> 最后更新：2026-04-21
>
> **任务开始前扫读，注意"检测信号"匹配当前任务。**

---

<!-- 格式模板：
## [YYYY-MM-DD] 简短标题
- **类别**：思路错误 / 代码错误 / 工具误用 / 需求误解
- **现象**：发生了什么、报错信息或用户原话
- **根因**：为什么会犯（追到底层假设）
- **教训**：下次类似情况应该怎么做
- **检测信号**：什么样的代码/任务/上下文应该触发警觉
- **关联文件**：（可选）
- **复发次数**：1
-->

---

## [2026-04-20] Legacy 协议系统性失败：tr_auc 过拟合，val AUC 全面趋近随机

- **类别**：思路错误（实验设计层面）
- **现象**：在 college_msg / bitcoin_otc / email_eu 三个数据集上，GIN-last 和 GraphSAGE 使用 legacy 协议（mixed 负样本 + hard_2hop+historical 双路验证）训练 30 epoch，呈现以下一致规律：
  - `tr_auc` 在 ep1 即达到 0.95+，ep2-3 飙至 0.99+，此后不再下降
  - `val_auc_mean` 始终在 0.49–0.55 区间微幅震荡，无明显上升趋势
  - 最佳 val 往往出现在 ep1–3，之后持平或略降
  - 具体数据汇总（best val_auc_mean / 已跑 epoch 数）：

  | 数据集 | GIN-last | GraphSAGE |
  |---|---|---|
  | college_msg | 0.535 @ep1（8ep 后放弃） | 0.533 @ep1（8ep 后放弃） |
  | bitcoin_otc | 0.511 @ep6（7ep） | 0.550 @ep3（7ep） |
  | email_eu | 0.514 @ep8（13ep） | 0.531 @ep11（13ep） |

- **根因**：
  1. **hard_2hop 验证路径太难**：稠密社交图中 2-hop 内正负样本子图结构几乎无差别，模型无法区分，val_auc 趋近 0.50
  2. **historical 验证路径有固有假负**：历史邻居中 ~17% 会在未来重连，使验证信号含噪，上限约 0.53–0.55
  3. **训练负样本仍有结构泄露**：mixed 策略中 hard_2hop 负样本与正样本结构高度相同，模型在训练集上过拟合"2-hop 内是否与 u 直接相连"这一 trivial 特征，而非学习真正的链路形成模式
  4. **tr_auc 虚高与 val 脱节**：训练集的 hard_2hop 负样本对模型几乎无挑战（模型只需记住训练集节点对），导致过拟合但无泛化
- **教训**：legacy 协议在当前数据集和评估路径组合下，**无法产生有意义的模型间对比结果**；val_auc 的差异（如 GIN vs GraphSAGE）更多反映随机噪声而非真实性能差距
- **检测信号**：ep1 的 tr_auc > 0.95 且 val_auc_mean < 0.56；epoch 增加后 val 不升反降或持平
- **关联文件**：`src/train.py`（legacy 路径）、`DECISIONS.md`（负样本两极化问题记录）
- **正确方向**：切换到 `simulated_recall` 协议，用模拟召回候选集替代 hard_2hop+historical 验证路径，评估指标改为 MRR/NDCG@K
- **复发次数**：1（横跨 3 个数据集、2 个模型，共 6 次独立实验）

---

## [2026-04-20] bitcoin_alpha 三个模型 val_auc_mean 全部为 0.000

- **类别**：代码错误（待定）/ 数据问题（待定）
- **现象**：bitcoin_alpha 上 GIN-last（3ep）、GraphSAGE（30ep）、SEAL（11ep）的 `val_auc_mean` 均记录为 0.000；`tr_auc` 正常收敛（0.67–0.93）；hard_2hop 和 historical 字段未被正确记录
- **根因**：尚未诊断。可能原因：
  1. bitcoin_alpha 验证集边数极少（正样本数 < 2），AUC 计算退化为 0 或 NaN，被写为 0
  2. `train.py` 中 val_auc_mean 计算路径存在 silent exception，静默写 0
  3. bitcoin_alpha 预处理后时间切分导致 val split 为空
- **教训**：数据集接入后需验证 val/test split 的正负样本数量；AUC 计算须对样本数不足的情况显式报错而非静默返回 0
- **检测信号**：train.json 中 val_auc_mean=0.0 但 tr_auc 正常；或 hard_2hop/historical 字段缺失
- **关联文件**：`src/dataset/real/bitcoin_alpha.py`、`src/train.py`
- **复发次数**：1（待修复后确认根因）

---

## [2026-04-20] SEAL 在 bitcoin_otc / email_eu 上 historical AUC 异常高（~0.99）

- **类别**：代码错误
- **现象**：SEAL 在 bitcoin_otc 上 `val_auc_historical`=0.9977，email_eu 上=0.9323；GIN/GraphSAGE 同数据集 historical AUC 约 0.50–0.51；SEAL 的 `val_auc_mean`=0.000（字段缺失）
- **根因**：尚未诊断。可能原因：
  1. SEAL 的 `collate_fn` 或评估循环未正确使用修复后的 historical 候选池（仍使用旧的 `all_time_adj_out` 排除逻辑，导致候选池塌缩为空 → fallback 到 random → trivially 高 AUC）
  2. SEAL 模型的 score 输出方向相反（负样本打分反而更高），使 historical AUC 虚高
  3. `val_auc_mean` 字段在 SEAL 训练路径中未被写入（说明 SEAL 走了不同的代码分支）
- **教训**：新增 baseline 模型后须验证其评估路径与主模型完全一致；historical AUC > 0.9 是强烈的异常信号
- **检测信号**：某模型 historical AUC > 0.80，而同数据集其他模型 historical AUC ≈ 0.50；`val_auc_mean` 字段缺失
- **关联文件**：`src/baseline/seal.py`、`src/train.py`（collate_fn / eval 路径）
- **复发次数**：1（待修复）

---

## [2026-04-15] 离线子图缓存磁盘空间爆炸
- **类别**：思路错误
- **现象**：college_msg train split（83768 样本）的 DGL binary 缓存文件达 12GB；三个数据集全量缓存估计超 100GB
- **根因**：DGL binary 格式每个 DGLGraph 有固定 metadata 开销；college_msg 节点数只有 1899，2-hop 子图接近全图，每个图 ~146KB
- **教训**：子图缓存方案只适合节点数量大（子图远小于全图）的稀疏数据集；对小图不可用
- **检测信号**：数据集节点数 < 5000 且拟用 DGL binary 缓存子图时，先用 `(n_nodes × max_hop_k² × 8B × n_samples)` 估算磁盘占用
- **正确方案**：在内存中预构建 TimeAdjacency（时序邻接表），二分查找替代 DataFrame 全扫，每 epoch O(log degree) 而非 O(|E|)
- **复发次数**：1

## [2026-04-09] 快路径子图提取忽略 cutoff_time 导致时间泄露
- **类别**：代码错误
- **现象**：训练 AUC 迅速到达 1.0，val AUC 始终 ~0.52，完全无泛化
- **根因**：`extract_subgraph` 快路径直接使用 `prebuilt_adj_out`（构建于 `train_cutoff`），未按样本自身 `cutoff_time` 过滤；早期训练样本看到了未来训练边
- **教训**：prebuilt adj 只能用于负样本排除或离线缓存索引，不能直接替代 per-sample 时间过滤
- **检测信号**：tr_auc 快速到 1.0 但 val_auc 不动；`prebuilt_adj` 构建时刻 ≠ 样本 cutoff_time
- **复发次数**：1

## [2026-04-16] 验证集 hard_2hop 未传 time_adj，延续结构泄露 bug
- **类别**：代码错误
- **现象**：`val_ds_hard2hop` 构建时未传 `time_adj`，fallback 到 `all_time_adj_out` 构建二跳候选池；训练集的同类 bug 已修复但验证集遗漏
- **根因**：修复训练集 hard_2hop 时只检查了 `train_ds`，未检查三路验证集的构建参数
- **教训**：修复负样本 bug 后必须同时检查所有调用点（train / val / test）
- **检测信号**：`val_ds_hard2hop` 初始化缺少 `time_adj` 参数
- **关联文件**：`src/train.py:356-359`
- **修复**：2026-04-16，加入 `time_adj=time_adj`
- **复发次数**：1

## [2026-04-15] hard_2hop fallback 到全时段表引入轻微结构泄露
- **类别**：思路错误
- **现象**：`hard_2hop` 策略在 `prebuilt_adj_out` 未传入时，自动 fallback 到 `all_time_adj_out` 构建二跳候选池；该表包含 `t ≥ cutoff_time` 的未来边，使部分候选节点的可达性依赖未来结构
- **根因**：fallback 分支缺少区分"截断图"和"全时段图"的意识，把全时段邻接表当作截断图的替代品
- **教训**：`hard_2hop` 只应使用截断图（`t < cutoff_time`）构建二跳候选池；若 `prebuilt_adj_out` 未传入，应实时按 cutoff 过滤 edges 构建，而非 fallback 到 `all_time_adj_out`
- **检测信号**：`hard_2hop` 策略被调用但 `prebuilt_adj_out=None`；`hop_adj` 变量赋值为 `all_time_adj_out`
- **关联文件**：`src/graph/negative_sampling.py:106-111`
- **修复**：2026-04-16，改为优先使用 `time_adj.out_neighbors(u/nb, cutoff_time)` 精确查询；全时段表降为最终兜底
- **复发次数**：1

## [2026-04-09] 负样本只排除历史边导致假负样本
- **类别**：思路错误
- **现象**：负样本 v_neg 在未来可能成为 u 的真实目标，被标为 label=0
- **根因**：排除集只用 `t < cutoff_time` 的出边，未考虑 `t ≥ cutoff_time` 的未来边
- **教训**：负样本排除集应用全时段出边（`all_time_adj_out`），确保 v_neg 永远不是 u 的目标
- **检测信号**：负样本采样传入了 `cutoff_time` 相关的邻接表作为排除集
- **复发次数**：1

## [2026-04-21] 在线精排模型崩溃：loss→0，precision→0 ✓已解决

- **类别**：思路错误（实验设计）
- **现象**：college_msg 上首批实验显示 loss 快速收敛到 0.003，precision@K 随轮次单调下降至 0，coverage 100 轮增量只有 0.022–0.024
- **根因**：(1) Oracle 反馈（(u,v)∈G\* → 100% 接受）相当于把标签直接喂给模型，负样本全来自非 G\* 边，模型只需学"召回分数"即可区分正负，无需学习有用特征；(2) AA 召回与精排高度重叠，模型退化为直接复制召回分数
- **教训**：在线系统的反馈信号必须引入噪声（p_pos<1, p_neg>0）来模拟真实不确定性；召回与精排应使用不同信息来源
- **检测信号**：loss < 0.01 且 precision < 0.01 且 coverage 增量 < 0.003/轮
- **解决方案**：引入 p_pos=0.8/p_neg=0.02 概率反馈（P0-1）+ Mixture 召回多样化（P0-2）
- **复发次数**：1

## [2026-04-21] init_edge_ratio 消融失效：五档实验结果相同 ✓已解决

- **类别**：思路错误（实验设计）
- **现象**：init_edge_ratio=0.05/0.10/0.20/0.30/0.50 五档实验，100 轮 coverage 增量恒在 0.022–0.024 之间，初始稀疏度对效率无影响
- **根因**：瓶颈不在初始图密度，而在精排模型的判别能力——Oracle 反馈下模型无法区分真实"好"推荐和结构相似的"坏"推荐
- **教训**：消融实验前先确认被消融的变量不被其他瓶颈掩盖；先修复根本性能瓶颈再做消融
- **检测信号**：多个不同参数配置产出几乎相同的指标曲线
- **解决方案**：修复模型崩溃问题（P0-1+P0-2）后重新运行消融实验
- **复发次数**：1

## [2026-04-26] stratified init floor 导致 init_ratio 消融失效

- **类别**：思路错误（实验设计）
- **现象**：bitcoin_alpha init=0.1 与 init=0.05 实验结果完全相同（coverage、mrr 曲线重合）
- **根因**：stratified init 保证每个 source 节点至少一条出边（floor = n_unique_sources）；当 init_n < n_unique_sources 时，两个 ratio 都触发 floor，实际初始边数相同
- **教训**：使用 stratified init 做 ratio 消融前，先确认 init_n > n_unique_sources；否则改用 random init strategy
- **检测信号**：两个不同 init_ratio 的实验初始 coverage 完全相同
- **解决方案**：bitcoin_alpha 消融改用 init=0.2（init_n > n_unique_sources），init=0.1 档数据废弃
- **复发次数**：1
