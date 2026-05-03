# Experiment Template

> 创建时间：2026-05-03 00:00
>
> 所有在线仿真实验必须遵循本模板中的指标定义、表格格式、目录命名规范和进度记录格式。

---

## 1. 规范指标集

报告顺序严格固定，缺失项必须填 `—`：

| 字段 | 说明 |
|------|------|
| `coverage` | 推荐列表中至少有 1 条被接受的轮次比例 |
| `mrr@1` | Mean Reciprocal Rank，截断 K=1 |
| `mrr@3` | Mean Reciprocal Rank，截断 K=3 |
| `mrr@5` | Mean Reciprocal Rank，截断 K=5 |
| `mrr@10` | Mean Reciprocal Rank，截断 K=10 |
| `hits@5` | 推荐列表前 5 内命中率 |
| `hit_rate@1` | 推荐列表第 1 位命中率（等价 MRR@1，独立列出便于对比） |
| `auc` | 全量负样本 ROC-AUC（精排模型评估用，召回类算法填 `—`） |
| `precision_k` | 可选，放最后一列；K 值须与 top_k 对齐 |

**规则**：
- 指标值统一保留 **4 位小数**（如 `0.3127`）。
- `coverage` 用百分比表示（如 `31.27%`）。
- 多次运行取均值，若有方差须在备注列注明（如 `0.3127±0.008`）。

---

## 2. 跨数据集对比表格格式

```markdown
### [实验组名称] — [日期]

配置摘要：recall=two_hop_random(top_k=100), rounds=100, hidden_dim=32, seed=42

| Dataset         | Model        | coverage | mrr@1  | mrr@3  | mrr@5  | mrr@10 | hits@5 | hit_rate@1 | auc    |
|-----------------|--------------|----------|--------|--------|--------|--------|--------|------------|--------|
| sx_askubuntu    | random       | 28.00%   | 0.1234 | 0.1456 | 0.1523 | 0.1601 | 0.2100 | 0.1234     | —      |
| sx_askubuntu    | ground_truth | 35.00%   | 0.2100 | 0.2300 | 0.2400 | 0.2500 | 0.3100 | 0.2100     | —      |
| sx_askubuntu    | gnn_sum      | 33.00%   | 0.1900 | 0.2100 | 0.2200 | 0.2300 | 0.2800 | 0.1900     | 0.7200 |
| sx_mathoverflow | random       | ...      | ...    | ...    | ...    | ...    | ...    | ...        | —      |
```

**规则**：
- 同一数据集的行连续排列，不同数据集之间空一行。
- `ground_truth` 行必须包含，作为该召回设置下的效果上界。
- 最优值（非 ground_truth）用 `**bold**` 标注。
- 表格上方的配置摘要须写明所有影响可复现性的超参。

---

## 3. out_dir 命名规范

```
results/logs/<timestamp>_<dataset>_<model>_<variant>/
```

各段含义：

| 段 | 格式 | 示例 |
|----|------|------|
| `timestamp` | `YYYYMMDD_HHMMSS` | `20260503_143022` |
| `dataset` | 数据集名（下划线） | `sx_askubuntu` |
| `model` | 模型类型 | `gnn_sum` / `random` / `ground_truth` |
| `variant` | 区分同模型不同超参的标签 | `h32_r100` / `eps0.3` / `topk100` |

完整示例：
```
results/logs/20260503_143022_sx_askubuntu_gnn_sum_h32_r100/
```

**必须包含的子文件**：

```
<run_dir>/
├── config.yaml        # 完整配置快照（运行时自动 dump）
├── git_hash.txt       # git rev-parse HEAD 输出
├── metrics.json       # 最终指标（与上方表格一致的字段）
└── logs/
    └── train.json     # 每轮指标（online 场景：每 round 一条）
```

**规则**：
- `out_dir` 中不得省略 `variant` 段；若无变体，用 `default`。
- 同一实验组的所有 run 共享相同 `timestamp` 前缀（批量启动时统一生成）。
- 禁止用相同 `out_dir` 覆盖已有结果；若需重跑，更新时间戳。

---

## 4. PROGRESS.md 实验任务条目格式

```markdown
## [实验名称，如 algo_sweep_sx_askubuntu]
- 状态：进行中 / 已完成
- 开始时间：YYYY-MM-DD HH:MM
- 当前进度：第 N 步 / 共 M 步
- [x] 步骤1：生成/确认 configs
- [x] 步骤2：smoke test（最小数据集，rounds=5）
- [ ] 步骤3：全量运行（← 当前）
- [ ] 步骤4：生成对比表格（plot_algo_sweep.py）
- [ ] 步骤5：将结果表格追加至 docs/method_comparison.md
- 数据集：sx_askubuntu, sx_mathoverflow, slashdot
- 模型组：random, ground_truth, gnn_sum, mlp, cn, aa, jaccard
- out_dir 前缀：results/logs/YYYYMMDD_HHMMSS_<dataset>_<model>_<variant>
- 已变更文件：configs/online/algo_sweep_xxx/*.yaml
- 下一步行动：具体描述
```

**规则**：
- smoke test 步骤不可省略，必须在全量运行前完成并标 `[x]`。
- 步骤4（出图）紧随全量运行，不得拖延到下次会话。
- 步骤5（写入 method_comparison.md）为强制收尾步骤。
