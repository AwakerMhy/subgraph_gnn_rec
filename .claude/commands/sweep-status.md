# /sweep-status

读取当前实验的进度，汇总 UAUC、AUC、coverage、MRR@5、hits@5 指标，输出均值±标准差对比表格。

## 参数

可选：`/sweep-status <sweep_name>`，不传则自动检测最新/最活跃的 sweep。

## 执行步骤

### 1. 确定 sweep 目录

- 若用户传入了 `<sweep_name>`，使用 `results/orchestrator/<sweep_name>/`
- 否则，列出 `results/orchestrator/` 下所有子目录，选取 `experiments.db` 最近修改的那个

### 2. 读取进度（从 DB）

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("<sweep_dir>/experiments.db")
status_df = pd.read_sql(
    "SELECT dataset, status, COUNT(*) as n FROM cells GROUP BY dataset, status",
    conn
)
conn.close()
```

输出格式（ASCII）：
```
=== sweep: <name>  done/total ===
  [OK] dataset        done  pend  run  hold  total
  [->] dataset        done  pend  run  hold  total
  [..] dataset           0     0    0    70     70
  TOTAL               xxx   xxx  xxx   xxx    xxx
```

### 3. 读取已完成实验的指标（从 results.csv）

```python
df = pd.read_csv("<sweep_dir>/results.csv")
```

若文件不存在或为空，跳过指标汇总，只展示进度表。

### 4. 聚合：均值 ± 标准差

对以下每个指标，按 `dataset × method` 分组，计算 mean 和 std（保留3位小数）：

| 指标标签 | 列名 |
|----------|------|
| UAUC     | `final_uauc_feedback` |
| AUC      | `final_auc_feedback` |
| coverage | `final_coverage` |
| MRR@5    | `final_mrr@5` |
| hits@5   | `final_hits@5` |

```python
def agg_mean_std(df, col):
    g = df.groupby(['dataset', 'method'])[col]
    mean = g.mean().round(3)
    std  = g.std().round(3)
    return mean, std
```

单 seed 的格子只显示均值（无 std）；多 seed 显示 `mean±std`（如 `0.728±0.12`）。
缺失值显示 `-`。

### 5. 核心方法顺序

所有 pivot 表均按此列顺序：
`random`, `cn`, `aa`, `pa`, `jaccard`, `gnn`, `gnn_concat`, `gnn_sum`, `gnn_h32`, `gnn_concat_h8`, `gnn_sum_h8`, `gat_emb`, `graphsage_emb`, `seal`

缺失列直接跳过，不补空列。

### 6. 输出各指标 pivot 表

对每个指标依次输出：

```
[ UAUC  mean±std ]
method          random   cn             aa            ...  gnn_concat
advogato           -    0.681±0.02    0.696±0.02     ...  0.728±0.15
bitcoin_alpha      -    0.597±0.04    0.623±0.07     ...  0.803±0.15
...

[ AUC   mean±std ]
...

[ coverage  mean±std ]
...

[ MRR@5  mean±std ]
...

[ hits@5  mean±std ]
...
```

每张表下方标注参与计算的数据集列表。

### 7. 每个指标的平均 rank + 结论分析

对 UAUC、coverage、MRR@5、hits@5 **每个指标** 均输出：

#### 7a. 平均 rank

基于该指标均值 pivot，对每行（数据集）做方法 rank（ascending=False），
取各方法在已有数据集上的均值 rank，升序排列输出前 8 名：

```
[ UAUC avg rank ]         [ coverage avg rank ]      [ MRR@5 avg rank ]        [ hits@5 avg rank ]
 1. gnn_concat  3.00       1. aa        3.50           1. gnn_concat  3.00       1. gnn_concat  2.75
 2. gnn_sum     3.67       2. cn        4.00           2. gnn_sum     4.00       2. seal        4.50
 ...                       ...                         ...                       ...
```

#### 7b. 结论分析

对每个指标，基于 pivot 均值数据，自动生成简短结论（2~4 句话），覆盖：
- 哪个方法总体最优（rank 第 1）
- 是否存在"在某类数据集上例外"的情况（例如：小图 vs 大图、wiki_vote 例外等）
- GNN 和启发式的差距量级（如"gnn_concat 领先最强启发式约 +0.05~+0.12"）
- 方差特征（GNN std 普遍高于启发式的倍数）

输出格式：
```
[ UAUC 结论 ]
gnn_concat 总体最优（avg rank 3.00），在 9/12 数据集领先所有方法。
大图（sx_superuser/sx_askubuntu）上 gnn_sum/gnn_h32 表现更稳定，gnn_concat 优势收窄。
wiki_vote 是唯一例外：GNN 全面弱于 cn/aa，gnn_concat_h8 勉强最优（0.573 vs cn 0.558）。
GNN 方法 std 是启发式的 2~5 倍（gnn_concat std 0.09~0.20 vs cn std 0.02~0.11）。

[ coverage 结论 ]
...

[ MRR@5 结论 ]
...

[ hits@5 结论 ]
...
```

### 8. 总结

- 各指标最优方法一览（UAUC / coverage / MRR@5 / hits@5 各自 rank 第 1）
- 尚未完成的数据集及 cell 数
- 剩余 cell 总数

## 输出格式要求

- 全部使用 ASCII 字符（禁止 Unicode 方块/特殊符号），避免 Windows GBK 编码错误
- `mean±std` 格式：列宽统一为 12 字符，`-` 表示无数据
- pivot 表用 pandas `to_string(na_rep='-')` 输出
