# /exp-watch

快速可视化正在运行的实验进度：ASCII 进度条 + running cell 当前轮次 + 最近完成记录。

## 参数

可选：`/exp-watch <sweep_name>`，不传则自动选取 `experiments.db` 最近修改的 sweep。

## 执行步骤

### 1. 定位 sweep

与 `/sweep-status` 相同：选取 `results/orchestrator/` 下 `experiments.db` 最近修改的子目录。

### 2. 从 DB 读取状态

```python
import sqlite3, pandas as pd, time
from pathlib import Path

conn = sqlite3.connect("<sweep_dir>/experiments.db")
cells = pd.read_sql("SELECT * FROM cells", conn)
conn.close()
```

### 3. 输出总体进度条

```
=== ir40_constlr5e5_multiseed_bidir  505/840 (60.1%) ===
[████████████████████░░░░░░░░░░░░░] 60.1%   ETA ~2h14m
```

ETA 计算：取最近 20 个 completed cell 的 `duration_s` 均值，乘以剩余 cell 数（pending + hold）。

### 4. 每个数据集的进度条（紧凑格式）

```
dataset              [████████████░░░░░░░░] done/total  status
advogato             [████████████████████]  70/70       OK
bitcoin_alpha        [████████████████████]  70/70       OK
wiki_vote            [██████████████████░░]  63/70       3 running
sx_mathoverflow      [████░░░░░░░░░░░░░░░░]  16/70       8 running, 46 hold
epinions             [░░░░░░░░░░░░░░░░░░░░]   0/70       hold
```

进度条宽度 20 格，`#` = completed，`-` = running，`.` = pending/hold。
注意：Windows GBK 终端不支持 Unicode 方块字符，使用 ASCII `#`/`-`/`.` 代替。

### 5. 正在运行的 cell 详情

对每个 `status='running'` 的 cell，读取其 `log_path`，提取最后一行含 `Round` 的行，解析当前轮次：

```python
import re
with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()
round_lines = [l for l in lines if 'Round' in l]
last = round_lines[-1] if round_lines else None
# 解析: "Round  22/30  coverage=0.419  uauc=0.551  (12.3s)"
```

输出格式：
```
Running cells:
  wiki_vote / seal       s0   Round 26/30  uauc=0.614  (12.9s/round)
  sx_mathoverflow / gnn  s1   Round  8/30  uauc=0.672  (45.2s/round)
  ...
```

若 log 不存在或无 Round 行，显示 `(starting...)`。

### 6. 最近完成的 5 个 cell

按 `finished_at` 降序取前 5 条 completed：

```
Recently completed:
  wiki_vote / jaccard    s42   359s   uauc=-  (no results yet)
  wiki_vote / pa         s3    341s
  ...
```

（uauc 从 results.csv 查，查不到显示 `-`）

### 7. Phase 信息（若有 hold 状态）

若存在 `hold` 的 cell，输出：
```
Phase2 (hold): 326 cells waiting — will unlock after Phase1 completes
```

## 输出示例

```
=== ir40_constlr5e5_multiseed_bidir  505/840 (60.1%) ===
[████████████████████░░░░░░░░░░░░░] 60.1%   ETA ~3h22m

Per-dataset:
  advogato         [████████████████████]  70/70   OK
  bitcoin_alpha    [████████████████████]  70/70   OK
  bitcoin_otc      [████████████████████]  70/70   OK
  college_msg      [████████████████████]  70/70   OK
  dnc_email        [████████████████████]  70/70   OK
  email_eu         [████████████████████]  70/70   OK
  wiki_vote        [███████████████████░]  69/70   1 running
  sx_mathoverflow  [████░░░░░░░░░░░░░░░░]  16/70   8 running, 46 hold
  epinions         [░░░░░░░░░░░░░░░░░░░░]   0/70   hold
  slashdot         [░░░░░░░░░░░░░░░░░░░░]   0/70   hold
  sx_askubuntu     [░░░░░░░░░░░░░░░░░░░░]   0/70   hold
  sx_superuser     [░░░░░░░░░░░░░░░░░░░░]   0/70   hold

Running (9 cells):
  wiki_vote       / seal          s0   Round 26/30  uauc=0.614  (12.9s/round)
  sx_mathoverflow / gnn           s0   Round  5/30  uauc=0.643  (52.1s/round)
  sx_mathoverflow / gnn_concat_h8 s42  Round  3/30  uauc=0.589  (48.7s/round)
  ...

Recently completed (last 5):
  wiki_vote / jaccard s42   359s
  wiki_vote / pa      s3    341s
  wiki_vote / random  s2    298s
  ...

Phase2 (hold): 326 cells — unlocks after wiki_vote + sx_mathoverflow finish
```
