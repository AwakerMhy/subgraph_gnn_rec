# docs/reproducibility.md — 可复现性基建规范

> 创建时间：2026-04-08 15:30
>
> **CC 在以下时机必须完整读取本文件**：项目初始化、涉及训练/数据 pipeline/依赖/config 的任务开始前、引入新随机性来源时。

---

## 五大支柱

### 1. 依赖锁定

```bash
# 锁定方式
conda env export --no-builds > environment.yml   # 完整环境快照
pip freeze > requirements.lock                   # pip 精确版本

# 安装方式（复现时）
conda env create -f environment.yml
# 或
conda create -n gnn python=3.10
pip install -r requirements.lock
```

**规则**：
- `requirements.txt`：项目直接依赖（带版本范围，供人阅读）
- `requirements.lock`：完整精确版本（供复现用）
- 引入新依赖时同时更新两个文件，并在 `DECISIONS.md` 记录引入原因

### 2. 随机种子

```python
# src/utils/seed.py
def set_seed(seed: int) -> None:
    """统一设置所有随机数生成器的种子"""
    import random, numpy as np, torch, dgl
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**规则**：
- 所有脚本（train.py / evaluate.py / experiments/\*.py）第一行调用 `set_seed(cfg.seed)`
- 种子值在 `configs/default.yaml` 中定义：`seed: 42`
- 消融实验必须保持相同种子，避免随机性干扰对比

### 3. 实验目录约定

```
results/logs/<timestamp>_<name>/
├── config.yaml        # 本次实验完整 config 快照
├── git_hash.txt       # git commit hash（git rev-parse HEAD）
├── environment.txt    # conda env list 输出
├── seed.txt           # 使用的随机种子
├── metrics.json       # 最终测试集指标 {"auc": 0.xx, "ap": 0.xx, ...}
├── logs/
│   └── train.json     # 每 epoch 追加：{"epoch": N, "loss": ..., "val_auc": ...}
└── checkpoints/
    ├── best.pt        # 最优 checkpoint（val AUC 最高）
    └── last.pt        # 最后一个 epoch 的 checkpoint
```

**规则**：
- `<timestamp>` 格式：`%Y%m%d_%H%M%S`
- 每次 run 自动创建上述目录结构（`train.py` 负责）
- `git_hash.txt` 在训练开始时自动写入（`git rev-parse HEAD`）
- `metrics.json` 仅在评估完成后写入，训练中途不写

### 4. 数据版本

```
data/
├── raw/                    # 只读，从网络下载后不再修改
│   └── <dataset_name>/
│       └── <原始文件> + sha256.txt   # 下载后校验哈希
├── processed/              # 由 dataset.process() 生成
│   └── <dataset_name>/
│       ├── edges.csv
│       ├── nodes.csv
│       ├── meta.json
│       └── subgraphs/      # dgl.save_graphs 缓存
└── synthetic/              # 合成数据集，由生成器脚本产生
    └── <generator>_<params>/
        ├── edges.csv
        ├── nodes.csv
        └── meta.json
```

**规则**：
- `data/raw/` 每个文件旁存 `sha256.txt`，`dataset.load()` 时校验
- `data/processed/` 如果删除，运行 `python -m src.dataset.<name>` 可重建
- `data/raw/` 在 `.gitignore` 中（体积大），但 `sha256.txt` 进 git

### 5. 配置外置

```yaml
# configs/default.yaml 是所有超参的权威来源
# 命令行 override 格式（使用 OmegaConf 或 argparse）：
python experiments/run_main.py dataset=college_msg model.hidden_dim=128 training.lr=0.0005
```

**规则**：
- 所有超参必须在 yaml 中定义，禁止在 `.py` 文件中硬编码数值
- 命令行 override 会被记录在 `config.yaml` 快照中
- `config.yaml` 快照包含 `_override_` 字段，标注实验与 default 的差异

---

## 违规检测信号

| 违规行为 | 检测方法 |
|----------|---------|
| 超参硬编码在 .py 中 | grep `lr=` / `hidden_dim=` 在 src/ 下 |
| 训练未调用 set_seed | grep `set_seed` 在 train.py 开头 |
| 实验目录缺失字段 | 检查 `results/logs/<run>/` 目录结构 |
| raw 数据无 sha256 | ls `data/raw/<name>/sha256.txt` |
| 缓存当数据源 | 确认缓存删除后脚本能重建 |
