"""scripts/run_comparison.py — 模型对比实验（支持并行）

用法：
    # 并行跑（默认 4 个 worker，每个 run 一个子进程）
    PYTHONIOENCODING=utf-8 PYTHONPATH=. C:/conda/envs/gnn/python.exe \
        scripts/run_comparison.py --max_workers 4

    # 串行（调试用）
    ... run_comparison.py --max_workers 1

    # smoke test
    ... run_comparison.py --max_samples 500 --epochs 5 --max_workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASETS = [
    "data/processed/college_msg",
    "data/processed/bitcoin_otc",
    "data/processed/email_eu",
]

MODELS = [
    {"model_type": "gin",       "encoder_type": "last", "label": "GIN-last"},
    {"model_type": "graphsage", "encoder_type": "last", "label": "GraphSAGE"},
    {"model_type": "seal",      "encoder_type": "last", "label": "SEAL"},
    # TGAT 后续不投入：时间成本过高（300-500s/epoch），时序优势在子图框架下无体现
]

PYTHON = "C:/conda/envs/gnn/python.exe"


def run_one(dataset_dir: str, model_cfg: dict, base_args: argparse.Namespace) -> dict:
    """训练单个 (dataset, model) 组合，返回结果 dict。"""
    ds_name = Path(dataset_dir).name
    run_name = f"cmp_{ds_name}_{model_cfg['label']}"

    cmd = [
        PYTHON, "src/train.py",
        "--data_dir",    dataset_dir,
        "--run_name",    run_name,
        "--epochs",      str(base_args.epochs),
        "--batch_size",  str(base_args.batch_size),
        "--hidden_dim",  str(base_args.hidden_dim),
        "--num_layers",  str(base_args.num_layers),
        "--max_samples", str(base_args.max_samples),
        "--patience",    str(base_args.patience),
        "--seed",        str(base_args.seed),
        "--device",      base_args.device,
        "--model_type",  model_cfg["model_type"],
        "--encoder_type", model_cfg["encoder_type"],
        "--neg_strategy", base_args.neg_strategy,
    ]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": "."}

    # 捕获输出，完成后统一打印（避免多进程交叉输出）
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")

    # 读取最佳 val AUC
    log_path = Path("results/logs") / run_name / "train.json"
    best_val_auc = float("nan")
    if log_path.exists():
        records = json.loads(log_path.read_text())
        if records:
            best_val_auc = max(r.get("val_auc_mean", r.get("val_auc", float("nan"))) for r in records)

    tag = f"[{ds_name} | {model_cfg['label']}]"
    # 打印每 epoch 的最后几行（截取关键信息）
    lines = [l for l in proc.stdout.splitlines() if "Epoch" in l or "完成" in l or "早停" in l]
    print(f"\n{'='*65}")
    print(f"  {tag}  →  best_val_auc = {best_val_auc:.4f}")
    print(f"{'='*65}")
    for line in lines[-5:]:   # 只打印最后 5 行 epoch 日志
        print(f"  {line}")
    if proc.returncode != 0:
        print(f"  [STDERR] {proc.stderr[-300:]}")

    return {
        "dataset":      ds_name,
        "model":        model_cfg["label"],
        "best_val_auc": round(best_val_auc, 4),
    }


def print_table(rows: list[dict]) -> None:
    ds_names    = list(dict.fromkeys(r["dataset"] for r in rows))
    model_names = list(dict.fromkeys(r["model"]   for r in rows))
    lookup = {(r["dataset"], r["model"]): r["best_val_auc"] for r in rows}

    col_w  = 12
    header = f"{'Dataset':<22}" + "".join(f"{m:>{col_w}}" for m in model_names)
    print(f"\n{'='*len(header)}")
    print("  COMPARISON RESULTS  (best val AUC)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for ds in ds_names:
        row_str = f"{ds:<22}"
        for m in model_names:
            val = lookup.get((ds, m), "—")
            row_str += f"{str(val):>{col_w}}"
        print(row_str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int, default=30)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--hidden_dim",  type=int, default=64)
    parser.add_argument("--num_layers",  type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="0=不限制；>0 用于 smoke test")
    parser.add_argument("--patience",     type=int, default=10)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--neg_strategy", type=str, default="random:0.5,hard_2hop:0.3,degree:0.2",
                        help="训练负样本策略，支持混合格式如 'random:0.5,hard_2hop:0.3,degree:0.2'")
    parser.add_argument("--max_workers", type=int, default=4,
                        help="并行 worker 数（建议 ≤ 显卡数量；1=串行）")
    parser.add_argument("--datasets",    type=str, nargs="*", default=DATASETS)
    parser.add_argument("--models",      type=str, nargs="*", default=None,
                        help="只跑指定模型，如 gin graphsage")
    args = parser.parse_args()

    models = MODELS
    if args.models:
        models = [m for m in MODELS if m["model_type"] in args.models]

    # 生成所有 (dataset, model) 组合
    tasks = [(ds, m) for ds in args.datasets for m in models]
    total = len(tasks)
    print(f"共 {total} 个 run，max_workers={args.max_workers}")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_one, ds, m, args): (Path(ds).name, m["label"])
            for ds, m in tasks
        }
        done = 0
        for fut in as_completed(futures):
            done += 1
            ds_name, label = futures[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"\n[{done}/{total}] 完成: {ds_name} | {label}  "
                      f"best_val_auc={row['best_val_auc']:.4f}")
            except Exception as e:
                print(f"\n[{done}/{total}] 失败: {ds_name} | {label}  错误: {e}")

    # 保存 CSV
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "comparison.csv"
    rows_sorted = sorted(rows, key=lambda r: (r["dataset"], r["model"]))
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "model", "best_val_auc"])
        writer.writeheader()
        writer.writerows(rows_sorted)

    print_table(rows)
    print(f"\n结果已写入: {csv_path}")


if __name__ == "__main__":
    main()
