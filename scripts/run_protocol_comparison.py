"""scripts/run_protocol_comparison.py — legacy vs simulated_recall 协议对比实验

矩阵：2 datasets × 2 models × 2 protocols = 8 runs
输出：results/protocol_comparison/<timestamp>.csv

用法（smoke test）：
    PYTHONUTF8=1 PYTHONPATH=. C:/conda/envs/gnn/python.exe \
        scripts/run_protocol_comparison.py --max_samples 2000 --epochs 10
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

DATASETS = [
    "data/processed/college_msg",
    "data/processed/bitcoin_otc",
]

MODELS = [
    {"model_type": "gin",       "encoder_type": "last", "label": "GIN"},
    {"model_type": "graphsage", "encoder_type": "last", "label": "GraphSAGE"},
]

PROTOCOLS = ["legacy", "simulated_recall"]

PYTHON = "C:/conda/envs/gnn/python.exe"
ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONPATH": "."}


def run_one(dataset_dir: str, model_cfg: dict, protocol: str,
            args: argparse.Namespace) -> dict:
    ds_name = Path(dataset_dir).name
    run_name = f"proto_{protocol}_{ds_name}_{model_cfg['label']}"

    cmd = [
        PYTHON, "-m", "src.train",
        "--data_dir",    dataset_dir,
        "--run_name",    run_name,
        "--epochs",      str(args.epochs),
        "--batch_size",  str(args.batch_size),
        "--hidden_dim",  str(args.hidden_dim),
        "--num_layers",  str(args.num_layers),
        "--max_samples", str(args.max_samples),
        "--patience",    str(args.patience),
        "--seed",        str(args.seed),
        "--device",      args.device,
        "--model_type",  model_cfg["model_type"],
        "--encoder_type", model_cfg["encoder_type"],
        "--protocol",    protocol,
    ]

    if protocol == "simulated_recall":
        cmd += [
            "--first_time_only",
            "--recall_method",  "common_neighbors",
            "--recall_top_k",   str(args.recall_top_k),
        ]
    else:
        cmd += ["--neg_strategy", "random:0.5,hard_2hop:0.3,degree:0.2"]

    print(f"[START] {run_name}", flush=True)
    t0 = __import__("time").time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    elapsed = __import__("time").time() - t0

    # 读取最佳 checkpoint 指标
    ckpt_path = Path("results/checkpoints") / f"{run_name}_best.pt"
    metrics: dict = {"protocol": protocol, "dataset": ds_name,
                     "model": model_cfg["label"], "elapsed_s": round(elapsed, 1)}

    log_path = Path("results/logs") / run_name / "train.json"
    if log_path.exists():
        records = json.loads(log_path.read_text())
        if records:
            best = max(records, key=lambda r: r.get("mrr", r.get("val_auc_mean", 0)))
            metrics.update({k: round(v, 4) for k, v in best.items()
                            if k not in ("epoch",)})

    if result.returncode != 0:
        print(f"[FAIL] {run_name}\n{result.stderr[-500:]}", flush=True)
        metrics["error"] = result.stderr[-200:]
    else:
        print(f"[DONE] {run_name}  ({elapsed:.0f}s)", flush=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--hidden_dim",  type=int,   default=64)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--max_samples", type=int,   default=2000)
    parser.add_argument("--patience",    type=int,   default=5)
    parser.add_argument("--recall_top_k", type=int,  default=100)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      type=str,   default="cpu")
    parser.add_argument("--max_workers", type=int,   default=2)
    args = parser.parse_args()

    jobs = [
        (ds, model, proto)
        for ds in DATASETS
        for model in MODELS
        for proto in PROTOCOLS
    ]
    print(f"共 {len(jobs)} 个任务，max_workers={args.max_workers}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {ex.submit(run_one, ds, model, proto, args): (ds, model, proto)
                   for ds, model, proto in jobs}
        for fut in as_completed(futures):
            results.append(fut.result())

    # 保存结果
    out_dir = Path("results/protocol_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{ts}.csv"

    if results:
        all_keys = sorted({k for r in results for k in r})
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
        print(f"\n结果保存至：{out_path}")

    # 打印汇总表
    print("\n=== 协议对比汇总 ===")
    print(f"{'protocol':<20} {'dataset':<15} {'model':<12} {'mrr':>7} {'val_auc_mean':>13} {'elapsed_s':>10}")
    for r in sorted(results, key=lambda x: (x.get("dataset",""), x.get("model",""), x.get("protocol",""))):
        mrr = r.get("mrr", "-")
        auc = r.get("val_auc_mean", "-")
        print(f"{r.get('protocol',''):<20} {r.get('dataset',''):<15} {r.get('model',''):<12} "
              f"{str(mrr):>7} {str(auc):>13} {r.get('elapsed_s',0):>10.0f}s")


if __name__ == "__main__":
    main()
