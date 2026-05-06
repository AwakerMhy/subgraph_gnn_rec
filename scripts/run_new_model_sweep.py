"""scripts/run_new_model_sweep.py — 新模型 vs GNN 对比实验。

对比：gnn_sum / seal / graphsage_emb / gat_emb
数据集：college_msg / dnc_email / bitcoin_alpha / email_eu
配置：constant lr=1e-4, init_ratio=0.4, 30轮, seed=42
"""
import subprocess
import sys
import concurrent.futures
from pathlib import Path
import yaml

PYTHON = sys.executable

DATASETS = [
    ("college_msg",   "data/processed/college_msg/edges.csv"),
    ("dnc_email",     "data/processed/dnc_email/edges.csv"),
    ("bitcoin_alpha", "data/processed/bitcoin_alpha/edges.csv"),
    ("email_eu",      "data/processed/email_eu/edges.csv"),
]

MODELS = ["gnn_sum", "seal", "graphsage_emb", "gat_emb"]

SWEEP_TAG = "new_model_sweep_ir40_s42"
INIT_RATIO = 0.4
SEED = 42
TOTAL_ROUNDS = 30


def make_cfg(dataset_name: str, dataset_path: str, model_type: str) -> dict:
    trainer_cfg = {
        "update_every_n_rounds": 1,
        "lr": 1e-4,
        "grad_clip": 1.0,
        "min_batch_size": 4,
        "max_neighbors": 30,
        "batch_subgraph_max_hop": 2,
        "scheduler": {"strategy": "constant"},
    }

    if model_type == "gnn_sum":
        model_cfg = {"type": "gnn", "hidden_dim": 32, "num_layers": 3,
                     "encoder_type": "layer_sum", "node_feat_dim": 0}
    elif model_type == "seal":
        model_cfg = {"type": "seal", "hidden_dim": 32, "num_layers": 3, "label_dim": 16}
    elif model_type == "graphsage_emb":
        model_cfg = {"type": "graphsage_emb", "emb_dim": 32, "hidden_dim": 32, "num_layers": 3}
    elif model_type == "gat_emb":
        model_cfg = {"type": "gat_emb", "emb_dim": 32, "hidden_dim": 32,
                     "num_layers": 3, "num_heads": 4}
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    return {
        "dataset": {"type": dataset_name, "path": dataset_path},
        "eval": {"degree_bins": 50, "graph_every_n": 10, "k_list": [1, 3, 5, 10]},
        "feedback": {"cooldown_mode": "decay", "cooldown_rounds": 5,
                     "p_neg": 0.0, "p_pos": 1.0},
        "init_edge_ratio": INIT_RATIO,
        "init_strategy": "random",
        "model": model_cfg,
        "recall": {"method": "two_hop_random", "top_k_recall": 100},
        "recommend": {"cold_start_random_fill": True, "top_k": 10},
        "replay": {"capacity": 0, "sample_n": 0},
        "runtime": {
            "device": "cpu",
            "log_every": 1,
            "out_dir": f"results/online/{SWEEP_TAG}/{dataset_name}_{model_type}",
            "seed": SEED,
        },
        "total_rounds": TOTAL_ROUNDS,
        "trainer": trainer_cfg,
        "user_selector": {
            "alpha": 0.5, "beta": 2.0, "gamma": 2.0,
            "lam": 0.1, "sample_ratio": 0.1,
            "strategy": "uniform", "w": 3,
        },
    }


def run_one(dataset_name: str, dataset_path: str, model_type: str) -> str:
    tag = f"{dataset_name}_{model_type}"
    out_dir = Path(f"results/online/{SWEEP_TAG}/{tag}")

    if (out_dir / "rounds.csv").exists():
        print(f"[SKIP] {tag}", flush=True)
        return tag

    cfg = make_cfg(dataset_name, dataset_path, model_type)
    cfg_dir = Path(f"configs/online/{SWEEP_TAG}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{tag}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"results/online/{SWEEP_TAG}") / f"{tag}.log"

    print(f"[RUN] {tag}", flush=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-u", "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    print(f"[{status}] {tag}", flush=True)
    return tag


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    tasks = [
        (ds, path, model)
        for ds, path in DATASETS
        for model in MODELS
        if not (Path(f"results/online/{SWEEP_TAG}/{ds}_{model}/rounds.csv")).exists()
    ]
    total_all = len(DATASETS) * len(MODELS)
    skip = total_all - len(tasks)
    print(f"共 {total_all} 个实验，跳过 {skip} 个，待跑 {len(tasks)} 个，workers={args.workers}",
          flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, ds, path, model): (ds, model)
                for ds, path, model in tasks}
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
            done += 1
            print(f"  进度 {done}/{len(tasks)}", flush=True)

    print(f"\n全部完成，结果在 results/online/{SWEEP_TAG}/", flush=True)
