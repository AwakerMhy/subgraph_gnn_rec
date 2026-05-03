"""scripts/run_init_strategy_sweep.py — 在小数据集上对比不同 init_strategy 对各算法的影响。

用法：
    python scripts/run_init_strategy_sweep.py                        # 全量
    python scripts/run_init_strategy_sweep.py --dataset college_msg  # 单数据集
    python scripts/run_init_strategy_sweep.py --strategies random stratified snowball
"""
import argparse
import subprocess
import sys
import itertools
from pathlib import Path

import yaml

PYTHON = sys.executable

DATASETS = [
    ("college_msg",   "data/processed/college_msg/edges.csv"),
    ("bitcoin_alpha", "data/processed/bitcoin_alpha/edges.csv"),
    ("dnc_email",     "data/processed/dnc_email/edges.csv"),
]

MODELS = [
    "ground_truth",
    "random",
    "cn", "aa", "jaccard", "pa",
    "gnn",
]

INIT_STRATEGIES = ["random", "stratified", "snowball", "forest_fire", "all_covered"]

TOTAL_ROUNDS = 40
INIT_RATIO   = 0.25
MAX_WORKERS  = 4


def make_cfg(dataset_name: str, dataset_path: str, model_type: str, init_strategy: str, total_rounds: int) -> dict:
    model_cfg: dict = {"type": model_type}
    if model_type == "gnn":
        model_cfg.update({"hidden_dim": 8, "num_layers": 3, "encoder_type": "last", "node_feat_dim": 0})

    trainer_cfg: dict = {
        "update_every_n_rounds": 1,
        "lr": 0.001,
        "grad_clip": 1.0,
        "min_batch_size": 4,
        "max_neighbors": 30,
        "batch_subgraph_max_hop": 2,
        "scheduler": {"strategy": "cosine_warmup", "warmup_rounds": 5, "min_lr": 1e-5},
    }

    return {
        "dataset": {"type": dataset_name, "path": dataset_path},
        "eval": {"degree_bins": 50, "graph_every_n": 10, "k_list": [1, 3, 5, 10]},
        "feedback": {
            "cooldown_mode": "decay",
            "cooldown_rounds": 5,
            "p_neg": 0.0,
            "p_pos": 1.0,
        },
        "init_edge_ratio": INIT_RATIO,
        "init_strategy": init_strategy,
        "model": model_cfg,
        "recall": {"method": "two_hop_random", "top_k_recall": 100},
        "recommend": {"cold_start_random_fill": True, "top_k": 10},
        "replay": {"capacity": 0, "sample_n": 0},
        "runtime": {
            "device": "cpu",
            "log_every": 1,
            "out_dir": f"results/online/init_strategy_sweep_{dataset_name}/{init_strategy}_{model_type}",
            "seed": 42,
        },
        "total_rounds": total_rounds,
        "trainer": trainer_cfg,
        "user_selector": {
            "alpha": 0.5, "beta": 2.0, "gamma": 2.0,
            "lam": 0.1, "sample_ratio": 0.1,
            "strategy": "uniform", "w": 3,
        },
    }


def run_one(args: tuple) -> str:
    dataset_name, dataset_path, model_type, init_strategy, total_rounds = args
    tag = f"{init_strategy}_{model_type}"

    cfg = make_cfg(dataset_name, dataset_path, model_type, init_strategy, total_rounds)
    cfg_dir = Path(f"configs/online/init_strategy_sweep_{dataset_name}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{tag}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    log_dir = Path(f"results/online/init_strategy_sweep_{dataset_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )

    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    label = f"init_strategy_sweep_{dataset_name}/{tag}"
    print(f"[{status}] {label}", flush=True)
    return tag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all",
                        help="all | college_msg | bitcoin_alpha | dnc_email")
    parser.add_argument("--strategies", nargs="+", default=INIT_STRATEGIES,
                        choices=INIT_STRATEGIES,
                        help="要跑的 init_strategy，默认全部")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="要跑的 model 列表")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--total_rounds", type=int, default=TOTAL_ROUNDS)
    args = parser.parse_args()

    dataset_list = DATASETS if args.dataset == "all" else [
        (ds, p) for ds, p in DATASETS if ds == args.dataset
    ]
    if not dataset_list:
        print(f"未知数据集: {args.dataset}")
        sys.exit(1)

    all_tasks = [
        (ds, path, model, strategy, args.total_rounds)
        for (ds, path), model, strategy in itertools.product(dataset_list, args.models, args.strategies)
    ]
    tasks = [
        t for t in all_tasks
        if not Path(f"results/online/init_strategy_sweep_{t[0]}/{t[3]}_{t[2]}/rounds.csv").exists()
    ]

    total = len(tasks)
    done_count = len(all_tasks) - total
    print(f"共 {len(all_tasks)} 个实验，已完成 {done_count}，待运行 {total}，最多 {args.workers} 并行", flush=True)

    if args.workers == 1:
        for i, t in enumerate(tasks, 1):
            result = run_one(t)
            print(f"  进度 {i}/{total}: {result}", flush=True)
    else:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, result in enumerate(pool.map(run_one, tasks), 1):
                print(f"  进度 {i}/{total}: {result}", flush=True)

    print(f"\n全部完成。", flush=True)
    for ds, _ in dataset_list:
        print(f"  python scripts/plot_init_strategy_sweep.py --dataset {ds}")
