"""scripts/run_algo_sweep.py — 各算法在小数据集上的对比（init/cooldown_mode 可配置，统一召回配置）。"""
import subprocess
import sys
import itertools
from pathlib import Path
import yaml
import concurrent.futures

PYTHON = sys.executable

DATASETS = [
    ("email_eu",      "data/processed/email_eu/edges.csv"),
    ("bitcoin_alpha", "data/processed/bitcoin_alpha/edges.csv"),
    ("dnc_email",     "data/processed/dnc_email/edges.csv"),
    ("college_msg",   "data/processed/college_msg/edges.csv"),
    ("wiki_vote",     "data/processed/wiki_vote/edges.csv"),
]

DATASETS_SBM = {
    "sbm5k": {
        "n_nodes": 5000, "n_communities": 5,
        "p_in": 0.3, "p_out": 0.05,
        "T": 2500, "edges_per_step": 40, "seed": 42,
    },
    "dcsbm5k": {
        "n_nodes": 5000, "n_communities": 5,
        "B_in": 1.0, "B_out": 0.05,
        "T": 2500, "edges_per_step": 40,
        "theta_alpha": 2.5, "seed": 42,
    },
}

# 数据集对应的生成器类型
DATASET_TYPE = {
    "sbm5k": "sbm",
    "dcsbm5k": "dcsbm",
}

MODELS = [
    "ground_truth",
    "random",
    "cn", "aa", "jaccard", "pa",
    "mlp",
    "node_emb",
    "gnn",
    "gnn_concat",
    "gnn_sum",
]

TOTAL_ROUNDS = 500  # default, overridden by --total_rounds
MAX_WORKERS  = 4


def make_cfg(dataset_name: str, dataset_path: str | None, model_type: str, init_ratio: float = 0.25, cooldown_mode: str = "decay", cooldown_rounds: int = 5) -> dict:
    model_cfg: dict = {"type": model_type}
    if model_type == "gnn":
        model_cfg.update({"hidden_dim": 8, "num_layers": 3, "encoder_type": "last", "node_feat_dim": 0})
    elif model_type == "gnn_concat":
        model_cfg.update({"type": "gnn", "hidden_dim": 8, "num_layers": 3, "encoder_type": "layer_concat", "node_feat_dim": 0})
    elif model_type == "gnn_sum":
        model_cfg.update({"type": "gnn", "hidden_dim": 8, "num_layers": 3, "encoder_type": "layer_sum", "node_feat_dim": 0})
    elif model_type == "node_emb":
        model_cfg.update({"emb_dim": 64, "hidden_dim": 64})
    elif model_type == "mlp":
        model_cfg.update({"hidden_dim": 64})

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
        "dataset": ({"type": DATASET_TYPE.get(dataset_name, "sbm"), "params": DATASETS_SBM[dataset_name]}
                    if dataset_name in DATASETS_SBM
                    else {"type": dataset_name, "path": dataset_path}),
        "eval": {"degree_bins": 50, "graph_every_n": 10, "k_list": [1, 3, 5, 10]},
        "feedback": {
            "cooldown_mode": cooldown_mode,
            "cooldown_rounds": cooldown_rounds,
            "p_neg": 0.0,
            "p_pos": 1.0,
        },
        "init_edge_ratio": init_ratio,
        "init_strategy": "stratified",
        "model": model_cfg,
        "recall": {"method": "two_hop_random", "top_k_recall": 100},
        "recommend": {"cold_start_random_fill": True, "top_k": 10},
        "replay": {"capacity": 0, "sample_n": 0},
        "runtime": {
            "device": "cpu",
            "log_every": 50,
            "out_dir": f"results/online/algo_sweep_init{int(init_ratio*100):03d}_cd_{cooldown_mode}{cooldown_rounds}/{dataset_name}_{model_type}",
            "seed": 42,
        },
        "total_rounds": TOTAL_ROUNDS,
        "trainer": trainer_cfg,
        "user_selector": {
            "alpha": 0.5, "beta": 2.0, "gamma": 2.0,
            "lam": 0.1, "sample_ratio": 0.1,
            "strategy": "composite", "w": 3,
        },
    }


def run_one(args: tuple[str, str, str, float, str, int, str, int]) -> str:
    dataset_name, dataset_path, model_type, init_ratio, cooldown_mode, cooldown_rounds, dataset_tag, total_rounds = args
    init_tag = f"init{int(init_ratio * 100):03d}"
    cooldown_tag = f"cd_{cooldown_mode}{cooldown_rounds}"
    sweep_tag = f"algo_sweep_{dataset_tag}_{init_tag}_{cooldown_tag}"
    tag = f"{dataset_name}_{model_type}"

    cfg = make_cfg(dataset_name, dataset_path, model_type, init_ratio, cooldown_mode, cooldown_rounds)
    cfg["total_rounds"] = total_rounds
    cfg["runtime"]["out_dir"] = f"results/online/{sweep_tag}/{tag}"

    cfg_dir = Path(f"configs/online/{sweep_tag}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{tag}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    log_dir = Path(f"results/online/{sweep_tag}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )

    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    print(f"[{status}] {sweep_tag}/{tag}", flush=True)
    return tag


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=float, default=0.25)
    parser.add_argument("--cooldown_mode", type=str, default="decay", choices=["decay", "hard"])
    parser.add_argument("--cooldown_rounds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="real", help="real | sbm5k | dcsbm5k | <单个数据集名>")
    parser.add_argument("--exclude_models", type=str, default="", help="逗号分隔，如 gnn,gnn_concat,mlp")
    parser.add_argument("--total_rounds", type=int, default=TOTAL_ROUNDS)
    args = parser.parse_args()
    init_ratio = args.init
    cooldown_mode = args.cooldown_mode
    cooldown_rounds = args.cooldown_rounds
    dataset_tag = args.dataset
    exclude = set(args.exclude_models.split(",")) - {""} if args.exclude_models else set()
    total_rounds = args.total_rounds
    init_tag = f"init{int(init_ratio * 100):03d}"
    cooldown_tag = f"cd_{cooldown_mode}{cooldown_rounds}"
    rounds_tag = f"r{total_rounds}"
    sweep_tag = f"algo_sweep_{dataset_tag}_{init_tag}_{cooldown_tag}_{rounds_tag}"

    if dataset_tag in DATASETS_SBM:
        dataset_list = [(dataset_tag, None)]
    else:
        # 支持单数据集名（如 wiki_vote）
        single = next(((ds, p) for ds, p in DATASETS if ds == dataset_tag), None)
        dataset_list = [single] if single else DATASETS

    active_models = [m for m in MODELS if m not in exclude]
    all_tasks = [
        (ds, path, model, init_ratio, cooldown_mode, cooldown_rounds, dataset_tag, total_rounds)
        for (ds, path), model in itertools.product(dataset_list, active_models)
    ]
    tasks = [t for t in all_tasks
             if not Path(f"results/online/{sweep_tag}/{t[0]}_{t[2]}/rounds.csv").exists()]
    total = len(tasks)
    print(f"dataset={dataset_tag}  init={init_ratio}  cooldown_mode={cooldown_mode}  cooldown_rounds={cooldown_rounds}  共 {total} 个实验，最多 {MAX_WORKERS} 并行", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, result in enumerate(pool.map(run_one, tasks), 1):
            print(f"  进度 {i}/{total}: {result}", flush=True)

    print(f"\n全部完成，结果在 results/online/{sweep_tag}/", flush=True)
