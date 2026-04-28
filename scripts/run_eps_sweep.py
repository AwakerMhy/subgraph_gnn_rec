"""scripts/run_eps_sweep.py — epsilon-greedy × GNN-sum 在 init=0.25 下的扫描。"""
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
]

# (epsilon_start, epsilon_end) 组合
EPSILONS = [
    (0.0,  0.0),   # baseline（无探索）
    (0.1,  0.0),
    (0.3,  0.0),
    (0.5,  0.0),
    (0.5,  0.1),   # 保留少量探索到最后
]

TOTAL_ROUNDS = 500
MAX_WORKERS  = 4


def eps_tag(eps_start: float, eps_end: float) -> str:
    s = f"eps{int(eps_start*10):02d}"
    if eps_end > 0:
        s += f"_{int(eps_end*10):02d}"
    return s


def make_cfg(dataset_name: str, dataset_path: str, eps_start: float, eps_end: float) -> dict:
    tag = eps_tag(eps_start, eps_end)
    return {
        "dataset": {"type": dataset_name, "path": dataset_path},
        "eval": {"degree_bins": 50, "graph_every_n": 10, "k_list": [1, 3, 5, 10]},
        "feedback": {
            "cooldown_mode": "decay",
            "cooldown_rounds": 5,
            "p_neg": 0.0,
            "p_pos": 1.0,
        },
        "init_edge_ratio": 0.25,
        "init_strategy": "stratified",
        "model": {
            "type": "gnn",
            "hidden_dim": 8,
            "num_layers": 3,
            "encoder_type": "layer_sum",
            "node_feat_dim": 0,
        },
        "recall": {"method": "two_hop_random", "top_k_recall": 100},
        "recommend": {"cold_start_random_fill": True, "top_k": 10},
        "replay": {"capacity": 0, "sample_n": 0},
        "runtime": {
            "device": "cpu",
            "log_every": 50,
            "out_dir": f"results/online/eps_sweep_tanh/{dataset_name}_{tag}",
            "seed": 42,
        },
        "total_rounds": TOTAL_ROUNDS,
        "trainer": {
            "update_every_n_rounds": 1,
            "lr": 0.001,
            "grad_clip": 1.0,
            "min_batch_size": 4,
            "max_neighbors": 30,
            "batch_subgraph_max_hop": 2,
            "epsilon_start": eps_start,
            "epsilon_end": eps_end,
            "scheduler": {"strategy": "cosine_warmup", "warmup_rounds": 5, "min_lr": 1e-5},
        },
        "user_selector": {
            "alpha": 0.5, "beta": 2.0, "gamma": 2.0,
            "lam": 0.1, "sample_ratio": 0.1,
            "strategy": "composite", "w": 3,
        },
    }


def run_one(args: tuple) -> str:
    dataset_name, dataset_path, eps_start, eps_end = args
    tag = eps_tag(eps_start, eps_end)
    run_tag = f"{dataset_name}_{tag}"

    cfg = make_cfg(dataset_name, dataset_path, eps_start, eps_end)
    cfg_dir = Path("configs/online/eps_sweep")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{run_tag}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    log_dir = Path("results/online/eps_sweep_tanh")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_tag}.log"

    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )

    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    print(f"[{status}] {run_tag}", flush=True)
    return run_tag


if __name__ == "__main__":
    tasks = [
        (ds, path, eps_s, eps_e)
        for (ds, path), (eps_s, eps_e) in itertools.product(DATASETS, EPSILONS)
        if not Path(f"results/online/eps_sweep_tanh/{ds}_{eps_tag(eps_s, eps_e)}/rounds.csv").exists()
    ]
    total = len(tasks)
    print(f"共 {total} 个实验，最多 {MAX_WORKERS} 并行", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, result in enumerate(pool.map(run_one, tasks), 1):
            print(f"  进度 {i}/{total}: {result}", flush=True)

    print("\n全部完成，结果在 results/online/eps_sweep_tanh/", flush=True)
