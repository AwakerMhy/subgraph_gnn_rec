"""scripts/run_gt_init_sweep.py — ground_truth 精排在不同数据集 × init_edge_ratio 下的 coverage 扫描。"""
import subprocess
import sys
import itertools
from pathlib import Path
import yaml
import concurrent.futures

PYTHON = sys.executable

DATASETS = [
    ("college_msg",   "data/processed/college_msg/edges.csv"),
    ("bitcoin_otc",   "data/processed/bitcoin_otc/edges.csv"),
    ("bitcoin_alpha", "data/processed/bitcoin_alpha/edges.csv"),
    ("dnc_email",     "data/processed/dnc_email/edges.csv"),
    ("email_eu",      "data/processed/email_eu/edges.csv"),
]

INIT_RATIOS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

TOTAL_ROUNDS = 500
MAX_WORKERS  = 4   # 并行进程数，避免 CPU 过载


def make_cfg(dataset_name: str, dataset_path: str, init_ratio: float) -> dict:
    ratio_tag = f"{int(init_ratio * 100):03d}"
    return {
        "dataset": {"type": dataset_name, "path": dataset_path},
        "eval": {"degree_bins": 50, "graph_every_n": 10, "k_list": [1, 3, 5, 10]},
        "feedback": {
            "cooldown_mode": "decay",
            "cooldown_rounds": 5,
            "p_neg": 0.0,
            "p_pos": 0.95,
        },
        "init_edge_ratio": init_ratio,
        "init_strategy": "stratified",
        "model": {"type": "ground_truth"},
        "recall": {"method": "two_hop_random", "top_k_recall": 100},
        "recommend": {"cold_start_random_fill": True, "top_k": 10},
        "replay": {"capacity": 0, "sample_n": 0},
        "runtime": {
            "device": "cpu",
            "log_every": 50,
            "out_dir": f"results/online/gt_sweep/{dataset_name}_init{ratio_tag}",
            "seed": 42,
        },
        "total_rounds": TOTAL_ROUNDS,
        "trainer": {"update_every_n_rounds": 1},
        "user_selector": {
            "alpha": 0.5, "beta": 2.0, "gamma": 2.0,
            "lam": 0.1, "sample_ratio": 0.1,
            "strategy": "composite", "w": 3,
        },
    }


def run_one(args: tuple[str, str, float]) -> str:
    dataset_name, dataset_path, init_ratio = args
    ratio_tag = f"{int(init_ratio * 100):03d}"
    tag = f"{dataset_name}_init{ratio_tag}"

    cfg = make_cfg(dataset_name, dataset_path, init_ratio)
    cfg_dir = Path("configs/online/gt_sweep")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{tag}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    log_dir = Path("results/online/gt_sweep")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{tag}.log"

    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )

    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    print(f"[{status}] {tag}", flush=True)
    return tag


if __name__ == "__main__":
    combos = list(itertools.product(DATASETS, INIT_RATIOS))
    tasks  = [(ds, path, ratio) for (ds, path), ratio in combos]
    total  = len(tasks)
    print(f"共 {total} 个实验，最多 {MAX_WORKERS} 并行", flush=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for i, result in enumerate(pool.map(run_one, tasks), 1):
            print(f"  进度 {i}/{total}: {result}", flush=True)

    print("\n全部完成，结果在 results/online/gt_sweep/", flush=True)
