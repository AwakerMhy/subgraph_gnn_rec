"""新数据集 v2 sweep：以 college_msg v2 为模板，替换数据集，跑全算法 × 多 seed × 多 init_ratio。
支持数据集：epinions、bitcoin_otc（及任意已处理数据集）。
"""
import json
import subprocess
import sys
import concurrent.futures
from pathlib import Path
import yaml

PYTHON = sys.executable

ALGOS = [
    "aa", "cn", "gnn", "gnn_concat", "gnn_concat_h8",
    "gnn_h32", "gnn_sum", "gnn_sum_h8", "ground_truth",
    "jaccard", "mlp", "node_emb", "pa", "random",
]

DATASET_PATHS = {
    "epinions":    "data/processed/epinions/edges.csv",
    "bitcoin_otc": "data/processed/bitcoin_otc/edges.csv",
}

INIT_RATIOS = [0.25, 0.4]
SEEDS = [42, 0, 1, 2, 3]


def sweep_name(dataset: str, init_ratio: float, seed: int) -> str:
    ir_tag = "" if abs(init_ratio - 0.25) < 1e-6 else f"_ir{int(init_ratio*100)}"
    return f"algo_sweep_{dataset}_v2{ir_tag}_s{seed}"


def make_cfg(dataset: str, algo: str, init_ratio: float, seed: int) -> dict:
    ref = json.load(open(f"results/online/algo_sweep_college_msg_v2/college_msg_{algo}/config.json"))
    cfg = dict(ref)
    cfg["dataset"] = {"type": dataset, "path": DATASET_PATHS[dataset]}
    cfg["init_edge_ratio"] = init_ratio
    cfg["runtime"] = dict(ref["runtime"])
    cfg["runtime"]["seed"] = seed
    cfg["runtime"]["device"] = "cuda"
    sweep = sweep_name(dataset, init_ratio, seed)
    cfg["runtime"]["out_dir"] = f"results/online/{sweep}/{dataset}_{algo}"
    return cfg


def run_one(dataset: str, algo: str, init_ratio: float, seed: int) -> str:
    sweep = sweep_name(dataset, init_ratio, seed)
    tag = f"{dataset}_{algo}"
    out_dir = Path(f"results/online/{sweep}/{tag}")

    if (out_dir / "rounds.csv").exists():
        print(f"[SKIP] {sweep}/{tag}", flush=True)
        return tag

    cfg = make_cfg(dataset, algo, init_ratio, seed)
    cfg_dir = Path(f"configs/online/{sweep}")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{tag}.yaml"
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(f"results/online/{sweep}") / f"{tag}.log"

    print(f"[RUN] {sweep}/{tag}", flush=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )
    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    print(f"[{status}] {sweep}/{tag}", flush=True)
    return tag


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", help="all / epinions / bitcoin_otc")
    parser.add_argument("--algo", default="all")
    parser.add_argument("--seeds", default="42,0,1,2,3")
    parser.add_argument("--init_ratios", default="0.25,0.4")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    datasets = list(DATASET_PATHS.keys()) if args.dataset == "all" else [args.dataset]
    algos    = ALGOS if args.algo == "all" else [args.algo]
    seeds    = [int(s) for s in args.seeds.split(",")]
    ratios   = [float(r) for r in args.init_ratios.split(",")]

    tasks = [(ds, al, ir, sd) for ir in ratios for sd in seeds for ds in datasets for al in algos]
    pending = [t for t in tasks
               if not (Path(f"results/online/{sweep_name(t[0], t[2], t[3])}/{t[0]}_{t[1]}/rounds.csv")).exists()]

    total = len(tasks)
    print(f"共 {total} 个实验，跳过 {total-len(pending)} 个，待跑 {len(pending)} 个，workers={args.workers}", flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, ds, al, ir, sd): (ds, al, ir, sd) for ds, al, ir, sd in pending}
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
            done += 1
            print(f"  进度 {done}/{len(pending)}", flush=True)

    print("全部完成", flush=True)
