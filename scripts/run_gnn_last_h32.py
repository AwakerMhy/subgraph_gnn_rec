"""补跑 GNN 变体：在各数据集 v2 sweep 目录中补充指定 encoder/hidden_dim 组合。

支持的变体（--variant）：
  gnn_h32        last + h32
  gnn_concat_h8  layer_concat + h8
  gnn_sum_h8     layer_sum + h8
"""
import json
import subprocess
import sys
from pathlib import Path
import yaml

PYTHON = sys.executable

DATASETS = [
    "college_msg", "email_eu", "bitcoin_alpha", "dnc_email", "wiki_vote",
    "slashdot", "sx_mathoverflow", "sx_askubuntu", "sx_superuser", "advogato",
]

VARIANTS = {
    "gnn_h32":       {"encoder_type": "last",         "hidden_dim": 32},
    "gnn_concat_h8": {"encoder_type": "layer_concat", "hidden_dim": 8},
    "gnn_sum_h8":    {"encoder_type": "layer_sum",    "hidden_dim": 8},
}


def make_cfg(dataset_name: str, tag: str) -> dict:
    ref = json.load(open(f"results/online/algo_sweep_{dataset_name}_v2/{dataset_name}_gnn/config.json"))
    cfg = dict(ref)
    v = VARIANTS[tag]
    cfg["model"] = {
        "type": "gnn",
        "encoder_type": v["encoder_type"],
        "hidden_dim": v["hidden_dim"],
        "num_layers": 3,
        "node_feat_dim": 0,
    }
    cfg["runtime"] = dict(ref["runtime"])
    cfg["runtime"]["out_dir"] = f"results/online/algo_sweep_{dataset_name}_v2/{dataset_name}_{tag}"
    return cfg


def run_one(dataset_name: str, tag: str) -> None:
    out_dir = Path(f"results/online/algo_sweep_{dataset_name}_v2/{dataset_name}_{tag}")
    if (out_dir / "rounds.csv").exists():
        print(f"[SKIP] {dataset_name}_{tag} already done", flush=True)
        return

    cfg = make_cfg(dataset_name, tag)
    cfg_path = Path(f"configs/online/sweep_v2_{dataset_name}_{tag}.yaml")
    cfg_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"{dataset_name}_{tag}.log"
    print(f"[RUN] {dataset_name}_{tag} ...", flush=True)
    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "-m", "src.online.loop", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )
    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    print(f"[{status}] {dataset_name}_{tag}", flush=True)


if __name__ == "__main__":
    import argparse
    import concurrent.futures
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", help="单个数据集名或 all")
    parser.add_argument("--variant", type=str, default="all",
                        help=f"变体名或 all，可选：{', '.join(VARIANTS)}")
    parser.add_argument("--workers", type=int, default=1, help="并行线程数")
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset != "all" else DATASETS
    variants = [args.variant] if args.variant != "all" else list(VARIANTS)
    tasks = [(ds, v) for ds in targets for v in variants]
    total = len(tasks)
    print(f"共 {total} 个实验，workers={args.workers}", flush=True)

    if args.workers == 1:
        for i, (ds, v) in enumerate(tasks, 1):
            run_one(ds, v)
            print(f"  进度 {i}/{total}", flush=True)
    else:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, ds, v): (ds, v) for ds, v in tasks}
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
                done += 1
                print(f"  进度 {done}/{total}", flush=True)
    print("全部完成", flush=True)
