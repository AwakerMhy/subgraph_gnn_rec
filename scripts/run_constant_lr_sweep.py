"""constant lr=1e-4 消融实验：在 v2 配置基础上替换 scheduler 为 constant lr。

从各数据集 algo_sweep_{dataset}_v2 中读取已有 GNN config，
替换 trainer.lr = 1e-4 和 trainer.scheduler.strategy = constant，
结果写入：
  results/online/algo_sweep_{dataset}_v2_constlr_s{seed}/

用法:
    python scripts/run_constant_lr_sweep.py
    python scripts/run_constant_lr_sweep.py --seeds 0,1,2 --workers 4
    python scripts/run_constant_lr_sweep.py --datasets college_msg,wiki_vote --seeds 0
"""
import json
import subprocess
import sys
import concurrent.futures
from pathlib import Path
import yaml
import argparse

PYTHON = sys.executable

DATASETS = [
    "college_msg", "email_eu", "bitcoin_alpha", "dnc_email", "wiki_vote",
    "slashdot", "sx_mathoverflow", "sx_askubuntu", "sx_superuser", "advogato",
]

GNN_ALGOS = ["gnn", "gnn_h32", "gnn_concat", "gnn_sum"]

CONSTANT_LR = 1e-4


def sweep_name(dataset: str, seed: int) -> str:
    return f"algo_sweep_{dataset}_v2_constlr_s{seed}"


def run_one(dataset: str, algo: str, seed: int) -> str:
    tag = f"{dataset}_{algo}"
    sweep = sweep_name(dataset, seed)
    out_dir = Path(f"results/online/{sweep}/{tag}")

    if (out_dir / "rounds.csv").exists():
        print(f"[SKIP] {sweep}/{tag}", flush=True)
        return tag

    ref_path = Path(f"results/online/algo_sweep_{dataset}_v2/{tag}/config.json")
    if not ref_path.exists():
        print(f"[MISS] 参考 config 不存在: {ref_path}", flush=True)
        return tag

    cfg = json.load(open(ref_path))

    # 覆盖 lr 和 scheduler 为 constant
    cfg["trainer"] = dict(cfg["trainer"])
    cfg["trainer"]["lr"] = CONSTANT_LR
    cfg["trainer"]["scheduler"] = {"strategy": "constant"}

    # 更新 seed 和 out_dir
    cfg["runtime"] = dict(cfg["runtime"])
    cfg["runtime"]["seed"] = seed
    cfg["runtime"]["out_dir"] = str(out_dir)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="all", help="逗号分隔数据集，默认全部10个")
    parser.add_argument("--algos", default="all", help="逗号分隔算法，默认gnn/gnn_h32/gnn_concat/gnn_sum")
    parser.add_argument("--seeds", default="0", help="逗号分隔seeds，默认0")
    parser.add_argument("--workers", type=int, default=1, help="并发数（默认1=串行）")
    args = parser.parse_args()

    targets = DATASETS if args.datasets == "all" else args.datasets.split(",")
    algos   = GNN_ALGOS if args.algos == "all" else args.algos.split(",")
    seeds   = [int(s) for s in args.seeds.split(",")]

    tasks = [(ds, al, sd) for sd in seeds for ds in targets for al in algos]
    pending = [t for t in tasks
               if not (Path(f"results/online/{sweep_name(t[0], t[2])}/{t[0]}_{t[1]}/rounds.csv")).exists()]

    total = len(tasks)
    skip  = total - len(pending)
    print(f"constant lr={CONSTANT_LR}  共 {total} 个实验，跳过 {skip} 个，待跑 {len(pending)} 个，workers={args.workers}", flush=True)

    if args.workers <= 1:
        for i, (ds, al, sd) in enumerate(pending, 1):
            run_one(ds, al, sd)
            print(f"  进度 {i}/{len(pending)}", flush=True)
    else:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, ds, al, sd): (ds, al, sd) for ds, al, sd in pending}
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
                done += 1
                print(f"  进度 {done}/{len(pending)}", flush=True)

    print("全部完成", flush=True)
