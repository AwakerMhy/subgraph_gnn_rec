"""多 seed 重复实验：在 v2 配置基础上用不同 seed / init_ratio 重跑所有算法。

从各数据集 algo_sweep_{dataset}_v2 中读取已有 config，
替换 seed、init_edge_ratio 和 out_dir，结果写入：
  - init_ratio=0.25：algo_sweep_{dataset}_v2_s{seed}/
  - 其他 init_ratio：algo_sweep_{dataset}_v2_ir{ir}_s{seed}/
"""
import json
import subprocess
import sys
import concurrent.futures
from pathlib import Path
import yaml

PYTHON = sys.executable

DATASETS = [
    "college_msg", "email_eu", "bitcoin_alpha", "dnc_email", "wiki_vote",
    "slashdot", "sx_mathoverflow", "sx_askubuntu", "sx_superuser", "advogato",
]

ALGOS = [
    "aa", "cn", "gnn", "gnn_concat", "gnn_concat_h8",
    "gnn_h32", "gnn_sum", "gnn_sum_h8", "ground_truth",
    "jaccard", "mlp", "node_emb", "pa", "random",
]


def sweep_name(dataset: str, init_ratio: float, seed: int) -> str:
    ir_tag = "" if abs(init_ratio - 0.25) < 1e-6 else f"_ir{int(init_ratio*100)}"
    return f"algo_sweep_{dataset}_v2{ir_tag}_s{seed}"


def run_one(dataset: str, algo: str, seed: int, init_ratio: float) -> str:
    tag = f"{dataset}_{algo}"
    sweep = sweep_name(dataset, init_ratio, seed)
    out_dir = Path(f"results/online/{sweep}/{tag}")

    if (out_dir / "rounds.csv").exists():
        print(f"[SKIP] {sweep}/{tag}", flush=True)
        return tag

    ref_path = Path(f"results/online/algo_sweep_{dataset}_v2/{tag}/config.json")
    if not ref_path.exists():
        print(f"[MISS] 参考 config 不存在: {ref_path}", flush=True)
        return tag

    cfg = json.load(open(ref_path))
    cfg["init_edge_ratio"] = init_ratio
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--algo", default="all")
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--init_ratio", type=float, default=0.25)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    targets = [args.dataset] if args.dataset != "all" else DATASETS
    algos   = [args.algo]    if args.algo    != "all" else ALGOS
    seeds   = [int(s) for s in args.seeds.split(",")]

    tasks = [(ds, al, sd) for sd in seeds for ds in targets for al in algos]
    pending = [t for t in tasks
               if not (Path(f"results/online/{sweep_name(t[0], args.init_ratio, t[2])}/{t[0]}_{t[1]}/rounds.csv")).exists()]

    total = len(tasks)
    skip  = total - len(pending)
    print(f"init_ratio={args.init_ratio}  共 {total} 个实验，跳过 {skip} 个，待跑 {len(pending)} 个，workers={args.workers}", flush=True)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, ds, al, sd, args.init_ratio): (ds, al, sd) for ds, al, sd in pending}
        for fut in concurrent.futures.as_completed(futs):
            fut.result()
            done += 1
            print(f"  进度 {done}/{len(pending)}", flush=True)

    print("全部完成", flush=True)
