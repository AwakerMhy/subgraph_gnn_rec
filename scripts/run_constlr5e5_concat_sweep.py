"""gnn_concat constant lr=5e-5 消融：验证是否优于 1e-4。

结果写入 results/online/algo_sweep_{dataset}_v2_constlr5e5_s{seed}/
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

LR = 5e-5


def sweep_name(dataset: str, seed: int) -> str:
    return f"algo_sweep_{dataset}_v2_constlr5e5_s{seed}"


def run_one(dataset: str, seed: int) -> str:
    algo = "gnn_concat"
    tag = f"{dataset}_{algo}"
    sweep = sweep_name(dataset, seed)
    out_dir = Path(f"results/online/{sweep}/{tag}")

    if (out_dir / "rounds.csv").exists():
        print(f"[SKIP] {sweep}/{tag}", flush=True)
        return tag

    ref_path = Path(f"results/online/algo_sweep_{dataset}_v2/{tag}/config.json")
    if not ref_path.exists():
        print(f"[MISS] {ref_path}", flush=True)
        return tag

    cfg = json.load(open(ref_path))
    cfg["trainer"] = dict(cfg["trainer"])
    cfg["trainer"]["lr"] = LR
    cfg["trainer"]["scheduler"] = {"strategy": "constant"}
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
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [(ds, sd) for sd in seeds for ds in DATASETS]
    pending = [t for t in tasks
               if not (Path(f"results/online/{sweep_name(t[0], t[1])}/{t[0]}_gnn_concat/rounds.csv")).exists()]

    print(f"lr={LR}  gnn_concat  共 {len(tasks)} 个，跳过 {len(tasks)-len(pending)} 个，待跑 {len(pending)} 个，workers={args.workers}", flush=True)

    if args.workers <= 1:
        for i, (ds, sd) in enumerate(pending, 1):
            run_one(ds, sd)
            print(f"  进度 {i}/{len(pending)}", flush=True)
    else:
        done = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(run_one, ds, sd): (ds, sd) for ds, sd in pending}
            for fut in concurrent.futures.as_completed(futs):
                fut.result()
                done += 1
                print(f"  进度 {done}/{len(pending)}", flush=True)

    print("全部完成", flush=True)
