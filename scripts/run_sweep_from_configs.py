"""从预生成的 config yaml 批量运行 algo_sweep，支持并行（Windows 兼容）。

用法:
    python scripts/run_sweep_from_configs.py --dirs configs/online/algo_sweep_sx_askubuntu [...]
    python scripts/run_sweep_from_configs.py --dirs configs/online/algo_sweep_digg_v2 --workers 4

每个 config 的 stdout/stderr 写入 results/online/algo_sweep_<dataset>/<dataset>_<model>.log
已存在 rounds.csv 的实验自动跳过。
--workers N  并发运行 N 个实验（默认 1，用 ThreadPoolExecutor，Windows 安全）
"""
import argparse
import concurrent.futures
import subprocess
import sys
import yaml
from pathlib import Path

STANDARD_MODELS = {
    "random", "ground_truth", "cn", "aa", "jaccard", "pa",
    "mlp", "node_emb", "gnn", "gnn_concat", "gnn_sum",
}

PYTHON = sys.executable


def is_standard(cfg_path: Path) -> bool:
    stem = cfg_path.stem
    for model in STANDARD_MODELS:
        if stem.endswith("_" + model):
            return True
    return False


def _run_one(cfg_path: Path) -> tuple[str, str]:
    """运行单个实验，返回 (status, log_path_str)。"""
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    out_dir = Path(cfg.get("runtime", {}).get("out_dir", ""))
    rounds_csv = out_dir / "rounds.csv"
    if rounds_csv.exists():
        return "SKIP", ""

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"{cfg_path.stem}.log"

    with open(log_path, "w", encoding="utf-8") as fp:
        proc = subprocess.run(
            [PYTHON, "scripts/run_online_sim_win.py", "--config", str(cfg_path)],
            stdout=fp, stderr=fp,
        )
    status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
    return status, str(log_path)


def run_sweep(cfg_dirs: list[str], workers: int = 1) -> None:
    tasks: list[Path] = []
    for d in cfg_dirs:
        p = Path(d)
        if not p.exists():
            print(f"[WARN] 目录不存在，跳过: {p}", flush=True)
            continue
        for yaml_file in sorted(p.glob("*.yaml")):
            if is_standard(yaml_file):
                tasks.append(yaml_file)

    total = len(tasks)
    print(f"共 {total} 个实验，并发 workers={workers}", flush=True)

    if workers <= 1:
        for i, cfg_path in enumerate(tasks, 1):
            status, log_path = _run_one(cfg_path)
            if status == "SKIP":
                print(f"[SKIP {i}/{total}] {cfg_path.stem}", flush=True)
            else:
                print(f"[{status} {i}/{total}] {cfg_path.stem}  log: {log_path}", flush=True)
        return

    # 并行模式：ThreadPoolExecutor（每个任务是独立子进程，Windows 安全）
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_cfg = {pool.submit(_run_one, cfg): cfg for cfg in tasks}
        for fut in concurrent.futures.as_completed(future_to_cfg):
            cfg_path = future_to_cfg[fut]
            done += 1
            try:
                status, log_path = fut.result()
            except Exception as exc:
                status, log_path = f"EXC({exc})", ""
            if status == "SKIP":
                print(f"[SKIP {done}/{total}] {cfg_path.stem}", flush=True)
            else:
                print(f"[{status} {done}/{total}] {cfg_path.stem}  log: {log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True, help="config 目录列表")
    parser.add_argument("--workers", type=int, default=1,
                        help="并发实验数（默认 1=串行）")
    args = parser.parse_args()
    run_sweep(args.dirs, workers=args.workers)
