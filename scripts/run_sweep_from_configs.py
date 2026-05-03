"""从预生成的 config yaml 批量运行 algo_sweep，逐个顺序执行（Windows 兼容）。

用法:
    python scripts/run_sweep_from_configs.py --dirs configs/online/algo_sweep_sx_askubuntu [configs/online/algo_sweep_sx_superuser ...]

每个 config 的 stdout/stderr 写入 results/online/algo_sweep_<dataset>/<dataset>_<model>.log
已存在 rounds.csv 的实验自动跳过。
"""
import argparse
import subprocess
import sys
from pathlib import Path

STANDARD_MODELS = {
    "random", "ground_truth", "cn", "aa", "jaccard", "pa",
    "mlp", "node_emb", "gnn", "gnn_concat", "gnn_sum",
}

PYTHON = sys.executable


def is_standard(cfg_path: Path) -> bool:
    stem = cfg_path.stem  # e.g. sx_askubuntu_gnn_sum
    parts = stem.split("_")
    # suffix after dataset name: join parts[2:] (works for sx_* which have 2-word prefix)
    # safer: check if stem ends with any known model suffix
    for model in STANDARD_MODELS:
        if stem.endswith("_" + model):
            return True
    return False


def run_sweep(cfg_dirs: list[str]) -> None:
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
    print(f"共 {total} 个实验", flush=True)

    for i, cfg_path in enumerate(tasks, 1):
        # 检查是否已有结果
        import yaml as _yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f)
        out_dir = Path(cfg.get("runtime", {}).get("out_dir", ""))
        rounds_csv = out_dir / "rounds.csv"
        if rounds_csv.exists():
            print(f"[SKIP {i}/{total}] {cfg_path.stem} (rounds.csv already exists)", flush=True)
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir.parent / f"{cfg_path.stem}.log"

        print(f"[RUN  {i}/{total}] {cfg_path.stem}", flush=True)
        with open(log_path, "w", encoding="utf-8") as fp:
            proc = subprocess.run(
                [PYTHON, "scripts/run_online_sim_win.py", "--config", str(cfg_path)],
                stdout=fp, stderr=fp,
            )
        status = "OK" if proc.returncode == 0 else f"ERR({proc.returncode})"
        print(f"       → {status}  log: {log_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirs", nargs="+", required=True, help="config 目录列表")
    args = parser.parse_args()
    run_sweep(args.dirs)
