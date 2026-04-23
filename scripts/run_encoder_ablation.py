"""在 Bitcoin-OTC 和 Email-EU 上对比三种 encoder_type。

用法（gnn 环境中）：
    python scripts/run_encoder_ablation.py [--epochs N] [--max_samples N]
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# 直接使用 gnn conda 环境的 Python，避免 conda run 的 GBK 编码问题
GNN_PYTHON = r"C:\conda\envs\gnn\python.exe"


def run(cmd: list[str], env: dict | None = None) -> int:
    print(f"\n{'='*60}")
    print("CMD:", " ".join(cmd))
    print("="*60, flush=True)
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    datasets = {
        "bitcoin_otc": "data/processed/bitcoin_otc",
        "email_eu":    "data/processed/email_eu",
    }
    encoders = ["last", "layer_concat", "layer_sum"]

    failed = []
    for ds_name, data_dir in datasets.items():
        for enc in encoders:
            run_name = f"{ds_name}_{enc}"
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            cmd = [
                GNN_PYTHON, "-m", "src.train",
                "--data_dir", data_dir,
                "--run_name", run_name,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--hidden_dim", str(args.hidden_dim),
                "--num_layers", str(args.num_layers),
                "--patience", str(args.patience),
                "--encoder_type", enc,
                "--seed", "42",
            ]
            if args.max_samples > 0:
                cmd += ["--max_samples", str(args.max_samples)]
            rc = run(cmd, env)
            if rc != 0:
                failed.append(run_name)

    print("\n" + "="*60)
    if failed:
        print(f"以下实验失败：{failed}")
        sys.exit(1)
    else:
        print("所有实验完成。")


if __name__ == "__main__":
    main()
