"""scripts/visualize_online_run.py — 在线学习实验结果可视化。

用法：
    python -m scripts.visualize_online_run --run_dir results/online/college_msg_full

输出（写入 <run_dir>/figures/）：
    coverage_curve.png     — coverage / precision / loss 三条曲线
    degree_kl_evolution.png — degree KL 随轮次曲线
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def plot_coverage_curve(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(df["round"], df["coverage"], color="steelblue", label="coverage", linewidth=2)
    if "precision_k" in df.columns:
        ax1.plot(df["round"], df["precision_k"], color="darkorange",
                 linestyle="--", label="precision@K", linewidth=1.5)
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Coverage / Precision@K", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.legend(loc="upper left")

    if "loss" in df.columns:
        loss = df["loss"].replace([np.inf, -np.inf], np.nan)
        ax2.plot(df["round"], loss, color="crimson", alpha=0.6, label="loss", linewidth=1)
        ax2.set_ylabel("Loss", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.legend(loc="upper right")

    ax1.set_title("Coverage / Precision / Loss over Rounds")
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_curve.png", dpi=120)
    plt.close(fig)
    print(f"  Saved coverage_curve.png")


def plot_degree_kl(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    if "degree_kl" not in df.columns:
        print("  degree_kl not found, skipping")
        return
    sub = df.dropna(subset=["degree_kl"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(sub["round"], sub["degree_kl"], color="purple", linewidth=2)
    ax.set_xlabel("Round")
    ax.set_ylabel("Degree KL Divergence")
    ax.set_title("Degree Distribution KL vs G*")
    fig.tight_layout()
    fig.savefig(out_dir / "degree_kl_evolution.png", dpi=120)
    plt.close(fig)
    print(f"  Saved degree_kl_evolution.png")


def plot_new_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # hit_rate, rec_coverage, novelty
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metrics = [
        ("hit_rate@5", "Hit Rate@5", "teal"),
        ("rec_coverage@5", "Rec Coverage@5", "olive"),
        ("novelty", "Novelty (avg path len)", "sienna"),
    ]
    for ax, (col, title, color) in zip(axes, metrics):
        if col in df.columns:
            sub = df.dropna(subset=[col])
            ax.plot(sub["round"], sub[col], color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Round")
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_metrics.png", dpi=120)
    plt.close(fig)
    print(f"  Saved diversity_metrics.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="实验输出目录（含 rounds.csv）")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    csv_path = run_dir / "rounds.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"Generating figures in {out_dir} ...")
    plot_coverage_curve(df, out_dir)
    plot_degree_kl(df, out_dir)
    plot_new_metrics(df, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
