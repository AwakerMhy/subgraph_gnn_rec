"""scripts/plot_init_ratio_sweep.py — 对比 init_edge_ratio 对各算法 coverage/MRR 的影响。

生成两类图：
  1. per-algorithm: 同一算法，不同 init_ratio 的曲线 (coverage vs round)
  2. summary bar: 最终轮 coverage / mrr@10，按 ratio 分组，各算法一条柱

用法：
    python scripts/plot_init_ratio_sweep.py --dataset college_msg
    python scripts/plot_init_ratio_sweep.py --dataset college_msg --metric coverage mrr@10
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

INIT_RATIOS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
MODELS = ["ground_truth", "random", "cn", "aa", "jaccard", "pa", "gnn"]


def ratio_tag(r: float) -> str:
    return f"r{int(r * 100):03d}"


def make_colors(ratios):
    cmap = cm.get_cmap("plasma", len(ratios))
    return {r: cmap(i) for i, r in enumerate(ratios)}


def load_data(base_dir: Path, ratios: list[float]) -> dict[tuple[float, str], pd.DataFrame]:
    data = {}
    for ratio in ratios:
        for model in MODELS:
            csv = base_dir / f"{ratio_tag(ratio)}_{model}" / "rounds.csv"
            if csv.exists():
                data[(ratio, model)] = pd.read_csv(csv)
    return data


def plot_per_algorithm(data, base_dir: Path, metrics: list[str], dataset: str, ratios: list[float]):
    present_models = [m for m in MODELS if any((r, m) in data for r in ratios)]
    if not present_models:
        return

    colors = make_colors(ratios)
    n_metrics = len(metrics)
    n_models = len(present_models)

    fig, axes = plt.subplots(n_metrics, n_models,
                             figsize=(4 * n_models, 3.5 * n_metrics),
                             squeeze=False)

    for col_i, model in enumerate(present_models):
        for row_i, metric in enumerate(metrics):
            ax = axes[row_i][col_i]
            for ratio in ratios:
                df = data.get((ratio, model))
                if df is None or metric not in df.columns:
                    continue
                ax.plot(df["round"], df[metric],
                        label=f"{ratio:.2f}",
                        color=colors[ratio],
                        linewidth=1.2)
            ax.set_title(model, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(metric, fontsize=8)
            if row_i == n_metrics - 1:
                ax.set_xlabel("Round", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=6, ncol=1, title="init_ratio")

    fig.suptitle(f"{dataset} — init_ratio effect per algorithm", fontsize=11)
    fig.tight_layout()
    out = base_dir / "init_ratio_per_algo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_summary_bar(data, base_dir: Path, metrics: list[str], dataset: str, ratios: list[float]):
    present_models = [m for m in MODELS if any((r, m) in data for r in ratios)]
    if not present_models:
        return

    colors = make_colors(ratios)
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4), squeeze=False)

    x = np.arange(len(present_models))
    width = 0.8 / len(ratios)

    for ax, metric in zip(axes[0], metrics):
        for i, ratio in enumerate(ratios):
            vals = []
            for model in present_models:
                df = data.get((ratio, model))
                if df is not None and metric in df.columns:
                    vals.append(float(df[metric].iloc[-1]))
                else:
                    vals.append(float("nan"))
            offsets = x + (i - len(ratios) / 2 + 0.5) * width
            ax.bar(offsets, vals, width=width * 0.9,
                   color=colors[ratio], label=f"{ratio:.2f}", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(present_models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f"{dataset} — final {metric}", fontsize=10)
        ax.legend(fontsize=7, title="init_ratio")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = base_dir / "init_ratio_summary_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="college_msg | bitcoin_alpha | dnc_email")
    parser.add_argument("--metric", nargs="+",
                        default=["coverage", "mrr@10"],
                        help="要画的指标列名，默认 coverage mrr@10")
    parser.add_argument("--ratios", nargs="+", type=float, default=INIT_RATIOS)
    parser.add_argument("--base_results", type=str, default="results/online")
    args = parser.parse_args()

    base_dir = Path(args.base_results) / f"init_ratio_sweep_{args.dataset}"
    if not base_dir.exists():
        print(f"结果目录不存在: {base_dir}")
        return

    data = load_data(base_dir, args.ratios)
    if not data:
        print("未找到任何 rounds.csv，请先运行实验。")
        return

    print(f"加载到 {len(data)} 条实验结果")
    plot_per_algorithm(data, base_dir, args.metric, args.dataset, args.ratios)
    plot_summary_bar(data, base_dir, args.metric, args.dataset, args.ratios)


if __name__ == "__main__":
    main()
