"""scripts/plot_init_strategy_sweep.py — 对比 init_strategy 对各算法 coverage/MRR 的影响。

生成两类图：
  1. per-algorithm: 同一算法，不同 init_strategy 的曲线 (coverage vs round)
  2. summary bar: 最终轮 coverage / mrr@10，按 strategy 分组，各算法一条柱

用法：
    python scripts/plot_init_strategy_sweep.py --dataset college_msg
    python scripts/plot_init_strategy_sweep.py --dataset college_msg --metric coverage mrr@10
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INIT_STRATEGIES = ["random", "stratified", "snowball", "forest_fire", "all_covered"]
MODELS = ["ground_truth", "random", "cn", "aa", "jaccard", "pa", "gnn"]

STRATEGY_COLORS = {
    "random":       "#e41a1c",
    "stratified":   "#377eb8",
    "snowball":     "#4daf4a",
    "forest_fire":  "#ff7f00",
    "all_covered":  "#984ea3",
}
STRATEGY_LS = {
    "random":       "-",
    "stratified":   "--",
    "snowball":     "-.",
    "forest_fire":  ":",
    "all_covered":  (0, (3, 1, 1, 1)),
}


def load_data(base_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    """返回 {(strategy, model): df}"""
    data = {}
    for strategy in INIT_STRATEGIES:
        for model in MODELS:
            csv = base_dir / f"{strategy}_{model}" / "rounds.csv"
            if csv.exists():
                data[(strategy, model)] = pd.read_csv(csv)
    return data


def plot_per_algorithm(data, base_dir: Path, metrics: list[str], dataset: str):
    """每个算法一张子图，横轴 round，各 strategy 一条线。"""
    present_models = sorted({m for _, m in data.keys()}, key=lambda m: MODELS.index(m) if m in MODELS else 99)
    n_metrics = len(metrics)
    n_models = len(present_models)
    if n_models == 0:
        return

    fig, axes = plt.subplots(n_metrics, n_models,
                             figsize=(4 * n_models, 3.5 * n_metrics),
                             squeeze=False)

    for col_i, model in enumerate(present_models):
        for row_i, metric in enumerate(metrics):
            ax = axes[row_i][col_i]
            for strategy in INIT_STRATEGIES:
                df = data.get((strategy, model))
                if df is None or metric not in df.columns:
                    continue
                ax.plot(df["round"], df[metric],
                        label=strategy,
                        color=STRATEGY_COLORS[strategy],
                        linestyle=STRATEGY_LS[strategy],
                        linewidth=1.2)
            ax.set_title(f"{model}", fontsize=9)
            if col_i == 0:
                ax.set_ylabel(metric, fontsize=8)
            if row_i == n_metrics - 1:
                ax.set_xlabel("Round", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=6, ncol=1)

    fig.suptitle(f"{dataset} — init_strategy effect per algorithm", fontsize=11)
    fig.tight_layout()
    out = base_dir / "init_strategy_per_algo.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_summary_bar(data, base_dir: Path, metrics: list[str], dataset: str):
    """最终轮指标的 bar chart：x 轴算法，每组算法内按 strategy 分柱。"""
    present_models = [m for m in MODELS if any((s, m) in data for s in INIT_STRATEGIES)]
    if not present_models:
        return

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 4), squeeze=False)

    x = range(len(present_models))
    n_strategies = len(INIT_STRATEGIES)
    width = 0.8 / n_strategies

    for ax, metric in zip(axes[0], metrics):
        for i, strategy in enumerate(INIT_STRATEGIES):
            vals = []
            for model in present_models:
                df = data.get((strategy, model))
                if df is not None and metric in df.columns:
                    vals.append(float(df[metric].iloc[-1]))
                else:
                    vals.append(float("nan"))
            offsets = [xi + (i - n_strategies / 2 + 0.5) * width for xi in x]
            ax.bar(offsets, vals, width=width * 0.9,
                   color=STRATEGY_COLORS[strategy], label=strategy, alpha=0.85)

        ax.set_xticks(list(x))
        ax.set_xticklabels(present_models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f"{dataset} — final {metric}", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = base_dir / "init_strategy_summary_bar.png"
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
    parser.add_argument("--base_results", type=str,
                        default="results/online",
                        help="结果根目录，默认 results/online")
    args = parser.parse_args()

    base_dir = Path(args.base_results) / f"init_strategy_sweep_{args.dataset}"
    if not base_dir.exists():
        print(f"结果目录不存在: {base_dir}")
        return

    data = load_data(base_dir)
    if not data:
        print(f"未找到任何 rounds.csv，请先运行实验。")
        return

    print(f"加载到 {len(data)} 条实验结果")
    plot_per_algorithm(data, base_dir, args.metric, args.dataset)
    plot_summary_bar(data, base_dir, args.metric, args.dataset)


if __name__ == "__main__":
    main()
