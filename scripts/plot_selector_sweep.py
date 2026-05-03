"""scripts/plot_selector_sweep.py — 对比 composite vs uniform user_selector 对各算法的影响。

用法：
    python scripts/plot_selector_sweep.py --dataset college_msg
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SELECTORS = ["composite", "uniform"]
MODELS = ["ground_truth", "random", "cn", "aa", "jaccard", "pa", "gnn"]

COLORS = {"composite": "#e41a1c", "uniform": "#377eb8"}
LINESTYLES = {"composite": "-", "uniform": "--"}


def load_data(base_dir: Path) -> dict[tuple[str, str], pd.DataFrame]:
    data = {}
    for sel in SELECTORS:
        for model in MODELS:
            csv = base_dir / f"{sel}_{model}" / "rounds.csv"
            if csv.exists():
                data[(sel, model)] = pd.read_csv(csv)
    return data


def plot_curves(data, base_dir: Path, metrics: list[str], dataset: str):
    present_models = [m for m in MODELS if any((s, m) in data for s in SELECTORS)]
    if not present_models:
        return

    n_m = len(metrics)
    n_mod = len(present_models)
    fig, axes = plt.subplots(n_m, n_mod, figsize=(4 * n_mod, 3.5 * n_m), squeeze=False)

    for col_i, model in enumerate(present_models):
        for row_i, metric in enumerate(metrics):
            ax = axes[row_i][col_i]
            for sel in SELECTORS:
                df = data.get((sel, model))
                if df is None or metric not in df.columns:
                    continue
                ax.plot(df["round"], df[metric],
                        label=sel, color=COLORS[sel], linestyle=LINESTYLES[sel], linewidth=1.5)
            ax.set_title(model, fontsize=9)
            if col_i == 0:
                ax.set_ylabel(metric, fontsize=8)
            if row_i == n_m - 1:
                ax.set_xlabel("Round", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)
            if col_i == 0 and row_i == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f"{dataset} — composite vs uniform user selector", fontsize=11)
    fig.tight_layout()
    out = base_dir / "selector_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_bar(data, base_dir: Path, metrics: list[str], dataset: str):
    present_models = [m for m in MODELS if any((s, m) in data for s in SELECTORS)]
    if not present_models:
        return

    import numpy as np
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    x = np.arange(len(present_models))
    width = 0.35

    for ax, metric in zip(axes[0], metrics):
        for i, sel in enumerate(SELECTORS):
            vals = []
            for model in present_models:
                df = data.get((sel, model))
                vals.append(float(df[metric].iloc[-1]) if df is not None and metric in df.columns else float("nan"))
            ax.bar(x + (i - 0.5) * width, vals, width * 0.9,
                   color=COLORS[sel], label=sel, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(present_models, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric, fontsize=9)
        ax.set_title(f"{dataset} — final {metric}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = base_dir / "selector_summary_bar.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metric", nargs="+", default=["coverage", "mrr@10"])
    parser.add_argument("--base_results", type=str, default="results/online")
    args = parser.parse_args()

    base_dir = Path(args.base_results) / f"selector_sweep_{args.dataset}"
    if not base_dir.exists():
        print(f"结果目录不存在: {base_dir}")
        return

    data = load_data(base_dir)
    if not data:
        print("未找到任何 rounds.csv。")
        return

    print(f"加载到 {len(data)} 条实验结果")
    plot_curves(data, base_dir, args.metric, args.dataset)
    plot_bar(data, base_dir, args.metric, args.dataset)


if __name__ == "__main__":
    main()
