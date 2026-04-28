"""画 algo_sweep 各算法的 coverage / precision_k / mrr@10 随 round 变化曲线。

用法：
    python scripts/plot_algo_sweep.py --sweep_dir results/online/algo_sweep_wiki_vote \
                                      --out results/online/algo_sweep_wiki_vote/curves.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALGO_ORDER = [
    "ground_truth", "gnn_sum", "gnn_concat", "gnn", "random",
    "node_emb", "cn", "aa", "jaccard", "pa", "mlp",
]

COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#8dd3c7", "#bebada", "#fb8072",
]


def load_sweep(sweep_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for d in sorted(sweep_dir.iterdir()):
        csv = d / "rounds.csv"
        if not d.is_dir() or not csv.exists():
            continue
        # 从目录名提取算法名（去掉数据集前缀）
        parts = d.name.split("_")
        # 找最后匹配 ALGO_ORDER 的后缀
        algo = None
        for n in range(len(parts), 0, -1):
            candidate = "_".join(parts[-n:])
            if candidate in ALGO_ORDER:
                algo = candidate
                break
        if algo is None:
            algo = d.name
        df = pd.read_csv(csv)
        data[algo] = df
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep_dir", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    out_path = Path(args.out) if args.out else sweep_dir / "curves.png"

    data = load_sweep(sweep_dir)
    if not data:
        print(f"No data found in {sweep_dir}")
        return

    # 排序：ALGO_ORDER 优先，其余按名称
    ordered = sorted(data.keys(), key=lambda a: (ALGO_ORDER.index(a) if a in ALGO_ORDER else 99, a))

    metrics = [
        ("coverage",    "Coverage"),
        ("precision_k", "Precision@K"),
        ("mrr@10",      "MRR@10"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    dataset_name = sweep_dir.name

    for ax, (col, label) in zip(axes, metrics):
        for i, algo in enumerate(ordered):
            df = data[algo]
            if col not in df.columns:
                continue
            color = COLORS[i % len(COLORS)]
            ls = "--" if algo == "ground_truth" else "-"
            lw = 1.5 if algo == "ground_truth" else 1.2
            ax.plot(df["round"], df[col], label=algo, color=color, linestyle=ls, linewidth=lw)
        ax.set_xlabel("Round")
        ax.set_ylabel(label)
        ax.set_title(f"{dataset_name} — {label}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
