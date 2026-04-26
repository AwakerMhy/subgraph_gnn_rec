"""Coverage + MRR@10 趋势对比图：有向稀疏图，GNN vs random vs MLP。"""
import pandas as pd
import matplotlib.pyplot as plt
import os

datasets = [
    {
        "name": "bitcoin_alpha\n(recip=0.83, deg=7.4)",
        "lines": [
            ("GNN cyclic", "results/online/bitcoin_alpha_cyclic_lr/rounds.csv",         "C0", "-"),
            ("MLP",        "results/online/bitcoin_alpha_mlp/rounds.csv",                "C2", "--"),
            ("random",     "results/online/bitcoin_alpha_weak_recall_random/rounds.csv", "C3", ":"),
        ],
    },
    {
        "name": "bitcoin_otc\n(recip=0.79, deg=7.4)",
        "lines": [
            ("GNN",    "results/online/bitcoin_otc_full/rounds.csv",     "C0", "-"),
            ("MLP",    "results/online/bitcoin_otc_mlp/rounds.csv",      "C2", "--"),
            ("random", "results/online/bitcoin_alpha_random/rounds.csv", "C3", ":"),
        ],
    },
    {
        "name": "epinions\n(recip=0.41, deg=8.4)",
        "lines": [
            ("GNN cyclic", "results/online/epinions_cyclic/rounds.csv", "C0", "-"),
            ("MLP",        "results/online/epinions_mlp/rounds.csv",    "C2", "--"),
            ("random",     "results/online/epinions_random/rounds.csv", "C3", ":"),
        ],
    },
    {
        "name": "sx_askubuntu\n(recip=0.33, deg=7.1)",
        "lines": [
            ("GNN cyclic", "results/online/sx_askubuntu_cyclic/rounds.csv", "C0", "-"),
            ("MLP",        "results/online/sx_askubuntu_mlp/rounds.csv",    "C2", "--"),
            ("random",     "results/online/sx_askubuntu_random/rounds.csv", "C3", ":"),
        ],
    },
    {
        "name": "dnc_email\n(recip=0.41, deg=35)",
        "lines": [
            ("GNN cyclic", "results/online/dnc_email_cyclic/rounds.csv",        "C0", "-"),
            ("random",     "results/online/dnc_email_random_cyclic/rounds.csv", "C3", ":"),
        ],
    },
    {
        "name": "email_eu\n(recip=0.71, deg=30)",
        "lines": [
            ("GNN", "results/online/email_eu_full/rounds.csv", "C0", "-"),
        ],
    },
]

fig, axes = plt.subplots(2, 6, figsize=(22, 7))

for col, ds in enumerate(datasets):
    for row, (metric, ylabel, smooth) in enumerate([
        ("coverage", "Coverage", False),
        ("mrr@10",   "MRR@10",   True),
    ]):
        ax = axes[row, col]
        for label, path, color, ls in ds["lines"]:
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            if metric not in df.columns:
                continue
            s = df[metric].dropna()
            x = s.index
            y = s.values
            if smooth and len(y) > 5:
                y = pd.Series(y).rolling(5, center=True, min_periods=1).mean().values
            ax.plot(x, y, color=color, linestyle=ls, linewidth=1.6,
                    label=label, alpha=0.85)

        if row == 0:
            ax.set_title(ds["name"], fontsize=8.5, fontweight="bold")
        ax.set_xlabel("Round" if row == 1 else "", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, None)
        ax.legend(fontsize=7, loc="lower right" if metric == "coverage" else "upper right")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=7)

plt.suptitle("Coverage (top) & MRR@10 (bottom, smoothed): GNN vs MLP vs Random",
             fontsize=11, y=1.01)
plt.tight_layout()

for out in ["results/online/coverage_trend.png", "results/online/mrr_trend.png"]:
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")
