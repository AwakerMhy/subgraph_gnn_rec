import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

variants = {
    "gnn":            ("results/online/bitcoin_alpha_thr_gnn/rounds.csv",            "#2196F3"),
    "mlp":            ("results/online/bitcoin_alpha_thr_mlp/rounds.csv",            "#FF9800"),
    "node_emb":       ("results/online/bitcoin_alpha_thr_node_emb/rounds.csv",       "#9C27B0"),
    "node_emb+pool":  ("results/online/bitcoin_alpha_thr_node_emb_pool_neg/rounds.csv", "#E91E63"),
    "random":         ("results/online/bitcoin_alpha_thr_random/rounds.csv",         "#9E9E9E"),
}

dfs = {k: pd.read_csv(p) for k, (p, _) in variants.items() if Path(p).exists()}

def smooth(s, w=5):
    return s.rolling(w, min_periods=1, center=True).mean()

metrics = [
    ("coverage",    "Coverage (G_t / G*)"),
    ("precision_k", "Precision@K"),
    ("n_accepted",  "Accepted / round"),
    ("mrr@5",       "MRR@5"),
    ("novelty",     "Novelty"),
]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Bitcoin-Alpha: node_emb vs node_emb+pool_neg vs baselines", fontsize=12)

for ax, (col, title) in zip(axes, metrics):
    for name, (_, color) in variants.items():
        if name not in dfs or col not in dfs[name].columns:
            continue
        df = dfs[name]
        lw = 2.2 if "pool" in name else 1.6
        ls = "-" if "pool" in name or name == "gnn" else "--"
        ax.plot(df["round"], smooth(df[col]), label=name, color=color,
                linewidth=lw, linestyle=ls)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Round")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
out = "results/online/bitcoin_alpha_pool_neg_compare.png"
plt.savefig(out, dpi=150)
print(f"saved -> {out}")

# 数字汇总
print("\n=== 后20轮均值 ===")
cols = ["coverage", "precision_k", "n_accepted", "novelty"]
rows = []
for name, (_, _) in variants.items():
    if name not in dfs:
        continue
    tail = dfs[name].tail(20)
    row = {"method": name}
    for c in cols:
        if c in tail.columns:
            row[c] = round(tail[c].mean(), 4)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
