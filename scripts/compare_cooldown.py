import pandas as pd
from pathlib import Path

decay_dir = Path("results/online/algo_sweep")
hard_dir  = Path("results/online/algo_sweep_init025_cd_hard")

DATASETS = ["bitcoin_alpha", "college_msg", "dnc_email", "email_eu"]
MODELS   = ["ground_truth", "random", "cn", "aa", "jaccard", "pa", "mlp", "node_emb", "gnn", "gnn_concat", "gnn_sum"]
METRICS  = ["coverage", "precision_k", "hits@10"]

rows = []
for ds in DATASETS:
    for m in MODELS:
        tag = f"{ds}_{m}"
        for mode, d in [("decay", decay_dir), ("hard", hard_dir)]:
            p = d / tag / "rounds.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            last = df.tail(50)
            row = {"dataset": ds, "model": m, "mode": mode}
            for col in METRICS:
                if col in last.columns:
                    row[col] = round(last[col].mean(), 4)
            rows.append(row)

df_all = pd.DataFrame(rows)
pivot = df_all.pivot_table(index=["dataset", "model"], columns="mode", values=METRICS)
pivot.columns = [f"{m}_{c}" for m, c in pivot.columns]
pivot = pivot.reset_index()

for met in METRICS:
    hc, dc = f"{met}_hard", f"{met}_decay"
    if hc in pivot.columns and dc in pivot.columns:
        pivot[f"D{met}"] = (pivot[hc] - pivot[dc]).round(4)

display_cols = ["dataset", "model"]
for met in METRICS:
    for mode in ["decay", "hard"]:
        c = f"{met}_{mode}"
        if c in pivot.columns:
            display_cols.append(c)
    if f"D{met}" in pivot.columns:
        display_cols.append(f"D{met}")
print(pivot[display_cols].to_string(index=False))
