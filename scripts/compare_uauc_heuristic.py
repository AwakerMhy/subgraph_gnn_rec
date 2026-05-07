import pandas as pd

df = pd.read_csv("results/orchestrator/ir40_constlr5e5/results.csv")
df = df[(df["method"] != "seal") & (df["dataset"] != "email_eu")]

heuristics = ["cn", "aa", "jaccard", "pa"]
gnn_methods = ["gnn", "gnn_h32", "gnn_concat", "gnn_sum"]

focus = heuristics + gnn_methods
df = df[df["method"].isin(focus)]

cols = ["dataset", "method", "final_uauc_feedback", "final_mrr@5"]
df = df[cols]

# 每个dataset：最优GNN uauc vs 最优启发式 uauc
print(f"{'dataset':<20} {'best_gnn':<16} {'gnn_uauc':>9} {'best_heur':<16} {'heur_uauc':>9} {'gnn_win':>8}")
print("-" * 85)
for ds, g in df.groupby("dataset"):
    gnn_g = g[g["method"].isin(gnn_methods)]
    heur_g = g[g["method"].isin(heuristics)]
    if gnn_g.empty or heur_g.empty:
        continue
    best_gnn = gnn_g.loc[gnn_g["final_uauc_feedback"].idxmax()]
    best_heur = heur_g.loc[heur_g["final_uauc_feedback"].idxmax()]
    win = best_gnn["final_uauc_feedback"] - best_heur["final_uauc_feedback"]
    marker = "YES" if win > 0 else "NO"
    print(f"{ds:<20} {best_gnn['method']:<16} {best_gnn['final_uauc_feedback']:>9.4f} "
          f"{best_heur['method']:<16} {best_heur['final_uauc_feedback']:>9.4f} "
          f"{win:>+7.4f} {marker}")
