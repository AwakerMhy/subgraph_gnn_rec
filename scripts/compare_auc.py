import pandas as pd

df = pd.read_csv("results/orchestrator/ir40_constlr5e5/results.csv")
df = df[(df["method"] != "seal") & (df["dataset"] != "email_eu")]

gnn_methods = ["gnn", "gnn_h32", "gnn_concat", "gnn_sum"]
new_methods = ["graphsage_emb", "gat_emb"]
focus = gnn_methods + new_methods

df = df[df["method"].isin(focus)]
cols = ["dataset", "method", "final_auc_feedback", "final_uauc_feedback", "final_mrr@5"]
df = df[cols].sort_values(["dataset", "final_uauc_feedback"], ascending=[True, False])

pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 160)
print(df.to_string(index=False, float_format=lambda x: "%.4f" % x))

# 每个dataset各方法uauc排名
print("\n=== 每数据集 uauc 排名 ===")
for ds, g in df.groupby("dataset"):
    g = g.sort_values("final_uauc_feedback", ascending=False).reset_index(drop=True)
    print(f"\n{ds}")
    print(g[["method","final_uauc_feedback","final_auc_feedback","final_mrr@5"]].to_string(
        index=True, float_format=lambda x: "%.4f" % x))
