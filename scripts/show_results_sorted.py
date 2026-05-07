import pandas as pd

df = pd.read_csv("results/orchestrator/ir40_constlr5e5/results.csv")
df = df[(df["method"] != "seal") & (df["dataset"] != "email_eu")]

cols = ["dataset", "method", "final_coverage", "final_mrr@1", "final_mrr@5",
        "final_hits@5", "final_auc_feedback", "final_uauc_feedback"]
cols = [c for c in cols if c in df.columns]
df = df[cols].sort_values(["dataset", "final_mrr@5"], ascending=[True, False])

pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 220)
print(df.to_string(index=False, float_format=lambda x: "%.4f" % x))
