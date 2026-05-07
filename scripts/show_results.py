import pandas as pd
import sys

csv = sys.argv[1] if len(sys.argv) > 1 else "results/orchestrator/ir40_constlr5e5/results.csv"
exclude = sys.argv[2].split(",") if len(sys.argv) > 2 else []

df = pd.read_csv(csv)
if exclude:
    df = df[~df["method"].isin(exclude)]

cols = ["dataset", "method", "final_coverage", "final_mrr@1", "final_mrr@5",
        "final_hits@5", "final_auc_feedback", "final_uauc_feedback"]
cols = [c for c in cols if c in df.columns]
df = df[cols].sort_values(["dataset", "method"])

fmt = {c: "{:.4f}".format for c in cols if c not in ("dataset", "method")}
pd.set_option("display.max_rows", 300)
pd.set_option("display.width", 200)
print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
