import sqlite3
import pandas as pd
from pathlib import Path

SWEEP_NAME = "ir60_constlr5e5_multiseed_bidir"
ORCH_DIR = Path("results/orchestrator")
sweep_dir = ORCH_DIR / SWEEP_NAME
db_path = sweep_dir / "experiments.db"
results_csv = sweep_dir / "results.csv"

if not db_path.exists():
    print(f"DB not found: {db_path}")
    exit()

conn = sqlite3.connect(str(db_path))
cells = pd.read_sql("SELECT dataset, method, status, seed FROM cells", conn)
conn.close()

total = len(cells)
done = (cells["status"] == "done").sum()
running = (cells["status"] == "running").sum()
pending = (cells["status"] == "pending").sum()
failed = (cells["status"] == "failed").sum()
hold = (cells["status"] == "hold").sum()

print(f"=== sweep: {SWEEP_NAME}  {done}/{total} ===")
print(f"  done={done}  running={running}  pending={pending}  failed={failed}  hold={hold}")
print()

# Per-dataset summary
by_ds = cells.groupby(["dataset", "status"]).size().unstack(fill_value=0)
for col in ["done", "running", "pending", "failed", "hold"]:
    if col not in by_ds.columns:
        by_ds[col] = 0
by_ds["total"] = by_ds.sum(axis=1)
by_ds = by_ds[["done", "running", "pending", "failed", "hold", "total"]]
by_ds = by_ds.sort_values("done", ascending=False)

print(f"{'dataset':<22} {'done':>5} {'run':>5} {'pend':>5} {'fail':>5} {'hold':>5} {'total':>6}")
print("-" * 60)
for ds, row in by_ds.iterrows():
    marker = "[OK]" if row["done"] == row["total"] else ("[->]" if row["running"] > 0 else "[..]")
    print(f"{marker} {ds:<18} {int(row['done']):>5} {int(row['running']):>5} {int(row['pending']):>5} {int(row['failed']):>5} {int(row['hold']):>5} {int(row['total']):>6}")
print("-" * 60)
totals = by_ds.sum()
print(f"{'TOTAL':<22} {int(totals['done']):>5} {int(totals['running']):>5} {int(totals['pending']):>5} {int(totals['failed']):>5} {int(totals['hold']):>5} {int(totals['total']):>6}")
print()

# Metrics from results.csv
if not results_csv.exists():
    print("results.csv not found — no metrics yet")
    exit()

df = pd.read_csv(str(results_csv))
if df.empty:
    print("results.csv is empty — no metrics yet")
    exit()

print(f"Completed results: {len(df)} rows")

METHOD_ORDER = ["random","cn","aa","pa","jaccard","gnn","gnn_concat","gnn_sum","gnn_h32","gnn_concat_h8","gnn_sum_h8","gat_emb","graphsage_emb","seal"]

METRICS = [
    ("UAUC",     "final_uauc_feedback"),
    ("AUC",      "final_auc_feedback"),
    ("coverage", "final_coverage"),
    ("MRR@5",    "final_mrr@5"),
    ("hits@5",   "final_hits@5"),
]

def make_pivot(df, col):
    if col not in df.columns:
        return None
    g = df.groupby(["dataset", "method"])[col]
    mean = g.mean().round(3)
    std  = g.std().round(3)
    count = g.count()
    result = {}
    for (ds, meth), m in mean.items():
        s = std.get((ds, meth), float("nan"))
        c = count.get((ds, meth), 0)
        if c > 1 and not pd.isna(s):
            result[(ds, meth)] = f"{m:.3f}+-{s:.3f}"
        else:
            result[(ds, meth)] = f"{m:.3f}"
    idx = pd.MultiIndex.from_tuples(result.keys(), names=["dataset","method"])
    return pd.Series(result, index=idx).unstack("method")

for label, col in METRICS:
    pivot = make_pivot(df, col)
    if pivot is None:
        print(f"\n[ {label} ] -- column not found")
        continue
    cols = [c for c in METHOD_ORDER if c in pivot.columns]
    pivot = pivot[cols]
    print(f"\n[ {label} mean+-std ]")
    print(pivot.to_string(na_rep="-"))

# Rank analysis for key metrics
print("\n\n=== Avg Rank Analysis ===")
for label, col in [("UAUC","final_uauc_feedback"),("coverage","final_coverage"),("MRR@5","final_mrr@5"),("hits@5","final_hits@5")]:
    if col not in df.columns:
        continue
    mean_piv = df.groupby(["dataset","method"])[col].mean().unstack("method")
    cols = [c for c in METHOD_ORDER if c in mean_piv.columns]
    mean_piv = mean_piv[cols]
    ranks = mean_piv.rank(axis=1, ascending=False)
    avg_rank = ranks.mean().sort_values()
    print(f"\n[ {label} avg rank (top 8) ]")
    for i, (meth, r) in enumerate(avg_rank.head(8).items(), 1):
        print(f"  {i}. {meth:<18} {r:.2f}")
