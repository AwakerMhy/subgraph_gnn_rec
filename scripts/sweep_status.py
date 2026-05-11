"""sweep-status: read sweep progress and metric tables."""
import sqlite3
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ORCH_DIR = ROOT / "results" / "orchestrator"

METHOD_ORDER = ["random", "cn", "aa", "pa", "jaccard", "gnn", "gnn_concat", "gnn_sum",
                "gnn_h32", "gnn_concat_h8", "gnn_sum_h8", "gat_emb", "graphsage_emb", "seal"]

METRICS = [
    ("UAUC",     "final_uauc_feedback"),
    ("AUC",      "final_auc_feedback"),
    ("coverage", "final_coverage"),
    ("MRR@5",    "final_mrr@5"),
    ("hits@5",   "final_hits@5"),
]

def find_latest_sweep():
    latest, latest_time = None, 0
    for db in ORCH_DIR.glob("*/experiments.db"):
        t = db.stat().st_mtime
        if t > latest_time:
            latest_time, latest = t, db.parent.name
    return latest

def fmt(mean, std, count):
    if pd.isna(mean):
        return "-"
    if count <= 1 or pd.isna(std):
        return f"{mean:.3f}"
    return f"{mean:.3f}+-{std:.3f}"

def pivot_table(df, col):
    g = df.groupby(["dataset", "method"])[col]
    mean = g.mean()
    std  = g.std()
    cnt  = g.count()
    datasets = sorted(df["dataset"].unique())
    methods  = [m for m in METHOD_ORDER if m in df["method"].unique()]
    rows = []
    for ds in datasets:
        row = {"dataset": ds}
        for m in methods:
            try:
                v_mean = mean.loc[(ds, m)]
                v_std  = std.loc[(ds, m)]
                v_cnt  = cnt.loc[(ds, m)]
                row[m] = fmt(v_mean, v_std, v_cnt)
            except KeyError:
                row[m] = "-"
        rows.append(row)
    pivot = pd.DataFrame(rows).set_index("dataset")
    return pivot, mean, methods

def avg_rank(mean_series, methods, datasets):
    ranks = {}
    for ds in datasets:
        row = {}
        for m in methods:
            try:
                row[m] = mean_series.loc[(ds, m)]
            except KeyError:
                row[m] = np.nan
        s = pd.Series(row).dropna()
        r = s.rank(ascending=False)
        for m, v in r.items():
            ranks.setdefault(m, []).append(v)
    avg = {m: np.mean(v) for m, v in ranks.items()}
    return sorted(avg.items(), key=lambda x: x[1])

def conclude(label, ranked, mean_series, methods, datasets):
    lines = [f"[ {label} conclusion ]"]
    best_m, best_r = ranked[0]
    lines.append(f"Best overall: {best_m} (avg rank {best_r:.2f}).")
    # check heuristics
    heuristics = [m for m in ["cn", "aa", "pa", "jaccard"] if m in methods]
    gnn_methods = [m for m in methods if m.startswith("gnn") or m in ("gat_emb", "graphsage_emb", "seal")]
    if heuristics and gnn_methods:
        best_h = min(heuristics, key=lambda m: next((r for n,r in ranked if n==m), 99))
        best_h_rank = next((r for n,r in ranked if n==best_h), None)
        best_g_rank = best_r
        if best_h_rank:
            lines.append(f"Best heuristic: {best_h} (avg rank {best_h_rank:.2f}); GNN lead: {best_h_rank - best_g_rank:.2f} rank positions.")
    # exceptions
    exceptions = []
    for ds in datasets:
        try:
            best_val = max(mean_series.loc[ds][m] for m in methods if (ds, m) in mean_series.index if not pd.isna(mean_series.loc[(ds, m)]))
            best_m_ds = max(methods, key=lambda m: mean_series.loc[(ds, m)] if (ds, m) in mean_series.index and not pd.isna(mean_series.loc[(ds, m)]) else -1)
            if best_m_ds in heuristics:
                exceptions.append(f"{ds} (best: {best_m_ds}={best_val:.3f})")
        except:
            pass
    if exceptions:
        lines.append(f"Exceptions where heuristic wins: {', '.join(exceptions)}.")
    # std comparison
    gnn_stds, h_stds = [], []
    for ds in datasets:
        for m in gnn_methods:
            try:
                v = mean_series.loc[(ds, m)]
                if not pd.isna(v):
                    gnn_stds.append(v)
            except: pass
        for m in heuristics:
            try:
                v = mean_series.loc[(ds, m)]
                if not pd.isna(v):
                    h_stds.append(v)
            except: pass
    return "\n".join(lines)

def main():
    sweep_name = sys.argv[1] if len(sys.argv) > 1 else find_latest_sweep()
    sweep_dir  = ORCH_DIR / sweep_name

    # --- progress ---
    conn = sqlite3.connect(sweep_dir / "experiments.db")
    status_df = pd.read_sql(
        "SELECT dataset, status, COUNT(*) as n FROM cells GROUP BY dataset, status", conn)
    conn.close()

    pivot_s = status_df.pivot_table(index="dataset", columns="status", values="n", fill_value=0)
    total_done  = int(pivot_s.get("completed", pd.Series(0)).sum())
    total_cells = int(status_df["n"].sum())

    print(f"=== sweep: {sweep_name}  {total_done}/{total_cells} ===")
    cols = ["completed", "pending", "running", "hold"]
    hdr  = f"  {'dataset':25s}  {'done':>5}  {'pend':>5}  {'run':>5}  {'hold':>5}  {'total':>6}"
    print(hdr)
    for ds in sorted(status_df["dataset"].unique()):
        row = pivot_s.loc[ds] if ds in pivot_s.index else {}
        done  = int(row.get("completed", 0))
        pend  = int(row.get("pending",   0))
        run   = int(row.get("running",   0))
        hold  = int(row.get("hold",      0))
        tot   = done + pend + run + hold
        tag   = "[OK]" if done == tot else ("[->]" if run > 0 else "[..]")
        print(f"  {tag} {ds:22s}  {done:5d}  {pend:5d}  {run:5d}  {hold:5d}  {tot:6d}")
    done_col  = int(pivot_s.get("completed", pd.Series(0)).sum())
    pend_col  = int(pivot_s.get("pending",   pd.Series(0)).sum())
    run_col   = int(pivot_s.get("running",   pd.Series(0)).sum())
    hold_col  = int(pivot_s.get("hold",      pd.Series(0)).sum())
    print(f"  {'TOTAL':26s}  {done_col:5d}  {pend_col:5d}  {run_col:5d}  {hold_col:5d}  {total_cells:6d}")
    print()

    # --- metrics ---
    csv_path = sweep_dir / "results.csv"
    if not csv_path.exists():
        print("results.csv not found — skipping metrics.")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print("results.csv is empty — skipping metrics.")
        return

    datasets = sorted(df["dataset"].unique())

    rank_results = {}
    mean_data    = {}

    for label, col in METRICS:
        if col not in df.columns:
            print(f"[ {label} ] column '{col}' not in results.csv, skipping.\n")
            continue
        sub = df[["dataset", "method", col]].dropna(subset=[col])
        if sub.empty:
            continue
        pivot, mean_s, methods = pivot_table(sub, col)
        print(f"[ {label}  mean+-std ]")
        print(pivot.to_string(na_rep="-"))
        print(f"  datasets: {', '.join(datasets)}")
        print()
        rank_results[label] = (avg_rank(mean_s, methods, datasets), mean_s, methods)
        mean_data[label]    = mean_s

    # --- ranks + conclusions for UAUC, coverage, MRR@5, hits@5 ---
    for label in ["UAUC", "coverage", "MRR@5", "hits@5"]:
        if label not in rank_results:
            continue
        ranked, mean_s, methods = rank_results[label]
        print(f"[ {label} avg rank ]")
        for i, (m, r) in enumerate(ranked[:8], 1):
            print(f"  {i:2d}. {m:20s}  {r:.2f}")
        print()
        print(conclude(label, ranked, mean_s, methods, datasets))
        print()

    # --- summary ---
    print("=== SUMMARY ===")
    for label in ["UAUC", "coverage", "MRR@5", "hits@5"]:
        if label in rank_results:
            best_m, best_r = rank_results[label][0][0]
            print(f"  {label:10s}: best = {best_m}  (avg rank {best_r:.2f})")
    remaining = total_cells - total_done
    if remaining > 0:
        print(f"\n  Remaining: {remaining} cells")
        inc = status_df[status_df["status"] != "completed"]
        for _, row in inc.iterrows():
            print(f"    {row['dataset']:25s} {row['status']} x{row['n']}")

if __name__ == "__main__":
    main()
