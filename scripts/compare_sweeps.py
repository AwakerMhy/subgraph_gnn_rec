"""Compare GNN vs heuristic advantage between two sweeps."""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
ORCH_DIR = ROOT / "results" / "orchestrator"

METHOD_ORDER = ["random", "cn", "aa", "pa", "jaccard", "gnn", "gnn_concat", "gnn_sum",
                "gnn_h32", "gnn_concat_h8", "gnn_sum_h8", "gat_emb", "graphsage_emb", "seal"]
GNN_METHODS  = ["gnn", "gnn_concat", "gnn_sum", "gnn_h32", "gnn_concat_h8", "gnn_sum_h8",
                "gat_emb", "graphsage_emb", "seal"]
HEURISTICS   = ["cn", "aa", "pa", "jaccard"]

METRICS = [
    ("UAUC",  "final_uauc_feedback"),
    ("MRR@5", "final_mrr@5"),
    ("hits@5","final_hits@5"),
]

def load(sweep_name):
    p = ORCH_DIR / sweep_name / "results.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

def avg_rank_by_dataset(df, col, datasets=None):
    """For each dataset, rank methods; return avg rank per method."""
    if datasets:
        df = df[df["dataset"].isin(datasets)]
    mean = df.groupby(["dataset", "method"])[col].mean()
    methods_present = [m for m in METHOD_ORDER if m in df["method"].unique()]
    all_ranks = {m: [] for m in methods_present}
    for ds in df["dataset"].unique():
        row = {}
        for m in methods_present:
            try:
                row[m] = mean.loc[(ds, m)]
            except KeyError:
                pass
        s = pd.Series(row).dropna()
        r = s.rank(ascending=False)
        for m, v in r.items():
            all_ranks[m].append(v)
    return {m: np.mean(v) for m, v in all_ranks.items() if v}

def best_gnn_rank(ranks):
    return min((r for m, r in ranks.items() if m in GNN_METHODS), default=None)

def best_heuristic_rank(ranks):
    return min((r for m, r in ranks.items() if m in HEURISTICS), default=None)

def main():
    sweep_a = sys.argv[1] if len(sys.argv) > 1 else "ir50_constlr5e5_multiseed_bidir"
    sweep_b = sys.argv[2] if len(sys.argv) > 2 else "ir50_constlr5e5_ppos95_pneg01"

    df_a = load(sweep_a)
    df_b = load(sweep_b)

    if df_a is None or df_b is None:
        print("Missing results.csv for one of the sweeps.")
        return

    # Only compare datasets that are complete in BOTH sweeps
    # "complete" = all 5 seeds × all methods
    def complete_datasets(df):
        counts = df.groupby("dataset")["method"].nunique()
        all_methods = df["method"].nunique()
        # dataset is complete if it has all methods with >=5 seeds each
        seed_counts = df.groupby(["dataset","method"])["seed"].nunique()
        ds_complete = []
        for ds in df["dataset"].unique():
            try:
                mc = counts[ds]
                sc = seed_counts[ds].min()
                if sc >= 4:  # at least 4 seeds
                    ds_complete.append(ds)
            except:
                pass
        return set(ds_complete)

    complete_a = complete_datasets(df_a)
    complete_b = complete_datasets(df_b)
    common_ds  = sorted(complete_a & complete_b)

    print(f"Comparing (on {len(common_ds)} datasets with >=4 seeds in both sweeps):")
    print(f"  A: {sweep_a}")
    print(f"  B: {sweep_b}")
    print(f"  Common datasets: {', '.join(common_ds)}")
    print()

    for label, col in METRICS:
        sub_a = df_a[df_a["dataset"].isin(common_ds)]
        sub_b = df_b[df_b["dataset"].isin(common_ds)]

        ranks_a = avg_rank_by_dataset(sub_a, col, common_ds)
        ranks_b = avg_rank_by_dataset(sub_b, col, common_ds)

        best_gnn_a = best_gnn_rank(ranks_a)
        best_gnn_b = best_gnn_rank(ranks_b)
        best_h_a   = best_heuristic_rank(ranks_a)
        best_h_b   = best_heuristic_rank(ranks_b)

        best_gnn_name_a = min((m for m in GNN_METHODS if m in ranks_a), key=lambda m: ranks_a[m])
        best_gnn_name_b = min((m for m in GNN_METHODS if m in ranks_b), key=lambda m: ranks_b[m])
        best_h_name_a   = min((m for m in HEURISTICS if m in ranks_a), key=lambda m: ranks_a[m])
        best_h_name_b   = min((m for m in HEURISTICS if m in ranks_b), key=lambda m: ranks_b[m])

        gap_a = best_h_a - best_gnn_a if best_h_a and best_gnn_a else None
        gap_b = best_h_b - best_gnn_b if best_h_b and best_gnn_b else None

        print(f"[ {label} ]")
        print(f"  {'Method':20s}  {'A (ppos=1.0)':>14s}  {'B (ppos=0.95)':>14s}  {'delta':>8s}")
        print(f"  {'-'*62}")
        all_methods = sorted(set(ranks_a) | set(ranks_b), key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        for m in all_methods:
            ra = ranks_a.get(m, float('nan'))
            rb = ranks_b.get(m, float('nan'))
            delta = rb - ra if not (np.isnan(ra) or np.isnan(rb)) else float('nan')
            tag = ""
            if m in GNN_METHODS: tag = "*"
            print(f"  {m+tag:20s}  {ra:14.2f}  {rb:14.2f}  {delta:+8.2f}")
        print()
        print(f"  Best GNN:       A={best_gnn_name_a}({best_gnn_a:.2f})  B={best_gnn_name_b}({best_gnn_b:.2f})")
        print(f"  Best heuristic: A={best_h_name_a}({best_h_a:.2f})  B={best_h_name_b}({best_h_b:.2f})")
        if gap_a and gap_b:
            direction = "WIDER" if gap_b > gap_a else "NARROWER"
            print(f"  GNN lead (heuristic_rank - gnn_rank):  A={gap_a:.2f}  B={gap_b:.2f}  => {direction} by {abs(gap_b-gap_a):.2f}")
        print()

if __name__ == "__main__":
    main()
