import pandas as pd
from pathlib import Path

DATASETS = [
    "college_msg", "email_eu", "bitcoin_alpha", "dnc_email", "wiki_vote",
    "slashdot", "sx_mathoverflow", "sx_askubuntu", "sx_superuser", "advogato",
]
ALGOS = ["gnn", "gnn_sum", "gnn_concat"]

def avg(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df["mrr@1"].mean() if "mrr@1" in df.columns else None

for algo in ALGOS:
    print(f"\n=== {algo} ===")
    print(f"{'dataset':<18} {'cosine':>8} {'1e-4':>8} {'5e-5':>8}  {'best':<8} {'delta(5e-5 vs best_const)'}")
    print("-" * 72)
    wins = {"cosine": 0, "1e-4": 0, "5e-5": 0}
    for ds in DATASETS:
        tag = f"{ds}_{algo}"
        cos = avg(f"results/online/algo_sweep_{ds}_v2_s0/{tag}/rounds.csv")
        c1  = avg(f"results/online/algo_sweep_{ds}_v2_constlr_s0/{tag}/rounds.csv")
        c5  = avg(f"results/online/algo_sweep_{ds}_v2_constlr5e5_s0/{tag}/rounds.csv")
        vals = {k: v for k, v in {"cosine": cos, "1e-4": c1, "5e-5": c5}.items() if v is not None}
        best = max(vals, key=vals.get) if vals else "N/A"
        if best in wins:
            wins[best] += 1
        # delta: 5e-5 vs 1e-4 (best constant)
        delta = f"{c5 - c1:+.4f}" if (c5 is not None and c1 is not None) else "N/A"
        def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
        print(f"{ds:<18} {fmt(cos):>8} {fmt(c1):>8} {fmt(c5):>8}  {best:<8} {delta}")
    print(f"  胜出: cosine={wins['cosine']} | 1e-4={wins['1e-4']} | 5e-5={wins['5e-5']}")
