import pandas as pd
from pathlib import Path

DATASETS = [
    "college_msg", "email_eu", "bitcoin_alpha", "dnc_email", "wiki_vote",
    "slashdot", "sx_mathoverflow", "sx_askubuntu", "sx_superuser", "advogato",
]

def avg(path):
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return df["mrr@1"].mean() if "mrr@1" in df.columns else None

print(f"{'dataset':<18} {'cosine':>8} {'1e-4':>8} {'5e-5':>8}  {'best'}")
print("-" * 60)
for ds in DATASETS:
    tag = f"{ds}_gnn_concat"
    cos = avg(f"results/online/algo_sweep_{ds}_v2_s0/{tag}/rounds.csv")
    c1  = avg(f"results/online/algo_sweep_{ds}_v2_constlr_s0/{tag}/rounds.csv")
    c5  = avg(f"results/online/algo_sweep_{ds}_v2_constlr5e5_s0/{tag}/rounds.csv")

    vals = {"cosine": cos, "1e-4": c1, "5e-5": c5}
    avail = {k: v for k, v in vals.items() if v is not None}
    best = max(avail, key=avail.get) if avail else "N/A"

    def fmt(v): return f"{v:.4f}" if v is not None else "  N/A "
    print(f"{ds:<18} {fmt(cos):>8} {fmt(c1):>8} {fmt(c5):>8}  {best}")
