import pandas as pd
from pathlib import Path

sweep = 'new_model_sweep_ir40_s42'
models = ['gnn_sum', 'graphsage_emb', 'gat_emb']
datasets = ['college_msg', 'dnc_email', 'bitcoin_alpha', 'email_eu']

rows = []
for ds in datasets:
    for m in models:
        p = Path(f'results/online/{sweep}/{ds}_{m}/rounds.csv')
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rows.append({
            'dataset': ds, 'model': m,
            'mrr@1': round(df['mrr@1'].mean(), 4),
            'mrr@5': round(df['mrr@5'].mean(), 4),
            'hits@5': round(df['hits@5'].mean(), 4),
            'coverage': round(df['coverage'].mean(), 4),
        })

out = pd.DataFrame(rows)
print(out.to_string(index=False))
