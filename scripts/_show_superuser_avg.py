import pandas as pd, pathlib

methods = ['random','ground_truth','cn','aa','jaccard','pa','mlp','node_emb','gnn','gnn_concat','gnn_sum']
rows = []
for m in methods:
    p = pathlib.Path(f'results/online/algo_sweep_sx_superuser_v2/sx_superuser_{m}/rounds.csv')
    if not p.exists():
        rows.append({'method': m, 'coverage': 'MISSING'})
        continue
    df = pd.read_csv(p)
    nan = float('nan')
    rows.append({'method': m,
        'coverage':   f"{df['coverage'].mean():.2%}",
        'mrr@1':      f"{df['mrr@1'].mean():.4f}",
        'mrr@5':      f"{df['mrr@5'].mean():.4f}",
        'mrr@10':     f"{df['mrr@10'].mean():.4f}",
        'hits@5':     f"{df['hits@5'].mean():.4f}",
        'hit_rate@1': f"{df['hit_rate@1'].mean():.4f}",
    })
print(pd.DataFrame(rows).to_string(index=False))
