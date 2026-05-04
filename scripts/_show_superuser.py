import pandas as pd, pathlib

methods = ['random','ground_truth','cn','aa','jaccard','pa','mlp','node_emb','gnn','gnn_concat','gnn_sum']
rows = []
for m in methods:
    p = pathlib.Path(f'results/online/algo_sweep_sx_superuser_v2/sx_superuser_{m}/rounds.csv')
    if not p.exists():
        rows.append({'method': m, 'coverage': 'MISSING', 'mrr@1': '', 'mrr@5': '', 'mrr@10': '', 'hits@5': '', 'hit_rate@1': ''})
        continue
    df = pd.read_csv(p)
    last = df.iloc[-1]
    nan = float('nan')
    rows.append({'method': m,
        'coverage':    f"{last.get('coverage',    nan):.2%}",
        'mrr@1':       f"{last.get('mrr@1',       nan):.4f}",
        'mrr@5':       f"{last.get('mrr@5',       nan):.4f}",
        'mrr@10':      f"{last.get('mrr@10',      nan):.4f}",
        'hits@5':      f"{last.get('hits@5',      nan):.4f}",
        'hit_rate@1':  f"{last.get('hit_rate@1',  nan):.4f}",
    })
print(pd.DataFrame(rows).to_string(index=False))
