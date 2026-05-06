import pandas as pd, pathlib

datasets = [
    'college_msg', 'email_eu', 'bitcoin_alpha', 'dnc_email',
    'wiki_vote', 'slashdot', 'sx_mathoverflow',
    'sx_askubuntu', 'sx_superuser', 'advogato',
]
methods = ['random','ground_truth','cn','aa','jaccard','pa','mlp','node_emb','gnn','gnn_concat','gnn_sum']

for ds in datasets:
    rows = []
    base = pathlib.Path(f'results/online/algo_sweep_{ds}_v2')
    if not base.exists():
        continue
    for m in methods:
        p = base / f'{ds}_{m}' / 'rounds.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p)
        rows.append({'method': m,
            'mrr@1':  round(df['mrr@1'].mean(), 4),
            'mrr@5':  round(df['mrr@5'].mean(), 4),
            'coverage': round(df['coverage'].mean(), 4),
        })
    if rows:
        print(f'\n=== {ds} ===')
        print(pd.DataFrame(rows).to_string(index=False))
