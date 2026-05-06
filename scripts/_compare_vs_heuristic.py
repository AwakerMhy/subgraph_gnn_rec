import pandas as pd
from pathlib import Path

datasets = ['college_msg', 'dnc_email', 'bitcoin_alpha', 'email_eu']

# 新模型来自 new_model_sweep，启发式来自 v2 sweep
new_sweep = 'new_model_sweep_ir40_s42'
v2_prefix = 'algo_sweep_{}_v2_ir40_s42'

heuristics = ['random', 'cn', 'aa', 'jaccard', 'pa']
new_models = ['seal', 'graphsage_emb', 'gat_emb']

def load(path):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return {
        'mrr@1': round(df['mrr@1'].mean(), 4),
        'mrr@5': round(df['mrr@5'].mean(), 4),
        'hits@5': round(df['hits@5'].mean(), 4),
        'coverage': round(df['coverage'].mean(), 4),
    }

rows = []
for ds in datasets:
    v2_dir = Path('results/online') / v2_prefix.format(ds)
    new_dir = Path('results/online') / new_sweep

    for m in heuristics:
        p = v2_dir / (ds + '_' + m) / 'rounds.csv'
        r = load(p)
        if r:
            rows.append({'dataset': ds, 'model': m, **r})

    for m in new_models:
        p = new_dir / (ds + '_' + m) / 'rounds.csv'
        r = load(p)
        if r:
            rows.append({'dataset': ds, 'model': m, **r})

out = pd.DataFrame(rows)
for ds in datasets:
    sub = out[out['dataset'] == ds].drop(columns='dataset')
    if sub.empty:
        continue
    print(f'\n=== {ds} ===')
    print(sub.to_string(index=False))
