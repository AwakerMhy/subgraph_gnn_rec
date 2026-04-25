import pandas as pd, numpy as np

df = pd.read_csv('data/processed/college_msg/edges.csv')
nodes = set(df.src) | set(df.dst)
print(f"college_msg: edges={len(df)}, nodes={len(nodes)}, avg_total_degree={len(df)/len(nodes)*2:.1f}")
deg = df.groupby('src').size()
print(f"out-degree: mean={deg.mean():.1f}, median={deg.median():.1f}, p95={int(deg.quantile(0.95))}, max={deg.max()}")

r = pd.read_csv('results/online/college_msg_full/rounds.csv', header=None)
print(f"candidates_per_round mean={r[3].mean():.0f}, accepted mean={r[2].mean():.0f}")
# col3 is ~1800 (pool size), col2 is accepted count
