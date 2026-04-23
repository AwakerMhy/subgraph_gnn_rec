"""检查子图大小分布，诊断数据稀疏性。"""
import sys, json
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.graph.subgraph import extract_subgraph
from src.utils.split import temporal_split

edges = pd.read_csv("data/synthetic/sbm/edges.csv").sort_values("timestamp").reset_index(drop=True)
with open("data/synthetic/sbm/meta.json") as f:
    meta = json.load(f)

train_edges, val_edges, _ = temporal_split(edges)

sizes, edge_counts = [], []
for _, row in train_edges.iterrows():
    u, v, t = int(row["src"]), int(row["dst"]), float(row["timestamp"])
    g = extract_subgraph(u, v, t, edges, max_hop=2, max_neighbors_per_node=30, seed=42)
    sizes.append(g.num_nodes())
    edge_counts.append(g.num_edges())

sizes = np.array(sizes)
edge_counts = np.array(edge_counts)
print(f"训练样本数: {len(sizes)}")
print(f"子图节点数  mean={sizes.mean():.1f}  min={sizes.min()}  max={sizes.max()}  median={np.median(sizes):.1f}")
print(f"子图边数    mean={edge_counts.mean():.1f}  min={edge_counts.min()}  max={edge_counts.max()}  median={np.median(edge_counts):.1f}")
print(f"子图节点=2（孤立对）的比例: {(sizes==2).mean()*100:.1f}%")
