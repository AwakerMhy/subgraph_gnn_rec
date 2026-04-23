import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd, numpy as np
from src.baseline.heuristic import score_cn, score_aa, score_jaccard, score_katz, batch_score
from src.utils.split import temporal_split
from src.utils.metrics import compute_auc, compute_ap
from src.graph.negative_sampling import sample_negatives

edges = pd.read_csv("data/processed/college_msg/edges.csv")
edges = edges.sort_values("timestamp").reset_index(drop=True)
train_edges, val_edges, test_edges = temporal_split(edges)
n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1

# 取前100个测试正样本，每个配1个负样本评估
test_sample = test_edges.head(100)
pairs_pos, pairs_neg = [], []
for _, row in test_sample.iterrows():
    u, v, t = int(row["src"]), int(row["dst"]), float(row["timestamp"])
    pairs_pos.append((u, v, t))
    neg_vs = sample_negatives(u, t, edges, n_nodes, strategy="random", k=1, seed=42)
    for nv in neg_vs:
        pairs_neg.append((u, nv, t))

for method in ["cn", "aa", "jaccard", "katz"]:
    s_pos = batch_score(pairs_pos, edges, method=method)
    s_neg = batch_score(pairs_neg[:len(pairs_pos)], edges, method=method)
    y_true = np.array([1]*len(s_pos) + [0]*len(s_neg))
    y_score = np.concatenate([s_pos, s_neg])
    try:
        auc = compute_auc(y_true, y_score)
        ap  = compute_ap(y_true, y_score)
        print(f"{method:8s}: AUC={auc:.4f}  AP={ap:.4f}")
    except Exception as e:
        print(f"{method:8s}: {e}")

print("heuristic smoke test PASSED")
