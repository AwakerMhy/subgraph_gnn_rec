"""生成 SBM 合成数据集到 data/synthetic/sbm/，用于端到端测试。"""
import sys, json
from pathlib import Path
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.dataset.synthetic.sbm import SBMGenerator

out_dir = Path("data/synthetic/sbm")
out_dir.mkdir(parents=True, exist_ok=True)

gen = SBMGenerator(n_nodes=1000, n_communities=5, p_in=0.3, p_out=0.02,
                   T=200, edges_per_step=15, seed=42)
edges_df = gen.generate()
node_feats = gen.get_node_features()

# 归一化时间戳到 [0,1]
t_min, t_max = edges_df["timestamp"].min(), edges_df["timestamp"].max()
edges_df["timestamp_raw"] = edges_df["timestamp"]
edges_df["timestamp"] = (edges_df["timestamp"] - t_min) / (t_max - t_min + 1e-8)
edges_df = edges_df.sort_values("timestamp").reset_index(drop=True)
edges_df.to_csv(out_dir / "edges.csv", index=False)

# nodes.csv
feat_cols = [f"feat_{i}" for i in range(node_feats.shape[1])]
nodes_df = pd.DataFrame(node_feats, columns=feat_cols)
nodes_df.insert(0, "node_id", np.arange(len(nodes_df)))
nodes_df.to_csv(out_dir / "nodes.csv", index=False)

# meta.json
meta = {
    "n_nodes": 1000,
    "n_edges": len(edges_df),
    "has_native_node_feature": True,
    "feat_dim": node_feats.shape[1],
    "t_min": float(t_min),
    "t_max": float(t_max),
    "is_directed": True,
}
with open(out_dir / "meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"生成完成：{len(edges_df)} 条边，节点数={meta['n_nodes']}，保存到 {out_dir}")
print(edges_df.head())
