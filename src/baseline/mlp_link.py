"""src/baseline/mlp_link.py — MLP 链接预测 Baseline（无消息传递）。

使用 (u, v) 的拓扑特征拼接作为输入，不依赖图结构聚合，
用于验证 GNN 的结构学习相对于纯特征拼接的增量贡献。

特征（每节点 3 维）：out_degree, in_degree, clustering_coef（占位，复杂度低）
若提供 node_feat，则拼接在 3 维拓扑特征后。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.online.static_adj import StaticAdjacency


class MLPLinkScorer(nn.Module):
    """2 层 MLP 链接预测 scorer。

    forward(u_feat, v_feat) -> logit: (B,)
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, u_feat: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([u_feat, v_feat], dim=-1)
        return self.net(x).squeeze(-1)


def extract_topo_features(
    adj: StaticAdjacency,
    n_nodes: int,
    node_feat: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """计算每节点 3 维拓扑特征：[out_degree, in_degree, out+in]（归一化）。

    如果提供 node_feat，则拼接在后。
    Returns: (n_nodes, 3 + node_feat_dim)
    """
    out_deg = np.array([len(adj.out_neighbors(u)) for u in range(n_nodes)], dtype=np.float32)
    in_deg = np.array([len(adj.in_neighbors(u)) for u in range(n_nodes)], dtype=np.float32)
    total = out_deg + in_deg
    max_deg = max(total.max(), 1.0)
    topo = np.stack([out_deg / max_deg, in_deg / max_deg, total / max_deg], axis=1)
    feat = torch.tensor(topo, dtype=torch.float32)
    if device is not None:
        feat = feat.to(device)
    if node_feat is not None:
        feat = torch.cat([feat, node_feat], dim=-1)
    return feat
