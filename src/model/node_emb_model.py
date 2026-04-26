"""src/model/node_emb_model.py — 节点嵌入精排模型。

每个节点维护一个可学习的 embedding，链接评分 = MLP(concat(h_u, h_v))。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NodeEmbModel(nn.Module):
    """节点嵌入链接预测模型：embedding lookup + concat + MLP。

    Args:
        n_nodes:    节点数
        emb_dim:    每个节点的嵌入维度
        hidden_dim: MLP 隐层维度
    """

    def __init__(self, n_nodes: int, emb_dim: int = 64, hidden_dim: int = 64) -> None:
        super().__init__()
        self.emb = nn.Embedding(n_nodes, emb_dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, u_ids: torch.Tensor, v_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_ids: (B,) long tensor — 请求节点 ID
            v_ids: (B,) long tensor — 候选节点 ID

        Returns:
            scores: (B,) float tensor ∈ (0, 1)
        """
        h = torch.cat([self.emb(u_ids), self.emb(v_ids)], dim=-1)
        return self.mlp(h).squeeze(-1)
