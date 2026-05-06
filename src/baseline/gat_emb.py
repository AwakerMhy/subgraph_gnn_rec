"""src/baseline/gat_emb.py — GAT with learnable node embeddings.

节点特征由 nn.Embedding(n_nodes, emb_dim) 查表，通过 g.ndata["_node_id"]
获取全局节点 ID，经多头 GATConv 聚合后 mean pool → Scorer。

注意：hidden_dim 应能被 num_heads 整除（head_dim = hidden_dim // num_heads）。
"""
from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from src.model.scorer import Scorer


class GATEmbModel(nn.Module):
    """GAT 链接预测模型（可学习节点嵌入作输入特征）。

    Args:
        n_nodes:           图中节点总数（用于构建 Embedding 表）
        emb_dim:           节点嵌入维度
        hidden_dim:        每层输出维度（= num_heads × head_dim）
        num_layers:        GATConv 层数
        num_heads:         注意力头数；hidden_dim 须能被 num_heads 整除
        scorer_hidden_dim: Scorer MLP 隐层维度（None 则等于 hidden_dim）
    """

    def __init__(
        self,
        n_nodes: int,
        emb_dim: int = 32,
        hidden_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        scorer_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) 须能被 num_heads ({num_heads}) 整除"
        )
        self.node_emb = nn.Embedding(n_nodes, emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        head_dim = hidden_dim // num_heads
        self.layers = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(num_layers):
            self.layers.append(
                dglnn.GATConv(
                    in_dim, head_dim,
                    num_heads=num_heads,
                    feat_drop=0.0,
                    attn_drop=0.0,
                    activation=torch.relu,
                    allow_zero_in_degree=True,
                )
            )
            in_dim = head_dim * num_heads   # = hidden_dim

        _scorer_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else hidden_dim
        self.scorer = Scorer(in_dim=hidden_dim, hidden_dim=_scorer_hidden)

    def _encode(self, g: dgl.DGLGraph, node_ids: torch.Tensor) -> torch.Tensor:
        h = self.node_emb(node_ids)
        for conv in self.layers:
            h = conv(g, h)      # (N, num_heads, head_dim)
            h = h.flatten(1)    # (N, hidden_dim)
        return h

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        node_ids = g.ndata["_node_id"].to(g.device)
        h = self._encode(g, node_ids)
        g.ndata["_h"] = h
        h_graph = dgl.mean_nodes(g, "_h").squeeze(0)
        g.ndata.pop("_h")
        return self.scorer(h_graph)

    def forward_batch(self, bg: dgl.DGLGraph) -> torch.Tensor:
        node_ids = bg.ndata["_node_id"].to(bg.device)
        h = self._encode(bg, node_ids)
        bg.ndata["_h"] = h
        h_graphs = dgl.mean_nodes(bg, "_h")   # (B, hidden_dim)
        bg.ndata.pop("_h")
        return self.scorer(h_graphs)           # (B,)
