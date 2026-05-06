"""src/baseline/graphsage_emb.py — GraphSAGE with learnable node embeddings.

与 GraphSAGEModel 的区别：节点特征由 nn.Embedding(n_nodes, emb_dim) 查表，
通过 g.ndata["_node_id"] 获取全局节点 ID，而非 2-dim one-hot 角色特征。
"""
from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from src.model.scorer import Scorer


class GraphSAGEEmbModel(nn.Module):
    """GraphSAGE 链接预测模型（可学习节点嵌入作输入特征）。

    Args:
        n_nodes:           图中节点总数（用于构建 Embedding 表）
        emb_dim:           节点嵌入维度
        hidden_dim:        SAGEConv 隐层维度
        num_layers:        SAGEConv 层数
        scorer_hidden_dim: Scorer MLP 隐层维度（None 则等于 hidden_dim）
        aggregator_type:   SAGEConv 聚合方式（"mean" / "pool" / "lstm" / "gcn"）
    """

    def __init__(
        self,
        n_nodes: int,
        emb_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        scorer_hidden_dim: int | None = None,
        aggregator_type: str = "mean",
    ) -> None:
        super().__init__()
        self.node_emb = nn.Embedding(n_nodes, emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        self.layers = nn.ModuleList()
        dims = [emb_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(
                dglnn.SAGEConv(dims[i], hidden_dim, aggregator_type=aggregator_type)
            )

        _scorer_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else hidden_dim
        self.scorer = Scorer(in_dim=hidden_dim, hidden_dim=_scorer_hidden)

    def _encode(self, g: dgl.DGLGraph, node_ids: torch.Tensor) -> torch.Tensor:
        h = self.node_emb(node_ids)
        for conv in self.layers:
            h = conv(g, h)
            h = torch.relu(h)
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
