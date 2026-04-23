"""src/baseline/graphsage.py — GraphSAGE baseline

与 LinkPredModel(encoder_type='last') 相同接口，只是将 GINConv 替换为 SAGEConv。
节点特征：2 维 one-hot（u→[1,0], v→[0,1], 其他→[0,0]）
Readout：最后一层全图 mean pooling
"""
from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from src.model.encoder_attr import AttrEncoder
from src.model.scorer import Scorer


class GraphSAGEEncoder(nn.Module):
    """L 层 GraphSAGE（mean 聚合），输出全图 mean pooling embedding。"""

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        aggregator_type: str = "mean",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            self.layers.append(
                dglnn.SAGEConv(dims[i], hidden_dim, aggregator_type=aggregator_type)
            )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        h = feat
        for conv in self.layers:
            h = conv(g, h)
            h = torch.relu(h)
        g.ndata["_h"] = h
        h_graph = dgl.mean_nodes(g, "_h")   # (1, hidden_dim)
        g.ndata.pop("_h")
        return h_graph.squeeze(0)            # (hidden_dim,)


class GraphSAGEModel(nn.Module):
    """GraphSAGE 链接预测模型。

    接口与 LinkPredModel 完全相同（forward / forward_batch），
    可直接替换进 train.py。

    Args:
        hidden_dim:        隐层维度
        num_layers:        SAGEConv 层数
        scorer_hidden_dim: Scorer MLP 隐层维度（None 则与 hidden_dim 相同）
        aggregator_type:   SAGEConv 聚合方式（"mean" / "pool" / "lstm" / "gcn"）
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        scorer_hidden_dim: int | None = None,
        aggregator_type: str = "mean",
        node_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_dim=2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            aggregator_type=aggregator_type,
        )
        if node_feat_dim > 0:
            self.attr_encoder = AttrEncoder(feat_dim=node_feat_dim, out_dim=hidden_dim)
            scorer_in_dim = hidden_dim * 2
        else:
            self.attr_encoder = None
            scorer_in_dim = hidden_dim
        _scorer_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else hidden_dim
        self.scorer = Scorer(in_dim=scorer_in_dim, hidden_dim=_scorer_hidden)

    def _make_feat(self, n: int, u_mask: torch.Tensor, v_mask: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
        feat = torch.zeros(n, 2, dtype=torch.float32, device=device)
        feat[u_mask, 0] = 1.0
        feat[v_mask, 1] = 1.0
        return feat

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        u_mask = g.ndata["_u_flag"]
        v_mask = g.ndata["_v_flag"]
        feat = self._make_feat(g.num_nodes(), u_mask, v_mask, g.device)
        h_graph = self.encoder(g, feat)
        if self.attr_encoder is not None:
            u_feat = g.ndata["node_feat"][u_mask]
            v_feat = g.ndata["node_feat"][v_mask]
            h_attr = self.attr_encoder(u_feat, v_feat).squeeze(0)
            h_graph = torch.cat([h_graph, h_attr], dim=-1)
        return self.scorer(h_graph)

    def forward_batch(self, bg: dgl.DGLGraph) -> torch.Tensor:
        n = bg.num_nodes()
        u_mask = bg.ndata["_u_flag"]
        v_mask = bg.ndata["_v_flag"]
        feat = self._make_feat(n, u_mask, v_mask, bg.device)

        h = feat
        for conv in self.encoder.layers:
            h = conv(bg, h)
            h = torch.relu(h)

        bg.ndata["_h"] = h
        h_graphs = dgl.mean_nodes(bg, "_h")   # (B, hidden_dim)
        bg.ndata.pop("_h")

        if self.attr_encoder is not None:
            u_feat = bg.ndata["node_feat"][u_mask]   # (B, feat_dim)
            v_feat = bg.ndata["node_feat"][v_mask]   # (B, feat_dim)
            h_attr = self.attr_encoder(u_feat, v_feat)
            h_graphs = torch.cat([h_graphs, h_attr], dim=-1)

        return self.scorer(h_graphs)           # (B,)
