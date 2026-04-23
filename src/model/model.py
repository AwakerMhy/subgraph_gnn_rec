"""src/model/model.py — 顶层模型组装

流程：
  1. 对子图中每个节点赋 2 维 one-hot 特征：u→[1,0], v→[0,1], 其他→[0,0]
  2. GINEncoder 将子图编码为 graph-level embedding
  3. Scorer 输出预测分数 ∈ (0, 1)
"""
from __future__ import annotations

import dgl
import torch
import torch.nn as nn

from src.model.encoder_attr import AttrEncoder
from src.model.gin_encoder import GINEncoder, GINEncoderLayerConcat, GINEncoderLayerSum
from src.model.scorer import Scorer


class LinkPredModel(nn.Module):
    """链接预测模型：one-hot 节点特征 + GIN + MLP。

    Args:
        hidden_dim:        GIN 隐层维度
        num_layers:        GIN 层数
        scorer_hidden_dim: Scorer MLP 隐层维度
        encoder_type:      "last"         — 最后一层全图 mean pooling（默认）
                           "layer_concat" — 每层对 u/v/其他节点分别 mean pooling 后 concat → (L*3H,)
                           "layer_sum"    — 每层对 u/v/其他节点分别 mean pooling 后 concat，各层相加 → (3H,)
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 3,
        scorer_hidden_dim: int | None = None,
        encoder_type: str = "last",
        node_feat_dim: int = 0,
    ) -> None:
        """
        Args:
            scorer_hidden_dim: Scorer MLP 隐层维度。
                               默认 None，自动设为 hidden_dim，使 Scorer 容量与 Encoder 对等。
        """
        super().__init__()
        self.encoder_type = encoder_type

        if encoder_type == "last":
            self.encoder = GINEncoder(in_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
            scorer_in_dim = hidden_dim
        elif encoder_type == "layer_concat":
            self.encoder = GINEncoderLayerConcat(in_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
            scorer_in_dim = self.encoder.out_dim
        elif encoder_type == "layer_sum":
            self.encoder = GINEncoderLayerSum(in_dim=2, hidden_dim=hidden_dim, num_layers=num_layers)
            scorer_in_dim = self.encoder.out_dim
        else:
            raise ValueError(f"未知 encoder_type: {encoder_type!r}，可选 'last' / 'layer_concat' / 'layer_sum'")

        if node_feat_dim > 0:
            self.attr_encoder = AttrEncoder(feat_dim=node_feat_dim, out_dim=hidden_dim)
            scorer_in_dim += hidden_dim
        else:
            self.attr_encoder = None

        _scorer_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else hidden_dim
        self.scorer = Scorer(in_dim=scorer_in_dim, hidden_dim=_scorer_hidden)

    def _assign_node_features(self, g: dgl.DGLGraph) -> torch.Tensor:
        """为子图节点赋 2 维 one-hot 特征。

        依赖 extract_subgraph 写入的 g.ndata['_u_flag'] 和 g.ndata['_v_flag']。

        Returns:
            feat: (N, 2) float tensor
        """
        n = g.num_nodes()
        feat = torch.zeros(n, 2, dtype=torch.float32, device=g.device)
        u_mask = g.ndata["_u_flag"]  # (N,) bool
        v_mask = g.ndata["_v_flag"]  # (N,) bool
        feat[u_mask, 0] = 1.0
        feat[v_mask, 1] = 1.0
        return feat

    def _concat_attr(
        self, h_graph: torch.Tensor, g: dgl.DGLGraph
    ) -> torch.Tensor:
        """若有 attr_encoder，取 u/v 属性并 concat 到 h_graph。"""
        if self.attr_encoder is None:
            return h_graph
        u_feat = g.ndata["node_feat"][g.ndata["_u_flag"]]  # (1, feat_dim)
        v_feat = g.ndata["node_feat"][g.ndata["_v_flag"]]  # (1, feat_dim)
        h_attr = self.attr_encoder(u_feat, v_feat)          # (1, hidden_dim)
        return torch.cat([h_graph.unsqueeze(0), h_attr], dim=-1).squeeze(0)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        """
        Args:
            g: DGLGraph，需含 g.ndata['_u_flag'] 和 g.ndata['_v_flag']

        Returns:
            score: scalar tensor ∈ (0, 1)
        """
        feat = self._assign_node_features(g)
        u_mask = g.ndata["_u_flag"]
        v_mask = g.ndata["_v_flag"]
        if self.encoder_type == "last":
            h_graph = self.encoder(g, feat)
        else:
            h_graph = self.encoder(g, feat, u_mask, v_mask)
        h_graph = self._concat_attr(h_graph, g)
        return self.scorer(h_graph)

    def forward_batch(self, bg: dgl.DGLGraph) -> torch.Tensor:
        """批量前向传播，bg 为 dgl.batch 后的合并图。

        Returns:
            scores: (B,) tensor ∈ (0, 1)
        """
        if self.encoder_type == "last":
            n = bg.num_nodes()
            feat = torch.zeros(n, 2, dtype=torch.float32, device=bg.device)
            u_mask = bg.ndata["_u_flag"]
            v_mask = bg.ndata["_v_flag"]
            feat[u_mask, 0] = 1.0
            feat[v_mask, 1] = 1.0

            h = feat
            for conv in self.encoder.layers:
                h = conv(bg, h)
                h = torch.relu(h)

            bg.ndata["_h"] = h
            h_graphs = dgl.mean_nodes(bg, "_h")  # (B, hidden_dim)
            bg.ndata.pop("_h")

            if self.attr_encoder is not None:
                u_feat = bg.ndata["node_feat"][u_mask]   # (B, feat_dim)
                v_feat = bg.ndata["node_feat"][v_mask]   # (B, feat_dim)
                h_attr = self.attr_encoder(u_feat, v_feat)  # (B, hidden_dim)
                h_graphs = torch.cat([h_graphs, h_attr], dim=-1)

            return self.scorer(h_graphs)          # (B,)

        else:  # layer_concat：unbatch 后逐图编码
            graphs = dgl.unbatch(bg)
            embeddings: list[torch.Tensor] = []
            for g in graphs:
                feat = self._assign_node_features(g)
                u_mask = g.ndata["_u_flag"]
                v_mask = g.ndata["_v_flag"]
                h_graph = self.encoder(g, feat, u_mask, v_mask)
                h_graph = self._concat_attr(h_graph, g)
                embeddings.append(h_graph)
            h_graphs = torch.stack(embeddings)    # (B, out_dim)
            return self.scorer(h_graphs)          # (B,)
