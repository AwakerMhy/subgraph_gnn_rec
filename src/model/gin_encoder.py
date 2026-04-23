"""src/model/gin_encoder.py — GIN 图编码器

提供三种编码器：
- GINEncoder:            最后一层节点表示做全图 mean pooling → (hidden_dim,)
- GINEncoderLayerConcat: 每层分别对 u、v、其他节点 mean pooling 后 concat（所有层）
                         → (num_layers * 3 * hidden_dim,)
- GINEncoderLayerSum:    每层分别对 u、v、其他节点 mean pooling 后 concat（得 3H），
                         再对所有层的结果相加 → (3 * hidden_dim,)
"""
from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn


class GINEncoder(nn.Module):
    """L 层 GIN，输出全图 mean pooling embedding。

    Args:
        in_dim:     节点初始特征维度（默认 2，对应 one-hot）
        hidden_dim: 隐层维度
        num_layers: GIN 层数
        dropout:    dropout 概率
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()

        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(dglnn.GINConv(mlp, aggregator_type="sum"))

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g:    DGLGraph（子图，有向或无向均可）
            feat: (N, in_dim) 节点初始特征

        Returns:
            h_graph: (hidden_dim,) 全图 mean pooling embedding
        """
        h = feat
        for conv in self.layers:
            h = conv(g, h)
            h = torch.relu(h)

        # graph-level readout
        g.ndata["_h"] = h
        h_graph = dgl.mean_nodes(g, "_h")  # (1, hidden_dim)
        g.ndata.pop("_h")                  # 清理，避免 batch 时 schema 不一致
        return h_graph.squeeze(0)           # (hidden_dim,)


class GINEncoderLayerConcat(nn.Module):
    """L 层 GIN，每层分别对请求节点 u、候选节点 v、其他节点做 mean pooling，
    将所有层的结果 concat 作为图表示。

    输出维度: num_layers * 3 * hidden_dim

    Args:
        in_dim:     节点初始特征维度（默认 2，对应 one-hot）
        hidden_dim: 隐层维度
        num_layers: GIN 层数
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(dglnn.GINConv(mlp, aggregator_type="sum"))

    @property
    def out_dim(self) -> int:
        return self.num_layers * 3 * self.hidden_dim

    def forward(
        self,
        g: dgl.DGLGraph,
        feat: torch.Tensor,
        u_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            g:      DGLGraph（子图）
            feat:   (N, in_dim) 节点初始特征
            u_mask: (N,) bool — True 对应请求节点 u 的局部位置
            v_mask: (N,) bool — True 对应候选节点 v 的局部位置

        Returns:
            h_graph: (num_layers * 3 * hidden_dim,)
                     每层: [h_u | h_v | mean_others]，按层顺序 concat
        """
        h = feat
        other_mask = ~(u_mask | v_mask)
        parts: list[torch.Tensor] = []

        for conv in self.layers:
            h = conv(g, h)
            h = torch.relu(h)

            h_u = h[u_mask].mean(0)      # (hidden_dim,)
            h_v = h[v_mask].mean(0)      # (hidden_dim,)
            if other_mask.any():
                h_other = h[other_mask].mean(0)
            else:
                h_other = torch.zeros(self.hidden_dim, device=h.device, dtype=h.dtype)

            parts.extend([h_u, h_v, h_other])

        return torch.cat(parts)          # (num_layers * 3 * hidden_dim,)


class GINEncoderLayerSum(nn.Module):
    """L 层 GIN，每层分别对请求节点 u、候选节点 v、其他节点做 mean pooling 后 concat，
    再将所有层的结果相加作为图表示。

    输出维度: 3 * hidden_dim

    Args:
        in_dim:     节点初始特征维度（默认 2，对应 one-hot）
        hidden_dim: 隐层维度
        num_layers: GIN 层数
    """

    def __init__(
        self,
        in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(dglnn.GINConv(mlp, aggregator_type="sum"))

    @property
    def out_dim(self) -> int:
        return 3 * self.hidden_dim

    def forward(
        self,
        g: dgl.DGLGraph,
        feat: torch.Tensor,
        u_mask: torch.Tensor,
        v_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            g:      DGLGraph（子图）
            feat:   (N, in_dim) 节点初始特征
            u_mask: (N,) bool — True 对应请求节点 u 的局部位置
            v_mask: (N,) bool — True 对应候选节点 v 的局部位置

        Returns:
            h_graph: (3 * hidden_dim,)
                     每层: [h_u | h_v | mean_others]（3H），各层相加
        """
        h = feat
        other_mask = ~(u_mask | v_mask)
        acc = torch.zeros(3 * self.hidden_dim, device=feat.device, dtype=feat.dtype)

        for conv in self.layers:
            h = conv(g, h)
            h = torch.relu(h)

            h_u = h[u_mask].mean(0)      # (hidden_dim,)
            h_v = h[v_mask].mean(0)      # (hidden_dim,)
            if other_mask.any():
                h_other = h[other_mask].mean(0)
            else:
                h_other = torch.zeros(self.hidden_dim, device=h.device, dtype=h.dtype)

            acc = acc + torch.cat([h_u, h_v, h_other])  # (3 * hidden_dim,)

        return acc
