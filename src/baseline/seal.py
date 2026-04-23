"""src/baseline/seal.py — SEAL baseline (simplified)

核心思想（Zhang & Chen 2018）：
  用 DRNL（Double-Radius Node Labeling）将每个节点按到 u、v 的拓扑距离打标签，
  再在带标签子图上跑 GNN，使模型能区分节点的结构角色。

本实现与原文的主要差异：
  - Readout：mean pooling（原文用 SortPooling + DGCNN，实现复杂度较高）
  - DRNL 计算基于子图局部索引（不依赖全局 ID）

接口与 LinkPredModel 完全相同（forward / forward_batch），可直接替换进 train.py。
"""
from __future__ import annotations

from collections import deque

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from src.model.encoder_attr import AttrEncoder
from src.model.scorer import Scorer


# ── 内部 DRNL 工具（操作局部索引）────────────────────────────────────────────

def _bfs_local(start: int, adj: dict[int, list[int]], n: int) -> dict[int, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def _compute_drnl(g: dgl.DGLGraph) -> torch.Tensor:
    """基于子图局部拓扑计算 DRNL 标签，返回 (N,) long tensor。"""
    src, dst = g.edges()
    # 无向邻接表（局部索引）
    adj: dict[int, list[int]] = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        adj.setdefault(s, []).append(d)
        adj.setdefault(d, []).append(s)

    n = g.num_nodes()
    u_mask = g.ndata["_u_flag"]
    v_mask = g.ndata["_v_flag"]

    u_local = int(u_mask.nonzero(as_tuple=False)[0].item())
    v_local = int(v_mask.nonzero(as_tuple=False)[0].item())

    dist_u = _bfs_local(u_local, adj, n)
    dist_v = _bfs_local(v_local, adj, n)

    labels = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        if i == u_local or i == v_local:
            labels[i] = 1
            continue
        du = dist_u.get(i)
        dv = dist_v.get(i)
        if du is None or dv is None:
            labels[i] = 0
        else:
            d = du + dv
            labels[i] = 1 + min(du, dv) + (d // 2) * ((d - 1) // 2)
    return labels


# ── SEAL 模型 ─────────────────────────────────────────────────────────────────

class SEALModel(nn.Module):
    """SEAL baseline：DRNL 标签嵌入 + GIN + mean pooling → Scorer。

    Args:
        hidden_dim:    GIN 隐层维度
        num_layers:    GIN 层数
        label_dim:     DRNL 标签嵌入维度
        max_label:     DRNL 最大标签值（超出 clamp）
        scorer_hidden_dim: Scorer MLP 隐层（None 则等于 hidden_dim）
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        label_dim: int = 32,
        max_label: int = 50,
        scorer_hidden_dim: int | None = None,
        node_feat_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.label_embedding = nn.Embedding(max_label + 1, label_dim, padding_idx=0)
        self.max_label = max_label

        in_dim = label_dim
        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(dglnn.GINConv(mlp, aggregator_type="sum"))

        if node_feat_dim > 0:
            self.attr_encoder = AttrEncoder(feat_dim=node_feat_dim, out_dim=hidden_dim)
            scorer_in_dim = hidden_dim * 2
        else:
            self.attr_encoder = None
            scorer_in_dim = hidden_dim
        _scorer_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else hidden_dim
        self.scorer = Scorer(in_dim=scorer_in_dim, hidden_dim=_scorer_hidden)

    def _encode(self, g: dgl.DGLGraph) -> torch.Tensor:
        labels = _compute_drnl(g).to(g.device)
        labels = labels.clamp(0, self.max_label)
        feat = self.label_embedding(labels)   # (N, label_dim)
        h = feat
        for conv in self.layers:
            h = conv(g, h)
            h = torch.relu(h)
        g.ndata["_h"] = h
        h_graph = dgl.mean_nodes(g, "_h")     # (1, hidden_dim)
        g.ndata.pop("_h")
        return h_graph.squeeze(0)             # (hidden_dim,)

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        h_graph = self._encode(g)
        if self.attr_encoder is not None:
            u_feat = g.ndata["node_feat"][g.ndata["_u_flag"]]
            v_feat = g.ndata["node_feat"][g.ndata["_v_flag"]]
            h_attr = self.attr_encoder(u_feat, v_feat).squeeze(0)
            h_graph = torch.cat([h_graph, h_attr], dim=-1)
        return self.scorer(h_graph)

    def forward_batch(self, bg: dgl.DGLGraph) -> torch.Tensor:
        graphs = dgl.unbatch(bg)
        embeddings = []
        for g in graphs:
            h_graph = self._encode(g)
            if self.attr_encoder is not None:
                u_feat = g.ndata["node_feat"][g.ndata["_u_flag"]]
                v_feat = g.ndata["node_feat"][g.ndata["_v_flag"]]
                h_attr = self.attr_encoder(u_feat, v_feat).squeeze(0)
                h_graph = torch.cat([h_graph, h_attr], dim=-1)
            embeddings.append(h_graph)
        h_graphs = torch.stack(embeddings)    # (B, hidden_dim*2 or hidden_dim)
        return self.scorer(h_graphs)          # (B,)
