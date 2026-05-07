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

import numpy as np
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from src.model.encoder_attr import AttrEncoder
from src.model.scorer import Scorer


# ── 内部 DRNL 工具（稠密矩阵 BFS，适用于小子图 n ≤ ~50）──────────────────────

def _compute_drnl(g: dgl.DGLGraph) -> torch.Tensor:
    """基于子图局部拓扑计算 DRNL 标签，返回 (N,) long tensor。

    用 numpy 稠密矩阵 BFS 替代 Python deque，对 ego_cn 小子图（n ≤ 32）
    快 10-50×：每轮 BFS 迭代 = 一次 uint8 矩阵-向量乘，≤ 3 次迭代即收敛。
    """
    src_t, dst_t = g.edges()
    n = g.num_nodes()

    src_np = src_t.cpu().numpy().astype(np.int32)
    dst_np = dst_t.cpu().numpy().astype(np.int32)

    # 无向对称邻接矩阵（uint8 节省内存，matmul 结果只需判断正负）
    A = np.zeros((n, n), dtype=np.uint8)
    if len(src_np):
        A[src_np, dst_np] = 1
        A[dst_np, src_np] = 1

    u_local = int(g.ndata["_u_flag"].nonzero(as_tuple=False)[0].item())
    v_local = int(g.ndata["_v_flag"].nonzero(as_tuple=False)[0].item())

    def _bfs(start: int) -> np.ndarray:
        dist = np.full(n, -1, dtype=np.int32)
        dist[start] = 0
        frontier = np.zeros(n, dtype=np.uint8)
        frontier[start] = 1
        d = 1
        while True:
            candidates = A @ frontier          # 邻居集合（值 > 0 表示可达）
            new_mask = (candidates > 0) & (dist < 0)
            if not new_mask.any():
                break
            dist[new_mask] = d
            frontier[:] = 0
            frontier[new_mask] = 1
            d += 1
        return dist

    du = _bfs(u_local)
    dv = _bfs(v_local)

    labels = np.zeros(n, dtype=np.int64)
    labels[u_local] = 1
    labels[v_local] = 1

    both_reached = (du >= 0) & (dv >= 0)
    both_reached[u_local] = False
    both_reached[v_local] = False
    if both_reached.any():
        d_sum = (du + dv)[both_reached]
        d_min = np.minimum(du, dv)[both_reached]
        labels[both_reached] = 1 + d_min + (d_sum // 2) * ((d_sum - 1) // 2)

    return torch.from_numpy(labels)


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
        if "_drnl" in bg.ndata:
            # 快速路径：DRNL 已由 trainer 预计算，整 batch 一次 GPU forward
            labels = bg.ndata["_drnl"].clamp(0, self.max_label)
            h = self.label_embedding(labels)
            for conv in self.layers:
                h = conv(bg, h)
                h = torch.relu(h)
            bg.ndata["_h"] = h
            h_graphs = dgl.mean_nodes(bg, "_h")   # (B, hidden_dim)
            bg.ndata.pop("_h")
            if self.attr_encoder is not None:
                u_feat = bg.ndata["node_feat"][bg.ndata["_u_flag"]]  # (B, feat_dim)
                v_feat = bg.ndata["node_feat"][bg.ndata["_v_flag"]]  # (B, feat_dim)
                h_attr = self.attr_encoder(u_feat, v_feat)            # (B, hidden_dim)
                h_graphs = torch.cat([h_graphs, h_attr], dim=-1)
            return self.scorer(h_graphs)           # (B,)

        # 降级路径：无预计算标签，逐图计算（兼容 score() 非批量路径）
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
        h_graphs = torch.stack(embeddings)
        return self.scorer(h_graphs)
