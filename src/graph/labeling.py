"""src/graph/labeling.py — DRNL 节点标记

DRNL (Double-Radius Node Labeling)：
    ℓ(w) = 1 + min(d_u, d_v) + floor(d/2) * floor((d-1)/2)
    其中 d = d_u + d_v

特殊情况：
- w == u 或 w == v：ℓ = 1（固定，由位置区分）
- 某方向不可达：ℓ = 0（不可达标签）

标签嵌入：离散标签 → nn.Embedding → label_dim 维向量
"""
from __future__ import annotations

from collections import deque

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import dgl
    HAS_DGL = True
except ImportError:
    HAS_DGL = False


def _bfs_distances(
    start: int,
    adj: dict[int, list[int]],
    nodes: list[int],
) -> dict[int, int]:
    """在 nodes 集合（子图）内，从 start 出发的 BFS 最短路径距离。

    使用无向邻接（忽略边方向），因为 DRNL 原始定义基于无向距离。
    不可达节点不出现在返回字典中。
    """
    node_set = set(nodes)
    dist = {start: 0}
    queue = deque([start])

    while queue:
        cur = queue.popleft()
        for nb in adj.get(cur, []):
            if nb in node_set and nb not in dist:
                dist[nb] = dist[cur] + 1
                queue.append(nb)

    return dist


def drnl_label(
    node_list: list[int],
    u_global: int,
    v_global: int,
    adj_undirected: dict[int, list[int]],
) -> np.ndarray:
    """计算子图中每个节点的 DRNL 标签。

    Args:
        node_list:       子图节点的全局 ID 列表（顺序对应局部索引）
        u_global:        u 的全局 ID
        v_global:        v 的全局 ID
        adj_undirected:  无向邻接表（{node: [neighbors]}），基于截断图

    Returns:
        labels: np.ndarray，shape (N,)，int，N = len(node_list)
    """
    dist_u = _bfs_distances(u_global, adj_undirected, node_list)
    dist_v = _bfs_distances(v_global, adj_undirected, node_list)

    labels = np.zeros(len(node_list), dtype=np.int64)

    for i, w in enumerate(node_list):
        if w == u_global or w == v_global:
            labels[i] = 1
            continue

        du = dist_u.get(w, None)
        dv = dist_v.get(w, None)

        if du is None or dv is None:
            labels[i] = 0  # 不可达
            continue

        d = du + dv
        label = 1 + min(du, dv) + (d // 2) * ((d - 1) // 2)
        labels[i] = label

    return labels


def build_undirected_adj(edges: "pd.DataFrame") -> dict[int, list[int]]:
    """从有向边表构建无向邻接表（用于 DRNL 距离计算）。"""
    import pandas as pd
    adj: dict[int, list[int]] = {}
    for _, row in edges.iterrows():
        u, v = int(row["src"]), int(row["dst"])
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    return adj


class LabelEmbedding(nn.Module):
    """将离散 DRNL 标签嵌入为向量。

    标签范围 [0, max_label]：
        0 → 不可达标签
        1 → u/v 固定标签
        2+ → DRNL 计算值
    """

    def __init__(self, max_label: int = 50, label_dim: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(max_label + 1, label_dim, padding_idx=0)
        self.max_label = max_label

    def forward(self, labels: "torch.Tensor") -> "torch.Tensor":
        """
        Args:
            labels: (N,) long tensor，值在 [0, max_label]

        Returns:
            (N, label_dim) float tensor
        """
        labels = labels.clamp(0, self.max_label)
        return self.embedding(labels)
