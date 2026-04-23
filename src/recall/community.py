"""src/recall/community.py — 社区内随机采样召回器。

用 networkx greedy_modularity_communities 检测无向社区，
每 recompute_every_n 轮重算一次，其余轮次复用缓存。
从 u 所在社区内随机均匀采样（排除已有出边），分数统一为 1.0。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from src.recall.base import RecallBase

if TYPE_CHECKING:
    from src.online.static_adj import StaticAdjacency


class CommunityRandomRecall(RecallBase):
    """社区内随机召回器。"""

    def __init__(
        self,
        adj: "StaticAdjacency",
        n_nodes: int,
        recompute_every_n: int = 20,
        seed: int = 42,
    ) -> None:
        self._adj = adj
        self._n = n_nodes
        self._recompute_every_n = recompute_every_n
        self._rng = np.random.default_rng(seed)
        # node -> sorted list of community members
        self._community: list[list[int]] = [[i] for i in range(n_nodes)]
        self._last_recompute = -1
        self._recompute_communities()

    def _recompute_communities(self) -> None:
        G = nx.Graph()
        G.add_nodes_from(range(self._n))
        for u, v in self._adj.iter_edges():
            G.add_edge(u, v)
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
        except Exception:
            communities = [{i} for i in range(self._n)]
        node_to_comm: list[list[int]] = [[] for _ in range(self._n)]
        for comm in communities:
            members = sorted(comm)
            for node in members:
                node_to_comm[node] = members
        self._community = node_to_comm

    def update_graph(self, round_idx: int) -> None:
        if round_idx - self._last_recompute >= self._recompute_every_n:
            self._recompute_communities()
            self._last_recompute = round_idx

    def candidates(
        self,
        u: int,
        cutoff_time: float,  # noqa: ARG002
        top_k: int,
    ) -> list[tuple[int, float]]:
        comm = self._community[u]
        exclude = set(self._adj.out_neighbors(u)) | {u}
        pool = [v for v in comm if v not in exclude]
        if not pool:
            return []
        k = min(top_k, len(pool))
        chosen = self._rng.choice(pool, size=k, replace=False)
        return [(int(v), 1.0) for v in chosen]
