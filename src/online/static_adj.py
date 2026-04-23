"""src/online/static_adj.py — 动态静态邻接表，duck-type TimeAdjacency 接口。

忽略 cutoff 参数，返回当前图中所有邻居。用于在线场景下图结构动态变化时
无缝替换 TimeAdjacency，与 extract_subgraph / RecallBase 完全兼容。
"""
from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


class StaticAdjacency:
    """基于 dict[int, set[int]] 的可变邻接表。

    接口与 TimeAdjacency 一致（cutoff 参数保留但忽略），可作为 time_adj 参数
    直接传入 extract_subgraph 和 RecallBase 的构造函数。
    """

    def __init__(self, n_nodes: int, edges: pd.DataFrame | None = None) -> None:
        self._n = n_nodes
        self._out: list[set[int]] = [set() for _ in range(n_nodes)]
        self._in: list[set[int]] = [set() for _ in range(n_nodes)]
        self._n_edges = 0
        if edges is not None:
            for u, v in zip(edges["src"].to_numpy(), edges["dst"].to_numpy()):
                self.add_edge(int(u), int(v))

    # ── 写接口 ────────────────────────────────────────────────────────────────

    def add_edge(self, u: int, v: int) -> None:
        if v not in self._out[u]:
            self._out[u].add(v)
            self._in[v].add(u)
            self._n_edges += 1

    def add_edges(self, pairs: list[tuple[int, int]]) -> None:
        for u, v in pairs:
            self.add_edge(u, v)

    # ── 查询接口（duck-type TimeAdjacency）───────────────────────────────────

    def out_neighbors(self, node: int, cutoff: float = float("inf")) -> list[int]:  # noqa: ARG002
        return list(self._out[node])

    def in_neighbors(self, node: int, cutoff: float = float("inf")) -> list[int]:  # noqa: ARG002
        return list(self._in[node])

    def neighbors(self, node: int, cutoff: float = float("inf")) -> list[int]:  # noqa: ARG002
        return list(self._out[node] | self._in[node])

    def out_edges_at(self, node: int, cutoff: float = float("inf")) -> list[tuple[int, float]]:  # noqa: ARG002
        """返回 (dst, synthetic_timestamp=0.0)，保持与 TimeAdjacency 接口一致。"""
        return [(v, 0.0) for v in self._out[node]]

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def has_edge(self, u: int, v: int) -> bool:
        return v in self._out[u]

    def num_edges(self) -> int:
        return self._n_edges

    def n_nodes(self) -> int:
        return self._n

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        for u in range(self._n):
            for v in self._out[u]:
                yield u, v
