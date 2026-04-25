"""src/online/static_adj.py — 动态静态邻接表，duck-type TimeAdjacency 接口。

忽略 cutoff 参数，返回当前图中所有邻居。用于在线场景下图结构动态变化时
无缝替换 TimeAdjacency，与 extract_subgraph / RecallBase 完全兼容。
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


class StaticAdjacency:
    """基于 dict[int, set[int]] 的可变邻接表。

    接口与 TimeAdjacency 一致（cutoff 参数保留但忽略），可作为 time_adj 参数
    直接传入 extract_subgraph 和 RecallBase 的构造函数。

    额外维护懒构建的出边 CSR（indptr/indices），供扁平化子图批量提取使用。
    每次 add_edge 后 dirty=True，首次被 get_csr() 访问时重建，每轮至多重建一次。
    """

    def __init__(self, n_nodes: int, edges: pd.DataFrame | None = None) -> None:
        self._n = n_nodes
        self._out: list[set[int]] = [set() for _ in range(n_nodes)]
        self._in: list[set[int]] = [set() for _ in range(n_nodes)]
        self._n_edges = 0
        # 出边 CSR（懒构建）
        self._csr_dirty: bool = True
        self._csr_indptr: np.ndarray | None = None
        self._csr_indices: np.ndarray | None = None
        if edges is not None:
            for u, v in zip(edges["src"].to_numpy(), edges["dst"].to_numpy()):
                self.add_edge(int(u), int(v))

    # ── 写接口 ────────────────────────────────────────────────────────────────

    def add_edge(self, u: int, v: int) -> None:
        if v not in self._out[u]:
            self._out[u].add(v)
            self._in[v].add(u)
            self._n_edges += 1
            self._csr_dirty = True

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

    # ── 高性能接口 ────────────────────────────────────────────────────────────

    def iter_out_neighbors(self, node: int, cutoff: float = float("inf")) -> set[int]:  # noqa: ARG002
        """直接返回出边邻居集合引用，避免复制，仅用于只读迭代。"""
        return self._out[node]

    def get_csr(self) -> tuple[np.ndarray, np.ndarray]:
        """返回出边 CSR 表示 (indptr, indices)，懒构建，每行 indices 升序排列。

        indices 升序是 _extract_edges_csr_fast 中 np.searchsorted 的前提。
        """
        if self._csr_dirty or self._csr_indptr is None:
            n = self._n
            indptr = np.zeros(n + 1, dtype=np.int32)
            for u in range(n):
                indptr[u + 1] = indptr[u] + len(self._out[u])
            total = int(indptr[n])
            indices = np.empty(total, dtype=np.int32)
            for u in range(n):
                s = int(indptr[u])
                nbrs = np.array(sorted(self._out[u]), dtype=np.int32)
                indices[s: s + len(nbrs)] = nbrs
            self._csr_indptr = indptr
            self._csr_indices = indices
            self._csr_dirty = False
        return self._csr_indptr, self._csr_indices

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    def has_edge(self, u: int, v: int) -> bool:
        return v in self._out[u]

    def out_degree(self, node: int) -> int:
        return len(self._out[node])

    def in_degree(self, node: int) -> int:
        return len(self._in[node])

    def out_neighbors_set(self, node: int) -> set[int]:
        """只读返回 out 邻居集合的引用，供需要 set 视图的调用者使用。"""
        return self._out[node]

    def in_neighbors_set(self, node: int) -> set[int]:
        """只读返回 in 邻居集合的引用，供需要 set 视图的调用者使用。"""
        return self._in[node]

    def num_edges(self) -> int:
        return self._n_edges

    def n_nodes(self) -> int:
        return self._n

    def iter_edges(self) -> Iterator[tuple[int, int]]:
        for u in range(self._n):
            for v in self._out[u]:
                yield u, v
