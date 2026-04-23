"""src/recall/ppr.py — 基于个性化 PageRank 的召回器。

算法：从 u 出发做 random walk with restart（power iteration），
返回稳态概率最高的非邻居节点作为候选。

每轮调用 update_graph() 时重建 G_t 的行归一化稀疏矩阵，
候选查询时只做向量乘法，无重复构图开销。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from src.recall.base import RecallBase

if TYPE_CHECKING:
    from src.online.static_adj import StaticAdjacency


class PPRRecall(RecallBase):
    """个性化 PageRank 召回器。

    score(u, v) = PPR 稳态概率（从 u 出发，restart prob = alpha）
    """

    def __init__(
        self,
        adj: "StaticAdjacency",
        n_nodes: int,
        alpha: float = 0.15,
        max_iter: int = 20,
    ) -> None:
        self._adj = adj
        self._n = n_nodes
        self._alpha = alpha
        self._max_iter = max_iter
        self._trans: sp.csr_matrix | None = None
        self._update_matrix()

    def _update_matrix(self) -> None:
        rows, cols = [], []
        for u in range(self._n):
            nbrs = self._adj.out_neighbors(u)
            if nbrs:
                for v in nbrs:
                    rows.append(u)
                    cols.append(v)
        if rows:
            data = np.ones(len(rows), dtype=np.float32)
            A = sp.csr_matrix((data, (rows, cols)), shape=(self._n, self._n))
            # 行归一化（对无出边节点，行和为 0，保留全零行）
            row_sums = np.array(A.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            inv = sp.diags(1.0 / row_sums)
            self._trans = (inv @ A).astype(np.float32)
        else:
            self._trans = sp.csr_matrix((self._n, self._n), dtype=np.float32)

    def update_graph(self, round_idx: int) -> None:  # noqa: ARG002
        self._update_matrix()

    def candidates(
        self,
        u: int,
        cutoff_time: float,  # noqa: ARG002
        top_k: int,
    ) -> list[tuple[int, float]]:
        if self._trans is None:
            return []
        # power iteration：p = alpha * e_u + (1-alpha) * p @ T
        p = np.zeros(self._n, dtype=np.float32)
        p[u] = 1.0
        e_u = p.copy()
        for _ in range(self._max_iter):
            p = self._alpha * e_u + (1.0 - self._alpha) * (self._trans.T @ p)

        # 排除 u 自身及已有出边邻居
        exclude = set(self._adj.out_neighbors(u)) | {u}
        scores = {v: float(p[v]) for v in range(self._n) if v not in exclude and p[v] > 0}
        if not scores:
            return []
        sorted_cands = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_cands[:top_k]
