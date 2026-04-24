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
        # 批量预计算缓存：user -> PPR 向量（由 precompute_for_users 填充）
        self._ppr_cache: dict[int, np.ndarray] = {}
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
        cur_edges = self._adj.num_edges()
        if hasattr(self, "_last_n_edges") and cur_edges == self._last_n_edges:
            self._ppr_cache.clear()
            return
        self._last_n_edges = cur_edges
        self._ppr_cache.clear()
        self._update_matrix()

    def precompute_for_users(self, users: list[int]) -> None:
        """批量 power iteration：一次矩阵×矩阵替代逐用户矩阵×向量，快 50-100×。"""
        if self._trans is None or not users:
            return
        n, m = self._n, len(users)
        users_arr = np.array(users, dtype=np.int32)

        P = np.zeros((n, m), dtype=np.float32)
        P[users_arr, np.arange(m)] = 1.0
        E = P.copy()
        T = self._trans.T  # CSR sparse (n×n)

        for _ in range(self._max_iter):
            P_new = self._alpha * E + (1.0 - self._alpha) * (T @ P)
            if np.abs(P_new - P).sum(axis=0).max() < 1e-4:
                P = P_new
                break
            P = P_new

        for j, u in enumerate(users):
            self._ppr_cache[u] = P[:, j]

    def candidates(
        self,
        u: int,
        cutoff_time: float,  # noqa: ARG002
        top_k: int,
    ) -> list[tuple[int, float]]:
        if self._trans is None:
            return []

        if u in self._ppr_cache:
            p = self._ppr_cache[u]
        else:
            # fallback：逐用户 power iteration（未调用 precompute_for_users 时）
            p = np.zeros(self._n, dtype=np.float32)
            p[u] = 1.0
            e_u = p.copy()
            for _ in range(self._max_iter):
                p_new = self._alpha * e_u + (1.0 - self._alpha) * (self._trans.T @ p)
                if np.linalg.norm(p_new - p, ord=1) < 1e-4:
                    p = p_new
                    break
                p = p_new

        # 排除 u 自身及已有出边邻居，只遍历 PPR 非零项
        exclude = set(self._adj.out_neighbors(u)) | {u}
        nonzero = np.where(p > 0)[0]
        scores = {int(v): float(p[v]) for v in nonzero if v not in exclude}
        if not scores:
            return []
        sorted_cands = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_cands[:top_k]
