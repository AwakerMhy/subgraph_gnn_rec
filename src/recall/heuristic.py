"""src/recall/heuristic.py — 启发式召回器（CN + AA）

召回语义（有向图，社交推荐方向：预测 u → v）：
    中间节点 z 被定义为：z ∈ N_out(u,t) 且 v ∈ N_out(z,t)
    即"u 关注的人 z，z 也关注 v"——典型的"朋友的朋友"路径。

CommonNeighbors score:  |{z : z ∈ N_out(u) ∧ v ∈ N_out(z)}|
AdamicAdar score:       Σ_{z} 1 / log(|N_out(z)| + 2)  （对高出度中间节点降权）

批量预计算（precompute_for_users）：
    1 次 scipy 稀疏 matmul A[users] @ A.T 替代 |users| 次 set intersection，
    适合在线场景每轮 100-400 个活跃用户的批量查询。
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from src.recall.base import RecallBase

if TYPE_CHECKING:
    from src.graph.subgraph import TimeAdjacency

try:
    import scipy.sparse as _sp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── 单用户 fallback（图较小 / scipy 不可用时）────────────────────────────────

def _two_hop_scores(
    u: int,
    cutoff_time: float,
    time_adj: "TimeAdjacency",
    use_adamic_adar: bool = False,
) -> dict[int, float]:
    """计算 u 的所有 2-hop 候选节点的得分（单用户 fallback）。"""
    _iter = getattr(time_adj, "iter_out_neighbors", None)
    if _iter is not None:
        n1_raw = _iter(u, cutoff_time)
        n1_set: set[int] = n1_raw if isinstance(n1_raw, set) else set(n1_raw)
    else:
        n1_set = set(time_adj.out_neighbors(u, cutoff_time))

    if not n1_set:
        return {}

    scores: dict[int, float] = {}
    for z in n1_set:
        z_out = _iter(z, cutoff_time) if _iter is not None else time_adj.out_neighbors(z, cutoff_time)
        weight = 1.0 / math.log(len(z_out) + 2) if use_adamic_adar else 1.0
        for v in z_out:
            if v == u or v in n1_set:
                continue
            scores[v] = scores.get(v, 0.0) + weight
    return scores


def _build_sparse_adj(time_adj: "TimeAdjacency", n: int):
    """从 StaticAdjacency 的 CSR 缓存直接构建 scipy CSR 矩阵，无额外遍历。"""
    if not _HAS_SCIPY:
        return None
    if hasattr(time_adj, "get_csr"):
        indptr, indices = time_adj.get_csr()
        data = np.ones(int(indptr[-1]), dtype=np.float32)
        return _sp.csr_matrix((data, indices, indptr), shape=(n, n))
    return None


# 小图（节点数 <= 阈值）使用逐用户 set intersection，大图使用 sparse matmul
# 经验阈值：todense() 分配 n² 内存，n<10k 时 set ops 更快
_SPARSE_MATMUL_THRESHOLD = 10_000


# ── CommonNeighborsRecall ─────────────────────────────────────────────────────

class CommonNeighborsRecall(RecallBase):
    """基于共同邻居数的召回器，支持批量 sparse matmul 预计算。"""

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes
        self._A = _build_sparse_adj(time_adj, n_nodes)   # scipy CSR or None
        self._cache: dict[int, np.ndarray] = {}           # user → CN scores (float32, shape n)

    # ── 图更新 ────────────────────────────────────────────────────────────────

    def update_graph(self, round_idx: int) -> None:  # noqa: ARG002
        self._cache.clear()
        self._A = _build_sparse_adj(self._time_adj, self._n_nodes)

    # ── 批量预计算（loop.py 中自动调用）──────────────────────────────────────

    def precompute_for_users(self, users: list[int]) -> None:
        """小图（n<=10k）逐用户 set intersection；大图 1 次 sparse matmul。"""
        if not users:
            return
        if self._A is None or self._n_nodes <= _SPARSE_MATMUL_THRESHOLD:
            # 小图：逐用户计算，避免 todense() 的 O(n²) 内存开销
            self._cache = {}
            for u in users:
                self._cache[u] = None   # 标记已处理，candidates() 走 fallback
            return
        # 大图：1 次 A[users] @ A，缓存 dense score 行
        users_arr = np.array(users, dtype=np.int32)
        CN = self._A[users_arr].dot(self._A)
        CN_dense = np.asarray(CN.todense(), dtype=np.float32)
        self._cache = {u: CN_dense[i] for i, u in enumerate(users)}

    # ── 候选查询 ──────────────────────────────────────────────────────────────

    def candidates(self, u: int, cutoff_time: float, top_k: int) -> list[tuple[int, float]]:
        scores = self._cache.get(u)
        if scores is not None:
            # 大图路径：dense score 向量
            exclude = set(self._time_adj.out_neighbors(u)) | {u}
            nonzero = np.where(scores > 0)[0]
            cands = [(int(v), float(scores[v])) for v in nonzero if v not in exclude]
            if not cands:
                return []
            cands.sort(key=lambda x: -x[1])
            return cands[:top_k]
        # 小图路径（scores is None 或 u 不在 cache）：逐用户 set intersection
        scores_d = _two_hop_scores(u, cutoff_time, self._time_adj, use_adamic_adar=False)
        if not scores_d:
            return []
        return sorted(scores_d.items(), key=lambda x: -x[1])[:top_k]


# ── AdamicAdarRecall ──────────────────────────────────────────────────────────

class AdamicAdarRecall(RecallBase):
    """基于 Adamic-Adar 指数的召回器，支持批量 sparse matmul 预计算。

    AA(u, v) = Σ_{z ∈ N_out(u) ∩ N_in(v)} 1 / log(|N_out(z)| + 2)
             = (A[u] @ diag(w) @ A.T)[v]，  w[z] = 1/log(deg_out(z)+2)
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes
        self._A = _build_sparse_adj(time_adj, n_nodes)
        self._cache: dict[int, np.ndarray] = {}

    def update_graph(self, round_idx: int) -> None:  # noqa: ARG002
        self._cache.clear()
        self._A = _build_sparse_adj(self._time_adj, self._n_nodes)

    def precompute_for_users(self, users: list[int]) -> None:
        """小图（n<=10k）逐用户 set intersection；大图 1 次 sparse matmul。"""
        if not users:
            return
        if self._A is None or self._n_nodes <= _SPARSE_MATMUL_THRESHOLD:
            self._cache = {u: None for u in users}
            return
        deg_out = np.asarray(self._A.sum(axis=1)).flatten()
        w = (1.0 / np.log(deg_out + 2)).astype(np.float32)
        W = _sp.diags(w)
        users_arr = np.array(users, dtype=np.int32)
        AA = self._A[users_arr].dot(W).dot(self._A)
        AA_dense = np.asarray(AA.todense(), dtype=np.float32)
        self._cache = {u: AA_dense[i] for i, u in enumerate(users)}

    def candidates(self, u: int, cutoff_time: float, top_k: int) -> list[tuple[int, float]]:
        scores = self._cache.get(u)
        if scores is not None:
            exclude = set(self._time_adj.out_neighbors(u)) | {u}
            nonzero = np.where(scores > 0)[0]
            cands = [(int(v), float(scores[v])) for v in nonzero if v not in exclude]
            if not cands:
                return []
            cands.sort(key=lambda x: -x[1])
            return cands[:top_k]
        scores_d = _two_hop_scores(u, cutoff_time, self._time_adj, use_adamic_adar=True)
        if not scores_d:
            return []
        return sorted(scores_d.items(), key=lambda x: -x[1])[:top_k]
