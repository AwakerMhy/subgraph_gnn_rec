"""src/recall/heuristic.py — 启发式召回器（CN + AA）

召回语义（有向图，N_bidir = N_out ∪ N_in）：
    中间节点 z 被定义为：z ∈ N_bidir(u,t)，候选 v ∈ N_bidir(z,t)。
    与 loop.py 的打分函数保持一致（均用无向邻居集合）。

CommonNeighbors score:  |{z : z ∈ N_bidir(u) ∧ v ∈ N_bidir(z)}|
AdamicAdar score:       Σ_{z} 1 / log(|N_bidir(z)| + 2)  （对高度节点降权）

批量预计算（precompute_for_users）：
    大图：1 次 A_sym[users] @ A_sym 替代 |users| 次 set intersection，
    A_sym = clip(A + A^T, 0, 1)；适合在线场景每轮 100-400 个活跃用户。
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


def _build_sparse_adj_sym(time_adj: "TimeAdjacency", n: int):
    """构建对称邻接矩阵 A_sym = clip(A + A^T, 0, 1)，用于双向 2-hop 计算。"""
    A = _build_sparse_adj(time_adj, n)
    if A is None:
        return None
    A_sym = (A + A.T).astype(np.float32)
    A_sym.data[:] = 1.0  # 消除 A+A^T 中值为 2 的双向重复边，保持 0/1
    return A_sym


def _two_hop_bidir_scores(
    u: int,
    cutoff_time: float,
    time_adj: "TimeAdjacency",
    use_adamic_adar: bool = False,
) -> dict[int, float]:
    """双向 2-hop 候选集得分（小图 fallback）。

    N_bidir(u) = N_out(u) ∪ N_in(u)；候选 v 满足 ∃z∈N_bidir(u) 使 v∈N_bidir(z)。
    排除条件：v==u 或 u→v 已存在（v∈N_out(u)）。
    use_adamic_adar=True 时权重为 1/log(|N_bidir(z)|+2)，否则为 1.0。
    """
    n_out_u: set[int] = set(time_adj.out_neighbors(u, cutoff_time))
    n_in_u: set[int] = set(time_adj.in_neighbors(u, cutoff_time))
    n1 = n_out_u | n_in_u
    if not n1:
        return {}
    exclude = n_out_u | {u}
    scores: dict[int, float] = {}
    for z in n1:
        z_bidir = set(time_adj.out_neighbors(z, cutoff_time)) | set(time_adj.in_neighbors(z, cutoff_time))
        weight = 1.0 / math.log(len(z_bidir) + 2) if use_adamic_adar else 1.0
        for v in z_bidir:
            if v not in exclude:
                scores[v] = scores.get(v, 0.0) + weight
    return scores


# 小图（节点数 <= 阈值）使用逐用户 set intersection，大图使用 sparse matmul
# 经验阈值：todense() 分配 n² 内存，n<10k 时 set ops 更快
_SPARSE_MATMUL_THRESHOLD = 10_000


# ── CommonNeighborsRecall ─────────────────────────────────────────────────────

class CommonNeighborsRecall(RecallBase):
    """基于共同邻居数的召回器，支持批量 sparse matmul 预计算。

    使用 N_bidir = N_out ∪ N_in 定义邻居，与 loop.py 打分保持一致。
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes
        self._A = _build_sparse_adj(time_adj, n_nodes)        # 有向 CSR（供子类使用）
        self._A_sym = _build_sparse_adj_sym(time_adj, n_nodes) # 对称 CSR（CN 计算用）
        self._cache: dict[int, np.ndarray] = {}
        self._last_n_edges: int = -1

    # ── 图更新 ────────────────────────────────────────────────────────────────

    def update_graph(self, round_idx: int) -> None:  # noqa: ARG002
        cur_edges = self._time_adj.num_edges()
        if cur_edges == self._last_n_edges:
            self._cache.clear()
            return
        self._last_n_edges = cur_edges
        self._cache.clear()
        self._A = _build_sparse_adj(self._time_adj, self._n_nodes)
        self._A_sym = _build_sparse_adj_sym(self._time_adj, self._n_nodes)

    # ── 批量预计算（loop.py 中自动调用）──────────────────────────────────────

    def precompute_for_users(self, users: list[int]) -> None:
        """小图（n<=10k）逐用户 set intersection；大图 1 次 A_sym @ A_sym。"""
        if not users:
            return
        if self._A_sym is None or self._n_nodes <= _SPARSE_MATMUL_THRESHOLD:
            self._cache = {}
            for u in users:
                self._cache[u] = None
            return
        users_arr = np.array(users, dtype=np.int32)
        CN = self._A_sym[users_arr].dot(self._A_sym)
        CN_dense = np.asarray(CN.todense(), dtype=np.float32)
        self._cache = {u: CN_dense[i] for i, u in enumerate(users)}

    # ── 候选查询 ──────────────────────────────────────────────────────────────

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
        scores_d = _two_hop_bidir_scores(u, cutoff_time, self._time_adj, use_adamic_adar=False)
        if not scores_d:
            return []
        return sorted(scores_d.items(), key=lambda x: -x[1])[:top_k]


# ── TwoHopRandomRecall ───────────────────────────────────────────────────────

class TwoHopRandomRecall(CommonNeighborsRecall):
    """2-hop 候选池随机截断——使用有向出边 2-hop（N_out only），随机采样。

    保留有向候选语义（区别于 TwoHopBidirRandomRecall 的双向版本）。
    precompute_for_users 覆盖父类，使用有向邻接矩阵 self._A。
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int, seed: int = 42) -> None:
        super().__init__(time_adj, n_nodes)
        self._rng = np.random.default_rng(seed)

    def precompute_for_users(self, users: list[int]) -> None:
        """使用有向 A（N_out only），保持原始 two_hop_random 语义。"""
        if not users:
            return
        if self._A is None or self._n_nodes <= _SPARSE_MATMUL_THRESHOLD:
            self._cache = {u: None for u in users}
            return
        users_arr = np.array(users, dtype=np.int32)
        CN = self._A[users_arr].dot(self._A)
        CN_dense = np.asarray(CN.todense(), dtype=np.float32)
        self._cache = {u: CN_dense[i] for i, u in enumerate(users)}

    def candidates(self, u: int, cutoff_time: float, top_k: int) -> list[tuple[int, float]]:
        scores = self._cache.get(u)
        if scores is not None:
            exclude = set(self._time_adj.out_neighbors(u)) | {u}
            nonzero = np.where(scores > 0)[0]
            cands = [int(v) for v in nonzero if v not in exclude]
        else:
            scores_d = _two_hop_scores(u, cutoff_time, self._time_adj, use_adamic_adar=False)
            cands = list(scores_d.keys())

        if not cands:
            return []
        if len(cands) <= top_k:
            return [(v, 1.0) for v in cands]
        chosen = self._rng.choice(len(cands), size=top_k, replace=False)
        return [(cands[i], 1.0) for i in chosen]


# ── TwoHopBidirRandomRecall ──────────────────────────────────────────────────

class TwoHopBidirRandomRecall(RecallBase):
    """双向 2-hop 候选池随机召回。

    以无向方式处理有向图：N_bidir(u) = N_out(u) ∪ N_in(u)，
    候选 v 满足 ∃z∈N_bidir(u) 使 v∈N_bidir(z)。
    从候选池中随机采样 top_k 个（不按路径数排序）。

    排除条件：v==u 或 u→v 已存在（v∈N_out(u)）。不排除 v→u 已存在的节点。

    大图路径（n>10k）：A_sym = A + A^T，C = A_sym[users] @ A_sym，
    1 次矩阵乘法覆盖所有活跃用户。
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int, seed: int = 42) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes
        self._A_sym = _build_sparse_adj_sym(time_adj, n_nodes)
        self._cache: dict[int, list[int] | None] = {}
        self._last_n_edges: int = -1
        self._rng = np.random.default_rng(seed)

    def update_graph(self, round_idx: int) -> None:  # noqa: ARG002
        cur_edges = self._time_adj.num_edges()
        if cur_edges == self._last_n_edges:
            self._cache.clear()
            return
        self._last_n_edges = cur_edges
        self._cache.clear()
        self._A_sym = _build_sparse_adj_sym(self._time_adj, self._n_nodes)

    def precompute_for_users(self, users: list[int]) -> None:
        if not users:
            return
        if self._A_sym is None or self._n_nodes <= _SPARSE_MATMUL_THRESHOLD:
            # 小图：标记为 None，candidates() 走 fallback
            for u in users:
                self._cache[u] = None
            return
        # 大图：1 次 A_sym[users] @ A_sym，缓存候选列表
        users_arr = np.array(users, dtype=np.int32)
        C_dense = np.asarray(self._A_sym[users_arr].dot(self._A_sym).todense(), dtype=np.float32)
        for i, u in enumerate(users):
            exclude = set(self._time_adj.out_neighbors(u)) | {u}
            nonzero = np.where(C_dense[i] > 0)[0]
            self._cache[u] = [int(v) for v in nonzero if v not in exclude]

    def candidates(self, u: int, cutoff_time: float, top_k: int) -> list[tuple[int, float]]:
        cached = self._cache.get(u)
        if cached is not None:
            cands = cached
        else:
            scores_d = _two_hop_bidir_scores(u, cutoff_time, self._time_adj)
            cands = list(scores_d.keys())

        if not cands:
            return []
        if len(cands) <= top_k:
            return [(v, 1.0) for v in cands]
        chosen = self._rng.choice(len(cands), size=top_k, replace=False)
        return [(cands[i], 1.0) for i in chosen]


# ── GlobalRandomRecall ───────────────────────────────────────────────────────

class GlobalRandomRecall(RecallBase):
    """从全图节点中随机采样候选，突破 2-hop 结构约束。

    不依赖图结构，直接从 {0..n_nodes-1} \ {u} 中均匀随机抽取 top_k 个节点。
    配合 MixtureRecall 使用时可覆盖结构距离远的 G* 边。
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int, seed: int = 42) -> None:
        self._n = n_nodes
        self._rng = np.random.default_rng(seed)

    def update_graph(self, round_idx: int) -> None:
        pass

    def candidates(self, u: int, cutoff_time: float, top_k: int) -> list[tuple[int, float]]:
        all_nodes = np.arange(self._n, dtype=np.int32)
        pool = all_nodes[all_nodes != u]
        k = min(top_k, len(pool))
        chosen = self._rng.choice(pool, size=k, replace=False)
        return [(int(v), 0.0) for v in chosen]


# ── AdamicAdarRecall ──────────────────────────────────────────────────────────

class AdamicAdarRecall(RecallBase):
    """基于 Adamic-Adar 指数的召回器，支持批量 sparse matmul 预计算。

    AA(u, v) = Σ_{z ∈ N_bidir(u) ∩ N_bidir(v)} 1 / log(|N_bidir(z)| + 2)
             = (A_sym[u] @ diag(w) @ A_sym)[v]，  w[z] = 1/log(deg_bidir(z)+2)
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes
        self._A_sym = _build_sparse_adj_sym(time_adj, n_nodes)
        self._cache: dict[int, np.ndarray] = {}
        self._last_n_edges: int = -1

    def update_graph(self, round_idx: int) -> None:  # noqa: ARG002
        cur_edges = self._time_adj.num_edges()
        if cur_edges == self._last_n_edges:
            self._cache.clear()
            return
        self._last_n_edges = cur_edges
        self._cache.clear()
        self._A_sym = _build_sparse_adj_sym(self._time_adj, self._n_nodes)

    def precompute_for_users(self, users: list[int]) -> None:
        """小图（n<=10k）逐用户 set intersection；大图 1 次 A_sym @ diag(w) @ A_sym。"""
        if not users:
            return
        if self._A_sym is None or self._n_nodes <= _SPARSE_MATMUL_THRESHOLD:
            self._cache = {u: None for u in users}
            return
        deg_bidir = np.asarray(self._A_sym.sum(axis=1)).flatten()
        w = (1.0 / np.log(deg_bidir + 2)).astype(np.float32)
        W = _sp.diags(w)
        users_arr = np.array(users, dtype=np.int32)
        AA = self._A_sym[users_arr].dot(W).dot(self._A_sym)
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
        scores_d = _two_hop_bidir_scores(u, cutoff_time, self._time_adj, use_adamic_adar=True)
        if not scores_d:
            return []
        return sorted(scores_d.items(), key=lambda x: -x[1])[:top_k]
