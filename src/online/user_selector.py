"""src/online/user_selector.py — 综合用户选择策略。

strategy='uniform'   : 均匀随机（向后兼容，等同原 sample_active_users）
strategy='composite' : 固有活跃度 × 度数因子 × 时间衰减 × 事件触发

公式（composite）：
    P_t(u) ∝ a_u · ((d_t(u)+1)/(d_max+1))^α · exp(-λ(t-t_last(u))) · (1 + γ·n_recent(u))
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.online.static_adj import StaticAdjacency


class UserSelector:
    """用户活跃度采样器。"""

    def __init__(
        self,
        n_nodes: int,
        strategy: str = "composite",
        alpha: float = 0.5,
        beta: float = 2.0,
        lam: float = 0.1,
        gamma: float = 2.0,
        w: int = 3,
        sample_ratio: float = 0.10,
        seed: int = 42,
    ) -> None:
        self._n = n_nodes
        self._strategy = strategy
        self._alpha = alpha
        self._lam = lam
        self._gamma = gamma
        self._w = w
        self._sample_ratio = sample_ratio
        self._rng = np.random.default_rng(seed)

        if strategy == "composite":
            # 固有活跃度：Pareto 分布（一次性采样，全程固定）
            raw = self._rng.pareto(beta, size=n_nodes) + 1.0
            self._a = raw / raw.sum()
        else:
            self._a = np.ones(n_nodes, dtype=np.float64) / n_nodes

        self._t_last = np.zeros(n_nodes, dtype=np.float64)
        # 环形缓冲：最近 w 轮每用户新增边数
        self._recent_edges = np.zeros((n_nodes, w), dtype=np.float64)

    def select(self, t: int, adj: "StaticAdjacency") -> list[int]:
        k = max(1, int(self._n * self._sample_ratio))
        if self._strategy == "uniform":
            return self._rng.choice(self._n, size=k, replace=False).tolist()

        # composite 权重
        # O(N) numpy diff on CSR indptr，避免每轮 N 次 list(set) 转换
        indptr, _ = adj.get_csr()
        degrees = np.diff(indptr).astype(np.float64)
        d_max = degrees.max() + 1.0
        degree_factor = ((degrees + 1.0) / d_max) ** self._alpha
        time_factor = np.exp(-self._lam * (t - self._t_last))
        n_recent = self._recent_edges.sum(axis=1)
        event_factor = 1.0 + self._gamma * n_recent

        weights = self._a * degree_factor * time_factor * event_factor
        total = weights.sum()
        if total <= 0:
            probs = np.ones(self._n) / self._n
        else:
            probs = weights / total

        selected = self._rng.choice(self._n, size=k, replace=False, p=probs)
        self._t_last[selected] = t
        return selected.tolist()

    def update_after_round(self, t: int, accepted_edges: list[tuple[int, int]]) -> None:
        """每轮结束后更新事件触发状态。"""
        col = t % self._w
        new_counts = np.zeros(self._n, dtype=np.float64)
        for u, _ in accepted_edges:
            if 0 <= u < self._n:
                new_counts[u] += 1
        self._recent_edges[:, col] = new_counts
