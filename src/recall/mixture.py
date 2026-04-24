"""src/recall/mixture.py — 多召回器组合（Mixture Recall）。

按配额从多个子召回器中各取候选，去重后合并。
各子召回器分数先做 min-max 归一化再合并（量纲统一）。
最终按归一化分数降序，不足 top_k 时不做随机填充（由上层 loop 负责冷启动兜底）。
"""
from __future__ import annotations

import numpy as np

from src.recall.base import RecallBase


class MixtureRecall(RecallBase):
    """按配额组合多个子召回器。

    components: list of (RecallBase, quota: int)
    总候选数 = sum(quota)，调用方应设 top_k <= sum(quota)。
    """

    def __init__(self, components: list[tuple[RecallBase, int]]) -> None:
        self._components = components

    def update_graph(self, round_idx: int) -> None:
        for recall, _ in self._components:
            recall.update_graph(round_idx)

    def precompute_for_users(self, users: list[int]) -> None:
        """委托给支持批量预计算的子召回器（如 PPRRecall）。"""
        for recall, _ in self._components:
            if hasattr(recall, "precompute_for_users"):
                recall.precompute_for_users(users)

    def candidates(
        self,
        u: int,
        cutoff_time: float,
        top_k: int,
    ) -> list[tuple[int, float]]:
        seen: set[int] = set()
        merged: list[tuple[int, float]] = []

        for recall, quota in self._components:
            cands = recall.candidates(u, cutoff_time, quota)
            for v, score in cands:
                if v not in seen:
                    seen.add(v)
                    merged.append((v, score))

        if not merged:
            return []

        # min-max 归一化（所有候选分数到 [0, 1]）
        scores_arr = np.array([s for _, s in merged], dtype=np.float64)
        s_min, s_max = scores_arr.min(), scores_arr.max()
        if s_max > s_min:
            scores_arr = (scores_arr - s_min) / (s_max - s_min)
        else:
            scores_arr = np.ones_like(scores_arr)

        result = [(v, float(s)) for (v, _), s in zip(merged, scores_arr)]
        result.sort(key=lambda x: -x[1])
        return result[:top_k]
