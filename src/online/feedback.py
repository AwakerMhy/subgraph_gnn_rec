"""src/online/feedback.py — 用户反馈模拟器。"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Feedback:
    accepted: list[tuple[int, int]] = field(default_factory=list)
    rejected: list[tuple[int, int]] = field(default_factory=list)
    recs: dict[int, list[int]] = field(default_factory=dict)


class FeedbackSimulator:
    """对推荐结果做概率化伯努利采样，决定哪些边被用户接受。

    (u,v) ∈ G*：以 p_pos 接受；(u,v) ∉ G*：以 p_neg 接受（探索性连接）。
    p_neg=0.0 时退化为旧行为（仅 G* 内的边可能被接受）。
    """

    def __init__(
        self,
        star_edge_set: set[tuple[int, int]],
        p_pos: float = 1.0,
        p_neg: float = 0.0,
        rng: np.random.Generator | None = None,
        # 向后兼容：p_accept 映射到 p_pos（已废弃，下个迭代删除）
        p_accept: float | None = None,
    ) -> None:
        if p_accept is not None:
            warnings.warn(
                "FeedbackSimulator: p_accept 已废弃，请改用 p_pos。"
                "p_accept 将在下个迭代删除。",
                DeprecationWarning,
                stacklevel=2,
            )
        self._star = star_edge_set
        # p_pos 优先；仅当 p_pos 未显式传入（仍为默认值 1.0）且 p_accept 存在时才用 p_accept
        self._p_pos = p_pos if p_pos != 1.0 or p_accept is None else p_accept
        self._p_neg = p_neg
        self._rng = rng or np.random.default_rng(42)

    def simulate(self, recs: dict[int, list[int]]) -> Feedback:
        fb = Feedback(recs=recs)
        for u, vs in recs.items():
            for v in vs:
                in_star = (u, v) in self._star
                p = self._p_pos if in_star else self._p_neg
                if p >= 1.0 or self._rng.random() < p:
                    fb.accepted.append((u, v))
                else:
                    fb.rejected.append((u, v))
        return fb
