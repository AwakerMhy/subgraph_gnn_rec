"""src/recall/heuristic.py — 启发式召回器（CN + AA）

召回语义（有向图，社交推荐方向：预测 u → v）：
    中间节点 z 被定义为：z ∈ N_out(u,t) 且 v ∈ N_out(z,t)
    即"u 关注的人 z，z 也关注 v"——典型的"朋友的朋友"路径。

CommonNeighbors score:  |{z : z ∈ N_out(u) ∧ v ∈ N_out(z)}|
AdamicAdar score:       Σ_{z} 1 / log(|N_out(z)| + 1)  （对高出度中间节点降权）

复用：TimeAdjacency.out_neighbors (src/graph/subgraph.py:58-62)，O(log degree) 查询
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.recall.base import RecallBase

if TYPE_CHECKING:
    from src.graph.subgraph import TimeAdjacency


def _two_hop_scores(
    u: int,
    cutoff_time: float,
    time_adj: "TimeAdjacency",
    use_adamic_adar: bool = False,
) -> dict[int, float]:
    """计算 u 的所有 2-hop 候选节点的得分。

    被 CommonNeighborsRecall 和 AdamicAdarRecall 共用，避免重复实现。
    """
    n1: list[int] = time_adj.out_neighbors(u, cutoff_time)
    if not n1:
        return {}

    n1_set = set(n1)
    scores: dict[int, float] = {}

    for z in n1:
        z_out = time_adj.out_neighbors(z, cutoff_time)
        if use_adamic_adar:
            # Adamic-Adar：对高出度中间节点 z 降权
            weight = 1.0 / math.log(len(z_out) + 2)  # +2 防止 log(1)=0
        else:
            weight = 1.0

        for v in z_out:
            if v == u or v in n1_set:
                continue  # 排除自身与已知 1-hop 邻居
            scores[v] = scores.get(v, 0.0) + weight

    return scores


class CommonNeighborsRecall(RecallBase):
    """基于共同邻居数的召回器。

    Score(u, v) = |{z : z ∈ N_out(u,t) ∧ v ∈ N_out(z,t)}|

    直觉：u 关注的人中，有多少人也关注了 v。
    分数越高表示 v 在 u 的社交圈内越受认可。
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes

    def candidates(
        self,
        u: int,
        cutoff_time: float,
        top_k: int,
    ) -> list[tuple[int, float]]:
        scores = _two_hop_scores(u, cutoff_time, self._time_adj, use_adamic_adar=False)
        if not scores:
            return []
        sorted_cands = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_cands[:top_k]


class AdamicAdarRecall(RecallBase):
    """基于 Adamic-Adar 指数的召回器。

    Score(u, v) = Σ_{z ∈ N_out(u) ∩ N_in(v)} 1 / log(|N_out(z)| + 2)

    比 CN 更优：对高出度"大 V"中间节点降权，对小圈子共同好友加权。
    """

    def __init__(self, time_adj: "TimeAdjacency", n_nodes: int) -> None:
        self._time_adj = time_adj
        self._n_nodes = n_nodes

    def candidates(
        self,
        u: int,
        cutoff_time: float,
        top_k: int,
    ) -> list[tuple[int, float]]:
        scores = _two_hop_scores(u, cutoff_time, self._time_adj, use_adamic_adar=True)
        if not scores:
            return []
        sorted_cands = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_cands[:top_k]
