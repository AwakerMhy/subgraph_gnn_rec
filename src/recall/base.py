"""src/recall/base.py — 召回器抽象基类"""
from __future__ import annotations

from abc import ABC, abstractmethod


class RecallBase(ABC):
    """召回器接口。

    给定源节点 u 和截断时刻 cutoff_time，在观测图 G_obs 上生成候选列表。
    候选列表只做结构检索，不负责标注正负；标注由 LinkPredDataset 完成。

    返回格式：[(v, score), ...]，按 score 降序，最多 top_k 条。
    score 本身也会被传递给 CurriculumScheduler 用于难度调节。
    """

    @abstractmethod
    def candidates(
        self,
        u: int,
        cutoff_time: float,
        top_k: int,
    ) -> list[tuple[int, float]]:
        """返回 u 在 cutoff_time 前的 top_k 候选节点及其分数。

        Args:
            u:            查询节点 ID
            cutoff_time:  截断时刻（只使用 t < cutoff_time 的边）
            top_k:        最多返回的候选数量

        Returns:
            [(v, score), ...] 按 score 降序排列，不含 u 自身
        """
        ...

    def update_graph(self, round_idx: int) -> None:
        """每轮开始前由主循环调用一次，用于更新图级别缓存（如稀疏矩阵、社区划分）。

        默认无操作；PPRRecall / CommunityRandomRecall / MixtureRecall 可选覆盖。
        """
