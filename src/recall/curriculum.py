"""src/recall/curriculum.py — 课程学习难度调度器"""
from __future__ import annotations

import math


class CurriculumScheduler:
    """控制负样本难度随训练 epoch 单调递增。

    difficulty(epoch) ∈ [0.0, 1.0]:
      0.0 → easy（取召回候选的低分尾部，易于区分正样本）
      1.0 → hard（取召回候选的高分头部，与正样本竞争激烈）

    调度策略：
      linear  — 线性增长
      cosine  — 余弦缓入缓出
      step    — 分两段跳跃（0.0 → 0.5 → 1.0）
    """

    SCHEDULES = ("linear", "cosine", "step")

    def __init__(
        self,
        total_epochs: int,
        schedule: str = "linear",
        warmup_epochs: int = 0,
    ) -> None:
        assert schedule in self.SCHEDULES, f"未知 schedule: {schedule!r}"
        assert total_epochs > 0
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.warmup_epochs = warmup_epochs

    def difficulty(self, epoch: int) -> float:
        """返回 epoch（1-based）对应的难度值 [0, 1]。"""
        if epoch <= self.warmup_epochs:
            return 0.0
        effective = epoch - self.warmup_epochs
        n = max(self.total_epochs - self.warmup_epochs, 1)
        t = min(effective / n, 1.0)
        if self.schedule == "linear":
            return t
        elif self.schedule == "cosine":
            return (1.0 - math.cos(math.pi * t)) / 2.0
        else:  # step
            return 0.0 if t < 0.5 else (0.5 if t < 1.0 else 1.0)

    def top_k_range(self, epoch: int, top_k: int) -> tuple[int, int]:
        """返回负样本候选的切片区间 [start, end)。

        难度=0 → 取低分尾部（easy）；难度=1 → 取高分头部（hard）。
        窗口大小固定为 top_k // 2，沿难度轴平滑滑动。
        """
        d = self.difficulty(epoch)
        window = max(top_k // 2, 1)
        start = int((1.0 - d) * (top_k - window))
        end = min(start + window, top_k)
        return start, end
