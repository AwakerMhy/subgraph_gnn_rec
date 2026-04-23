"""src/online/replay.py — Replay buffer，capacity=0 时为 no-op。"""
from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    """存储历史轮次的 (pos, neg) 样本对，防止灾难性遗忘。

    capacity=0 表示禁用，push/sample 均为 no-op。
    """

    def __init__(self, capacity: int = 0) -> None:
        self._cap = capacity
        self._buf: deque[tuple[list, list]] = deque(maxlen=capacity) if capacity > 0 else deque()

    def push(self, pos: list[tuple[int, int]], neg: list[tuple[int, int]], round_idx: int) -> None:  # noqa: ARG002
        if self._cap <= 0:
            return
        self._buf.append((list(pos), list(neg)))

    def sample(self, n: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
        if self._cap <= 0 or n <= 0 or len(self._buf) == 0:
            return [], []
        idx = np.random.choice(len(self._buf), size=min(n, len(self._buf)), replace=False)
        pos, neg = [], []
        for i in idx:
            p, q = self._buf[i]
            pos.extend(p)
            neg.extend(q)
        return pos, neg

    def __len__(self) -> int:
        return len(self._buf)
