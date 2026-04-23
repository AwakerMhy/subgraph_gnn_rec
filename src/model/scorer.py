"""src/model/scorer.py — MLP 评分头

输入: graph embedding 向量
输出: 标量分数 ∈ (0, 1)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Scorer(nn.Module):
    """两层 MLP + sigmoid，输出链接预测分数。

    Args:
        in_dim:     输入维度（= GINEncoder 的 hidden_dim）
        hidden_dim: 隐层维度
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (..., in_dim)

        Returns:
            score: (...,) ∈ (0, 1)
        """
        return self.mlp(h).squeeze(-1)
