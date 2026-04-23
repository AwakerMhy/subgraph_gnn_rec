"""src/model/encoder_attr.py — 节点属性编码器

将 u、v 两个节点的属性向量 concat 后经 MLP 编码为固定维 embedding，
与图级 GNN embedding concat 后送 Scorer。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AttrEncoder(nn.Module):
    """u/v 节点属性 → 属性 embedding。

    输入：u_feat (B, feat_dim) 和 v_feat (B, feat_dim)
    输出：h_attr (B, out_dim)

    Args:
        feat_dim: 每个节点的属性维度（如 in_degree/out_degree/total 为 3）
        out_dim:  输出维度，通常与 GNN hidden_dim 一致
    """

    def __init__(self, feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, u_feat: torch.Tensor, v_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u_feat: (B, feat_dim)
            v_feat: (B, feat_dim)

        Returns:
            h_attr: (B, out_dim)
        """
        return self.mlp(torch.cat([u_feat, v_feat], dim=-1))
