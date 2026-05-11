"""src/baseline/seal_full.py — SEAL-Full: SortPooling + 1D CNN (DGCNN style)

原版 SEAL（Zhang & Chen 2018）的完整 readout 实现：
  1. DRNL 标签嵌入 + GIN（与 seal.py 相同）
  2. SortPooling：按最后一层输出的末维值降序排列节点，截断/补零到固定长度 k
  3. 1D CNN：两层 Conv1d 在排序后的节点序列上滑动，提取局部模式
  4. AdaptiveAvgPool1d(1) + MLP Scorer → 标量分数

与 seal.py 的唯一差异在于 readout（步骤 2-4），DRNL 计算完全复用。
"""
from __future__ import annotations

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn

from src.baseline.seal import _compute_drnl   # 复用已有 DRNL 实现
from src.model.scorer import Scorer


class SEALFullModel(nn.Module):
    """SEAL with SortPooling + 1D CNN.

    Args:
        hidden_dim:      GIN 隐层维度
        num_layers:      GIN 层数
        label_dim:       DRNL 标签嵌入维度
        k:               SortPooling 保留节点数；子图节点数不足 k 时补零
        conv1d_channels: 两层 1D CNN 的输出通道数 (ch1, ch2)
        max_label:       DRNL 最大标签值（超出 clamp）
        scorer_hidden_dim: Scorer MLP 隐层维度（None 则等于 conv1d_channels[1]）
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 3,
        label_dim: int = 16,
        k: int = 10,
        conv1d_channels: tuple[int, int] = (32, 16),
        max_label: int = 50,
        scorer_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.k = k
        self.max_label = max_label

        self.label_embedding = nn.Embedding(max_label + 1, label_dim, padding_idx=0)

        # GIN layers（与 SEALModel 完全相同）
        self.layers = nn.ModuleList()
        dims = [label_dim] + [hidden_dim] * num_layers
        for i in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.layers.append(dglnn.GINConv(mlp, aggregator_type="sum"))

        # 1D CNN：输入 (B, hidden_dim, k)，channels-first
        ch1, ch2 = conv1d_channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(hidden_dim, ch1, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(ch1, ch2, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)   # (B, ch2, 1) → (B, ch2)

        _sc_hidden = scorer_hidden_dim if scorer_hidden_dim is not None else ch2
        self.scorer = Scorer(in_dim=ch2, hidden_dim=_sc_hidden)

    # ── 内部工具 ───────────────────────────────────────────────────────────────

    def _run_gin(self, bg: dgl.DGLGraph, labels: torch.Tensor) -> torch.Tensor:
        """DRNL 嵌入 + GIN 传播，返回所有节点的最终表示 (total_N, hidden_dim)。"""
        h = self.label_embedding(labels)
        for conv in self.layers:
            h = conv(bg, h)
            h = torch.relu(h)
        return h

    def _sort_pool(self, h: torch.Tensor, num_nodes: list[int]) -> torch.Tensor:
        """SortPooling：按最后一层末维值降序，截断/补零到 k 个节点。

        Args:
            h:         (total_N, hidden_dim)
            num_nodes: 每张图的节点数列表

        Returns:
            (B, k, hidden_dim)
        """
        B = len(num_nodes)
        device = h.device
        out = torch.zeros(B, self.k, self.hidden_dim, device=device, dtype=h.dtype)
        offset = 0
        for i, n in enumerate(num_nodes):
            h_g = h[offset: offset + n]              # (n, H)
            idx = torch.argsort(h_g[:, -1], descending=True)
            h_sorted = h_g[idx]                      # (n, H)
            take = min(n, self.k)
            out[i, :take] = h_sorted[:take]
            offset += n
        return out                                   # (B, k, H)

    def _get_labels(self, bg: dgl.DGLGraph) -> torch.Tensor:
        """优先使用预计算的 _drnl，否则逐图计算。"""
        if "_drnl" in bg.ndata:
            return bg.ndata["_drnl"].clamp(0, self.max_label)
        graphs = dgl.unbatch(bg)
        parts = [_compute_drnl(g).to(bg.device) for g in graphs]
        return torch.cat(parts).clamp(0, self.max_label)

    def _forward_bg(self, bg: dgl.DGLGraph) -> torch.Tensor:
        """共用路径：batched graph → (B,) scores。"""
        labels = self._get_labels(bg)
        h = self._run_gin(bg, labels)                        # (total_N, H)

        num_nodes = bg.batch_num_nodes().tolist()
        pooled = self._sort_pool(h, num_nodes)               # (B, k, H)

        x = pooled.permute(0, 2, 1)                          # (B, H, k)
        x = self.conv1(x)                                    # (B, ch1, k//2)
        x = self.conv2(x)                                    # (B, ch2, k//2)
        x = self.global_pool(x).squeeze(-1)                  # (B, ch2)
        return self.scorer(x)                                # (B,)

    # ── 公共接口（与 SEALModel / LinkPredModel 完全相同）────────────────────────

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        return self._forward_bg(dgl.batch([g])).squeeze(0)   # scalar

    def forward_batch(self, bg: dgl.DGLGraph) -> torch.Tensor:
        return self._forward_bg(bg)                          # (B,)
