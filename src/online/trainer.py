"""src/online/trainer.py — 在线训练器（梯度更新 + 批量打分）。"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.graph.subgraph import extract_subgraph
from src.online.static_adj import StaticAdjacency


class OnlineTrainer:
    """封装子图提取 → 批量前向 → BCE loss → 梯度更新。

    score() 调用时使用 no_grad + eval mode，update() 切回 train mode。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        device: torch.device | str,
        max_hop: int = 2,
        max_neighbors: int = 30,
        node_feat: "torch.Tensor | None" = None,
        min_batch_size: int = 4,
        grad_clip: float = 1.0,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.max_hop = max_hop
        self.max_neighbors = max_neighbors
        # extract_subgraph 在 CPU 上构建图，node_feat 必须保持在 CPU
        self.node_feat = node_feat.cpu() if node_feat is not None else None
        self.min_batch_size = min_batch_size
        self.grad_clip = grad_clip

    # ── 子图构建 ──────────────────────────────────────────────────────────────

    def _build_subgraphs(
        self,
        pairs: list[tuple[int, int]],
        adj: StaticAdjacency,
        seed: int = 42,
    ) -> tuple["list[dgl.DGLGraph]", list[int]]:
        """为 pairs 提取子图，返回 (graphs, valid_indices)。"""
        import dgl  # noqa: PLC0415

        graphs, valid_idx = [], []
        for i, (u, v) in enumerate(pairs):
            g = extract_subgraph(
                u, v,
                cutoff_time=float("inf"),
                edges=None,
                max_hop=self.max_hop,
                max_neighbors_per_node=self.max_neighbors,
                seed=seed + i,
                time_adj=adj,
                node_feat=self.node_feat,
            )
            if g is not None:
                graphs.append(g)
                valid_idx.append(i)
        return graphs, valid_idx

    # ── 打分（推理模式）───────────────────────────────────────────────────────

    def score(
        self,
        u: int,
        candidates: list[int],
        adj: StaticAdjacency,
    ) -> list[float]:
        """对 (u, v) for v in candidates 批量打分，返回 float 列表。"""
        if not candidates:
            return []

        import dgl  # noqa: PLC0415

        pairs = [(u, v) for v in candidates]
        graphs, valid_idx = self._build_subgraphs(pairs, adj)
        if not graphs:
            return [0.0] * len(candidates)

        bg = dgl.batch(graphs).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores_tensor = self.model.forward_batch(bg)

        scores_list = [0.0] * len(candidates)
        for rank, orig_i in enumerate(valid_idx):
            scores_list[orig_i] = scores_tensor[rank].item()
        return scores_list

    # ── 梯度更新（训练模式）──────────────────────────────────────────────────

    def update(
        self,
        pos_pairs: list[tuple[int, int]],
        neg_pairs: list[tuple[int, int]],
        adj: StaticAdjacency,
    ) -> dict[str, float]:
        """用本轮正负样本做一步梯度更新。返回 {'loss': float}。"""
        import dgl  # noqa: PLC0415

        all_pairs = pos_pairs + neg_pairs
        if len(all_pairs) < self.min_batch_size:
            return {"loss": float("nan"), "skipped": 1}

        graphs, valid_idx = self._build_subgraphs(all_pairs, adj)
        if not graphs:
            return {"loss": float("nan"), "skipped": 1}

        labels_all = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)
        labels = torch.tensor(
            [labels_all[i] for i in valid_idx], dtype=torch.float32, device=self.device
        )

        bg = dgl.batch(graphs).to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model.forward_batch(bg)
        loss = nn.functional.binary_cross_entropy(preds, labels)
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss.item()}
