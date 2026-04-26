"""src/online/schedule.py — 学习率调度工厂。"""
from __future__ import annotations

import math

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int = 5,
    min_lr: float = 1e-5,
    strategy: str = "cosine_warmup",
    cycle_steps: int = 25,
) -> torch.optim.lr_scheduler.LambdaLR:
    """构建学习率调度器。

    strategy:
        cosine_warmup  — 线性 warmup + cosine decay（默认，推荐）
        cyclic         — 线性 warmup + 周期性 cosine 重置（应对 distribution shift）
        constant       — 固定学习率
        step           — 每 total_steps/3 步衰减至 0.1x
    """
    base_lr = optimizer.param_groups[0]["lr"]

    if strategy == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    if strategy == "step":
        step_size = max(total_steps // 3, 1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    if strategy == "cyclic":
        # warmup 后每 cycle_steps 轮做一次完整 cosine 周期（base_lr → min_lr → 重置）
        def _cyclic_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            progress = ((current_step - warmup_steps) % cycle_steps) / cycle_steps
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(min_lr / base_lr, cosine)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_cyclic_lambda)

    # cosine_warmup（默认）
    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr / base_lr, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
