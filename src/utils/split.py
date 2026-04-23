"""src/utils/split.py — 时间感知数据集切分"""
from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(
    edges: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    # test_ratio 隐含 = 1 - train_ratio - val_ratio
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按时间戳分位数切分有向边列表，严格防止数据泄露。

    Args:
        edges: 含列 [src, dst, timestamp] 的 DataFrame，已按 timestamp 升序排序。
        train_ratio: 训练集比例（默认 0.70）。
        val_ratio:   验证集比例（默认 0.15）。

    Returns:
        (train_edges, val_edges, test_edges)，每个均为 DataFrame 子集（重置索引）。

    Raises:
        AssertionError: 切分后存在时间泄露或比例不合理。
    """
    assert "timestamp" in edges.columns, "edges 必须含 'timestamp' 列"
    assert train_ratio + val_ratio < 1.0, "train + val 比例之和必须小于 1"
    assert len(edges) >= 3, "边数量必须至少为 3"

    edges = edges.sort_values("timestamp").reset_index(drop=True)
    n = len(edges)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # 确保每个分割至少有 1 条边
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_edges = edges.iloc[:train_end].reset_index(drop=True)
    val_edges = edges.iloc[train_end:val_end].reset_index(drop=True)
    test_edges = edges.iloc[val_end:].reset_index(drop=True)

    # ── 防止时间泄露断言 ──────────────────────────────────────────────
    t_train_max = train_edges["timestamp"].max()
    t_val_min = val_edges["timestamp"].min()
    t_val_max = val_edges["timestamp"].max()
    t_test_min = test_edges["timestamp"].min()

    assert t_val_min >= t_train_max, (
        f"时间泄露：val 最小时间戳 {t_val_min} < train 最大时间戳 {t_train_max}"
    )
    assert t_test_min >= t_val_max, (
        f"时间泄露：test 最小时间戳 {t_test_min} < val 最大时间戳 {t_val_max}"
    )

    # ── 大小断言 ─────────────────────────────────────────────────────
    assert len(train_edges) > 0, "train_edges 为空"
    assert len(val_edges) > 0, "val_edges 为空"
    assert len(test_edges) > 0, "test_edges 为空"
    assert len(train_edges) + len(val_edges) + len(test_edges) == n

    return train_edges, val_edges, test_edges


def get_cutoff_times(
    train_edges: pd.DataFrame,
    val_edges: pd.DataFrame,
    test_edges: pd.DataFrame,
) -> dict[str, float]:
    """返回各阶段的截断时刻（cutoff_time）。

    子图提取时使用：提取 (u, v, t_q) 对应子图，cutoff_time = t_q，
    即只保留 timestamp < t_q 的历史边。
    """
    return {
        "train_cutoff": float(train_edges["timestamp"].max()),
        "val_cutoff": float(val_edges["timestamp"].max()),
        "test_cutoff": float(test_edges["timestamp"].max()),
    }
