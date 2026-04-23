"""src/graph/edge_split.py — 两层图构造（Step 1-3）

New Link Prediction 任务定义
────────────────────────────
目标：预测从未连接过的节点对 (u, v) 是否会**首次**建立有向连接。
正样本语义 = "新社交关系形成"（而非历史关系的重复激活）。

两层图语义
──────────
    G（完整图）  = 全部社交关系（已发生 + 将要发生）
    G_obs        = 模型可见的历史观测图（E_obs）
    E_hidden     = 模型不可见的隐藏边，作为 ground-truth 正样本

负样本语义
──────────
    候选集 C(u) 中，(u, v) ∉ E（整个完整图中均无此边）→ 真负样本
    候选集 C(u) 中，(u, v) ∈ E_obs                      → 丢弃（已知关系）
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.split import temporal_split


# ── Step 1：正样本净化 ────────────────────────────────────────────────────────


def filter_first_time_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """保留每个有向节点对 (src, dst) 的最早一条边，删除重复交互。

    效果示例：
        CollegeMsg  59,835 → ~20,296 条
        Email-EU   332,334 → ~24,929 条

    去重后每条边都代表一次新社交关系的建立，消除"关系延续"对正样本的污染。
    """
    idx = edges.groupby(["src", "dst"])["timestamp"].idxmin()
    return edges.loc[idx].sort_values("timestamp").reset_index(drop=True)


# ── Step 3：两层图数据结构 ────────────────────────────────────────────────────


@dataclass
class TwoLayerEdgeSet:
    """两层图切分结果。

    Attributes:
        E_obs:          观测图边集，TimeAdjacency **必须只基于此构建**（严禁使用全图）
        E_hidden_val:   验证集隐藏边（ground-truth 正样本）
        E_hidden_test:  测试集隐藏边（ground-truth 正样本）
        cutoff_val:     观测图的最大时间戳（验证阶段截断点）
        cutoff_test:    观测图+验证隐藏边的最大时间戳（测试阶段截断点）
    """
    E_obs: pd.DataFrame
    E_hidden_val: pd.DataFrame
    E_hidden_test: pd.DataFrame
    cutoff_val: float
    cutoff_test: float


def temporal_mask_split(
    edges: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> TwoLayerEdgeSet:
    """按时间切分构造两层图（与现有 70/15/15 协议兼容）。

    E_obs         = 训练期边（t < t_70）
    E_hidden_val  = 验证期中，(u,v) 对在 E_obs 中**从未出现过**的边
    E_hidden_test = 测试期中，(u,v) 对在 E_obs 和 E_hidden_val 中均未出现过的边

    只有首次边才会进入 E_hidden，"关系延续"自动被排除。
    建议在调用前先执行 filter_first_time_edges。
    """
    train_edges, val_edges, test_edges = temporal_split(edges, train_ratio, val_ratio)

    E_obs = train_edges.copy()
    obs_pairs = set(zip(E_obs["src"].tolist(), E_obs["dst"].tolist()))

    val_mask = [
        (int(r.src), int(r.dst)) not in obs_pairs
        for r in val_edges.itertuples(index=False)
    ]
    E_hidden_val = val_edges[val_mask].reset_index(drop=True)

    known_pairs = obs_pairs | set(zip(E_hidden_val["src"].tolist(), E_hidden_val["dst"].tolist()))
    test_mask = [
        (int(r.src), int(r.dst)) not in known_pairs
        for r in test_edges.itertuples(index=False)
    ]
    E_hidden_test = test_edges[test_mask].reset_index(drop=True)

    cutoff_val = float(train_edges["timestamp"].max())
    cutoff_test = float(pd.concat([train_edges, val_edges])["timestamp"].max())

    return TwoLayerEdgeSet(
        E_obs=E_obs,
        E_hidden_val=E_hidden_val,
        E_hidden_test=E_hidden_test,
        cutoff_val=cutoff_val,
        cutoff_test=cutoff_test,
    )


def random_mask_split(
    edges: pd.DataFrame,
    mask_ratio_val: float = 0.15,
    mask_ratio_test: float = 0.15,
    seed: int = 42,
    min_obs_per_node: int = 3,
) -> TwoLayerEdgeSet:
    """随机 edge-mask 构造两层图（inductive 设定）。

    按节点对随机切分，每个节点在 E_obs 中至少保留 min_obs_per_node 条出边，
    确保子图提取时每个节点都有邻居。
    """
    rng = np.random.default_rng(seed)
    pairs = edges.copy().reset_index(drop=True)
    n = len(pairs)

    shuffled_idx = rng.permutation(n)
    n_val_target = int(n * mask_ratio_val)
    n_test_target = int(n * mask_ratio_test)

    # 初始每个节点的 E_obs 出度 = 其全部出边数
    obs_count: dict[int, int] = pairs["src"].value_counts().to_dict()

    split_labels = np.zeros(n, dtype=np.int8)  # 0=obs, 1=val, 2=test
    n_val, n_test = 0, 0

    for i in shuffled_idx:
        u = int(pairs.iat[i, pairs.columns.get_loc("src")])
        if n_val < n_val_target and obs_count.get(u, 0) > min_obs_per_node:
            split_labels[i] = 1
            obs_count[u] -= 1
            n_val += 1
        elif n_test < n_test_target and obs_count.get(u, 0) > min_obs_per_node:
            split_labels[i] = 2
            obs_count[u] -= 1
            n_test += 1

    E_obs = pairs[split_labels == 0].sort_values("timestamp").reset_index(drop=True)
    E_hidden_val = pairs[split_labels == 1].sort_values("timestamp").reset_index(drop=True)
    E_hidden_test = pairs[split_labels == 2].sort_values("timestamp").reset_index(drop=True)

    obs_ts = E_obs["timestamp"]
    cutoff_val = float(obs_ts.max())
    cutoff_test = float(pd.concat([E_obs, E_hidden_val])["timestamp"].max())

    return TwoLayerEdgeSet(
        E_obs=E_obs,
        E_hidden_val=E_hidden_val,
        E_hidden_test=E_hidden_test,
        cutoff_val=cutoff_val,
        cutoff_test=cutoff_test,
    )


def build_two_layer(edges: pd.DataFrame, cfg: dict) -> TwoLayerEdgeSet:
    """根据 config 分派构造策略。

    cfg 字段：
        strategy:        'temporal'（默认）| 'random'
        mask_ratio_val:  float（random 专用）
        mask_ratio_test: float（random 专用）
        seed:            int（random 专用）
    """
    strategy = cfg.get("strategy", "temporal")
    if strategy == "temporal":
        return temporal_mask_split(edges)
    elif strategy == "random":
        return random_mask_split(
            edges,
            mask_ratio_val=cfg.get("mask_ratio_val", 0.15),
            mask_ratio_test=cfg.get("mask_ratio_test", 0.15),
            seed=cfg.get("seed", 42),
        )
    raise ValueError(f"未知 edge_split 策略: {strategy!r}，支持 'temporal' | 'random'")


# ── Step 5（占位）：互惠性标签 ────────────────────────────────────────────────


def compute_reciprocity_labels(edges: pd.DataFrame) -> dict[tuple[int, int], bool]:
    """对每个有向对 (u, v)，True 表示反向边 (v, u) 也在 edges 中存在。

    用于 Step 5 的互惠性损失加权：双向边 = 强正样本，单向边 = 弱正样本。
    建议在 filter_first_time_edges 之后调用。
    """
    pair_set: set[tuple[int, int]] = set(
        zip(edges["src"].tolist(), edges["dst"].tolist())
    )
    return {(u, v): (v, u) in pair_set for u, v in pair_set}
