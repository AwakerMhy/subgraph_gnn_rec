"""src/recall/registry.py — 召回器工厂"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.recall.base import RecallBase
from src.recall.heuristic import AdamicAdarRecall, CommonNeighborsRecall, TwoHopRandomRecall

if TYPE_CHECKING:
    from src.graph.subgraph import TimeAdjacency


def build_recall(
    cfg_recall: dict,
    time_adj: "TimeAdjacency",
    n_nodes: int,
) -> RecallBase:
    """根据 config 构造召回器实例。

    method 可选值：
        'common_neighbors' | 'two_hop_random' | 'adamic_adar' |
        'ppr' | 'community_random' | 'mixture'

    mixture 需要额外字段 components: [{name, top_k, ...}, ...]
    旧的 'union' 别名已废弃（DECISIONS.md [2026-04-21]），请改用 mixture。
    """
    method = cfg_recall.get("method", "common_neighbors")

    if method == "common_neighbors":
        return CommonNeighborsRecall(time_adj, n_nodes)
    elif method == "two_hop_random":
        return TwoHopRandomRecall(time_adj, n_nodes, seed=cfg_recall.get("seed", 42))
    elif method == "adamic_adar":
        return AdamicAdarRecall(time_adj, n_nodes)
    elif method == "ppr":
        from src.recall.ppr import PPRRecall  # noqa: PLC0415
        return PPRRecall(
            time_adj, n_nodes,
            alpha=cfg_recall.get("alpha", 0.15),
            max_iter=cfg_recall.get("max_iter", 20),
        )
    elif method == "community_random":
        from src.recall.community import CommunityRandomRecall  # noqa: PLC0415
        return CommunityRandomRecall(
            time_adj, n_nodes,
            recompute_every_n=cfg_recall.get("recompute_every_n", 20),
            seed=cfg_recall.get("seed", 42),
        )
    elif method == "mixture":
        from src.recall.mixture import MixtureRecall  # noqa: PLC0415
        components_cfg = cfg_recall.get("components", [
            {"name": "adamic_adar", "top_k": 30},
            {"name": "ppr",         "top_k": 10, "alpha": 0.15},
            {"name": "community_random", "top_k": 10},
        ])
        components = []
        for i, comp in enumerate(components_cfg):
            if not isinstance(comp, dict) or "name" not in comp:
                raise ValueError(
                    f"recall.components[{i}] 必须为 dict 且包含 'name' 字段，得到 {comp!r}"
                )
            if "top_k" not in comp:
                raise ValueError(
                    f"recall.components[{i}] (name={comp['name']!r}) 缺少 'top_k' 字段"
                )
            sub = build_recall({**comp, "method": comp["name"]}, time_adj, n_nodes)
            components.append((sub, int(comp["top_k"])))
        return MixtureRecall(components)
    elif method == "union":
        raise ValueError(
            "recall.method='union' 已废弃，请改用 'mixture' 并显式指定 components "
            "(参考 DECISIONS.md [2026-04-21])。"
        )
    else:
        raise ValueError(
            f"未知召回策略: {method!r}，支持 'common_neighbors' | 'adamic_adar' | "
            f"'ppr' | 'community_random' | 'mixture'"
        )
