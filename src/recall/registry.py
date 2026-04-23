"""src/recall/registry.py — 召回器工厂"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.recall.base import RecallBase
from src.recall.heuristic import AdamicAdarRecall, CommonNeighborsRecall

if TYPE_CHECKING:
    from src.graph.subgraph import TimeAdjacency


def build_recall(
    cfg_recall: dict,
    time_adj: "TimeAdjacency",
    n_nodes: int,
) -> RecallBase:
    """根据 config 构造召回器实例。

    method 可选值：
        'common_neighbors' | 'adamic_adar' | 'union' |
        'ppr' | 'community_random' | 'mixture'

    mixture 需要额外字段 components: [{name, top_k, ...}, ...]
    """
    method = cfg_recall.get("method", "common_neighbors")

    if method == "common_neighbors":
        return CommonNeighborsRecall(time_adj, n_nodes)
    elif method == "adamic_adar":
        return AdamicAdarRecall(time_adj, n_nodes)
    elif method == "union":
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
        for comp in components_cfg:
            sub = build_recall({**comp, "method": comp["name"]}, time_adj, n_nodes)
            components.append((sub, comp.get("top_k", 10)))
        return MixtureRecall(components)
    else:
        raise ValueError(
            f"未知召回策略: {method!r}，支持 'common_neighbors' | 'adamic_adar' | "
            f"'ppr' | 'community_random' | 'mixture'"
        )
