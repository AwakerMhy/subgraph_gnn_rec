"""src/graph/negative_sampling.py — 负样本采样策略

策略：
- random:     随机采样不存在的有向边目标节点
- degree:     按节点出度分布加权采样目标节点
- hard_2hop:  从候选节点的二跳可达节点中采，且当前无边
- historical: Poursafaei 2022 — v' 来自 u 在 cutoff_time 前曾连接过的节点（历史交互对象）
              考验模型能否区分"哪段历史关系会续期"；需传入 time_adj
- inductive:  Poursafaei 2022 — v' 是训练期从未出现过的节点（完全新实体）
              考验模型对未见实体的泛化能力；需传入 inductive_pool

参考：Poursafaei et al. "Towards Better Evaluation for Dynamic Link Prediction" NeurIPS 2022
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.graph.subgraph import TimeAdjacency


def _existing_targets(u: int, adj_out: dict[int, list[int]]) -> set[int]:
    """返回 u 已有出边的目标节点集合。"""
    return set(adj_out.get(u, []))


def build_adj_out(
    edges: pd.DataFrame,
    cutoff_time: float | None = None,
) -> tuple[dict[int, list[int]], np.ndarray | None]:
    """从边表预构建出边邻接表，供 sample_negatives 复用。

    Args:
        edges:        边表（含 src / dst / timestamp 列）
        cutoff_time:  若指定，只包含 timestamp < cutoff_time 的边；None 则全量

    Returns:
        (adj_out, out_degree_or_None)
        adj_out: {src: [dst, ...]}
        out_degree: None（调用方可自行从 adj_out 派生）
    """
    if cutoff_time is not None:
        edges = edges[edges["timestamp"] < cutoff_time]
    adj: dict[int, list[int]] = {}
    for src_arr, dst_arr in zip(edges["src"].to_numpy(), edges["dst"].to_numpy()):
        s, d = int(src_arr), int(dst_arr)
        adj.setdefault(s, []).append(d)
    return adj, None


def sample_negatives(
    u: int,
    cutoff_time: float,
    edges: pd.DataFrame,
    n_nodes: int,
    strategy: str = "random",
    k: int = 1,
    seed: int = 42,
    prebuilt_adj_out: dict[int, list[int]] | None = None,
    all_time_adj_out: dict[int, list[int]] | None = None,
    time_adj: "TimeAdjacency | None" = None,
    inductive_pool: list[int] | None = None,
) -> list[int]:
    """为候选节点 u 在截断时刻 cutoff_time 前的图中采样 k 个负样本目标节点。

    Args:
        u:                候选节点（查询方）
        cutoff_time:      截断时刻（严格 t < cutoff_time）
        edges:            完整边列表（当 prebuilt_adj_out 为 None 时才扫描）
        n_nodes:          图中节点总数
        strategy:         采样策略（random / degree / hard_2hop / historical / inductive）
        k:                负样本数量
        seed:             随机种子
        prebuilt_adj_out: 截断时刻前的出边邻接表，用于 hard_2hop
        all_time_adj_out: 全时段出边邻接表；排除集，防止假负样本
        time_adj:         TimeAdjacency 实例；historical 策略必需
        inductive_pool:   训练期从未出现的节点列表；inductive 策略必需

    Returns:
        负样本目标节点列表，长度为 k（可能少于 k，若候选不足）。
    """
    assert strategy in ("random", "degree", "hard_2hop", "historical", "inductive"), \
        f"不支持的策略：{strategy}"

    rng = np.random.default_rng(seed)

    # 排除集：优先用全时段邻接表，防止未来真实边被标为负样本
    if all_time_adj_out is not None:
        existing = _existing_targets(u, all_time_adj_out)
    elif prebuilt_adj_out is not None:
        existing = _existing_targets(u, prebuilt_adj_out)
    else:
        edges_t = edges[edges["timestamp"] < cutoff_time]
        adj_out = {}
        for s_val, d_val in zip(edges_t["src"].to_numpy(), edges_t["dst"].to_numpy()):
            s, d = int(s_val), int(d_val)
            adj_out.setdefault(s, []).append(d)
        existing = _existing_targets(u, adj_out)

    existing.add(u)  # 排除自环

    # hard_2hop 需要截断图结构；优先级：prebuilt_adj_out > time_adj > 兜底
    if strategy == "hard_2hop":
        hop_adj = prebuilt_adj_out  # None 时在分支内按优先级处理
    else:
        hop_adj = None

    if strategy == "random":
        # 拒绝采样：稀疏图中 |existing| << n_nodes，期望尝试次数极少
        results: list[int] = []
        seen = set(existing)
        attempts = 0
        max_attempts = k * 20
        while len(results) < k and attempts < max_attempts:
            v = int(rng.integers(0, n_nodes))
            if v not in seen:
                results.append(v)
                seen.add(v)
            attempts += 1
        return results

    # degree / hard_2hop 需要候选列表（相对少用，代价可接受）
    all_nodes = set(range(n_nodes))
    candidates = list(all_nodes - existing)

    if not candidates:
        return []

    if strategy == "degree":
        # 按出度分布加权（+1 平滑，避免全零）；用全时段表或截断表均可，这里用排除集对应的表
        deg_adj = all_time_adj_out if all_time_adj_out is not None else (
            prebuilt_adj_out if prebuilt_adj_out is not None else {}
        )
        out_degree = np.array([len(deg_adj.get(c, [])) for c in candidates], dtype=np.float64)
        out_degree += 1.0
        probs = out_degree / out_degree.sum()
        chosen = rng.choice(candidates, size=min(k, len(candidates)), replace=False, p=probs)
        return list(chosen)

    elif strategy == "hard_2hop":
        # 二跳可达节点（严格使用截断图 t < cutoff_time）
        two_hop: set[int] = set()
        if hop_adj is not None:
            # prebuilt_adj_out 路径（dict，已按 cutoff 过滤）
            for nb in hop_adj.get(u, []):
                two_hop.update(hop_adj.get(nb, []))
        elif time_adj is not None:
            # TimeAdjacency 精确路径：O(log degree) 每节点，无结构泄露
            for nb in time_adj.out_neighbors(u, cutoff_time):
                two_hop.update(time_adj.out_neighbors(nb, cutoff_time))
        else:
            # 兜底：全时段表（有轻度结构泄露，仅在无截断信息时使用）
            fallback = all_time_adj_out or {}
            for nb in fallback.get(u, []):
                two_hop.update(fallback.get(nb, []))
        two_hop -= existing

        if len(two_hop) >= k:
            return list(rng.choice(sorted(two_hop), size=k, replace=False))
        else:
            result = list(two_hop)
            remaining = k - len(result)
            fallback_pool = list(set(candidates) - two_hop)
            if fallback_pool:
                extra = rng.choice(
                    fallback_pool,
                    size=min(remaining, len(fallback_pool)),
                    replace=False,
                )
                result.extend(extra.tolist())
            return result

    elif strategy == "historical":
        # Poursafaei 2022 Historical：v' 是 u 在 cutoff_time 之前曾经连接过的节点
        # 仅排除自环；不排除 all_time_adj_out（否则历史邻居被全部清空，退化为 random）
        # 历史邻居中部分可能是未来正样本，但这是该策略的设计意图（测试"历史续期识别"）
        assert time_adj is not None, "historical 策略需要传入 time_adj"
        hist_pool = [
            v for v in time_adj.out_neighbors(u, cutoff_time)
            if v != u
        ]
        if len(hist_pool) == 0:
            # u 在 cutoff_time 前无历史出边，回退到 random
            results: list[int] = []
            seen = set(existing)
            attempts, max_attempts = 0, k * 20
            while len(results) < k and attempts < max_attempts:
                v = int(rng.integers(0, n_nodes))
                if v not in seen:
                    results.append(v)
                    seen.add(v)
                attempts += 1
            return results
        chosen = rng.choice(
            hist_pool,
            size=min(k, len(hist_pool)),
            replace=False,
        )
        return list(chosen)

    elif strategy == "inductive":
        # Poursafaei 2022 Inductive：v' 是训练期从未出现过的节点（完全新实体）
        assert inductive_pool is not None, "inductive 策略需要传入 inductive_pool"
        ind_candidates = [v for v in inductive_pool if v not in existing]
        if len(ind_candidates) == 0:
            # 无新实体，回退到 random
            results = []
            seen = set(existing)
            attempts, max_attempts = 0, k * 20
            while len(results) < k and attempts < max_attempts:
                v = int(rng.integers(0, n_nodes))
                if v not in seen:
                    results.append(v)
                    seen.add(v)
                attempts += 1
            return results
        chosen = rng.choice(
            ind_candidates,
            size=min(k, len(ind_candidates)),
            replace=False,
        )
        return list(chosen)

    return []


def sample_negatives_mixed(
    u: int,
    cutoff_time: float,
    edges: pd.DataFrame,
    n_nodes: int,
    strategy_mix: dict[str, float],
    k: int = 1,
    seed: int = 42,
    prebuilt_adj_out: dict[int, list[int]] | None = None,
    all_time_adj_out: dict[int, list[int]] | None = None,
    time_adj: "TimeAdjacency | None" = None,
    inductive_pool: list[int] | None = None,
) -> list[int]:
    """按 strategy_mix 各策略权重比例采样 k 个负样本。

    strategy_mix: {strategy_name: weight}，权重无需归一化。
    不同策略使用不同 seed offset，避免各策略间采样相关。
    若去重后数量不足 k，用 random 策略补足。
    """
    total_w = sum(strategy_mix.values())
    strategies = list(strategy_mix.items())

    # 计算各策略分配数量，最后一个策略承接剩余
    alloc: list[tuple[str, int]] = []
    assigned = 0
    for i, (strat, w) in enumerate(strategies):
        if i < len(strategies) - 1:
            n = max(1, round(w / total_w * k))
        else:
            n = max(0, k - assigned)
        alloc.append((strat, n))
        assigned += n

    results: list[int] = []
    seen: set[int] = set()

    for i, (strat, n) in enumerate(alloc):
        if n <= 0:
            continue
        negs = sample_negatives(
            u, cutoff_time, edges, n_nodes,
            strategy=strat, k=n,
            seed=seed + i * 1000,
            prebuilt_adj_out=prebuilt_adj_out,
            all_time_adj_out=all_time_adj_out,
            time_adj=time_adj,
            inductive_pool=inductive_pool,
        )
        for v in negs:
            if v not in seen:
                results.append(v)
                seen.add(v)

    # 去重后不足 k 时用 random 补足
    if len(results) < k:
        extra = sample_negatives(
            u, cutoff_time, edges, n_nodes,
            strategy="random", k=k - len(results),
            seed=seed + 9999,
            all_time_adj_out=all_time_adj_out,
        )
        for v in extra:
            if v not in seen:
                results.append(v)
                seen.add(v)

    return results[:k]
