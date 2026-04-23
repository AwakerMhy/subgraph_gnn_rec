"""src/baseline/heuristic.py — 启发式链接预测 baseline

实现 CN / AA / Jaccard / Katz（截断到二阶路径）。
均为 transductive，使用截断时刻 cutoff_time 前的历史边（严格 t < cutoff_time）。

所有函数签名：
    score_*(u, v, cutoff_time, edges) -> float
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd


# ── 内部工具 ──────────────────────────────────────────────────────────────────

def _build_neighbor_sets(
    edges_t: pd.DataFrame,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """构建出边邻居集和入边邻居集。"""
    out_nbrs: dict[int, set[int]] = defaultdict(set)
    in_nbrs: dict[int, set[int]] = defaultdict(set)
    for _, row in edges_t.iterrows():
        u, v = int(row["src"]), int(row["dst"])
        out_nbrs[u].add(v)
        in_nbrs[v].add(u)
    return dict(out_nbrs), dict(in_nbrs)


def _undirected_nbrs(
    node: int,
    out_nbrs: dict[int, set[int]],
    in_nbrs: dict[int, set[int]],
) -> set[int]:
    """无向邻居集（出邻居 ∪ 入邻居），排除自身。"""
    return (out_nbrs.get(node, set()) | in_nbrs.get(node, set())) - {node}


def _cutoff(edges: pd.DataFrame, cutoff_time: float) -> pd.DataFrame:
    """严格截断：只保留 timestamp < cutoff_time 的历史边。"""
    result = edges[edges["timestamp"] < cutoff_time]
    if len(result) > 0:
        assert result["timestamp"].max() < cutoff_time, "时间泄露"
    return result


# ── CN：Common Neighbors ──────────────────────────────────────────────────────

def score_cn(
    u: int,
    v: int,
    cutoff_time: float,
    edges: pd.DataFrame,
) -> float:
    """Common Neighbors：|N(u) ∩ N(v)|（无向邻居）。"""
    edges_t = _cutoff(edges, cutoff_time)
    out_nbrs, in_nbrs = _build_neighbor_sets(edges_t)
    nu = _undirected_nbrs(u, out_nbrs, in_nbrs)
    nv = _undirected_nbrs(v, out_nbrs, in_nbrs)
    return float(len(nu & nv))


# ── AA：Adamic-Adar ────────────────────────────────────────────────────────────

def score_aa(
    u: int,
    v: int,
    cutoff_time: float,
    edges: pd.DataFrame,
) -> float:
    """Adamic-Adar：∑_{w ∈ N(u)∩N(v)} 1 / log(|N(w)|)。"""
    edges_t = _cutoff(edges, cutoff_time)
    out_nbrs, in_nbrs = _build_neighbor_sets(edges_t)
    nu = _undirected_nbrs(u, out_nbrs, in_nbrs)
    nv = _undirected_nbrs(v, out_nbrs, in_nbrs)
    common = nu & nv
    score = 0.0
    for w in common:
        deg_w = len(_undirected_nbrs(w, out_nbrs, in_nbrs))
        if deg_w > 1:
            score += 1.0 / math.log(deg_w)
    return score


# ── Jaccard ───────────────────────────────────────────────────────────────────

def score_jaccard(
    u: int,
    v: int,
    cutoff_time: float,
    edges: pd.DataFrame,
) -> float:
    """Jaccard：|N(u)∩N(v)| / |N(u)∪N(v)|。"""
    edges_t = _cutoff(edges, cutoff_time)
    out_nbrs, in_nbrs = _build_neighbor_sets(edges_t)
    nu = _undirected_nbrs(u, out_nbrs, in_nbrs)
    nv = _undirected_nbrs(v, out_nbrs, in_nbrs)
    union = nu | nv
    if not union:
        return 0.0
    return float(len(nu & nv)) / len(union)


# ── Katz（截断到二阶路径）────────────────────────────────────────────────────

def score_katz(
    u: int,
    v: int,
    cutoff_time: float,
    edges: pd.DataFrame,
    beta: float = 0.005,
) -> float:
    """Katz（近似）：β * paths_1(u,v) + β² * paths_2(u,v)。

    paths_1(u,v) = 1 if (u,v) is a direct edge (in history), else 0.
    paths_2(u,v) = |N_out(u) ∩ N_in(v)| （经过中间节点的二阶路径数）。

    Args:
        beta: 衰减系数（默认 0.005，适合稀疏社交图）
    """
    edges_t = _cutoff(edges, cutoff_time)
    out_nbrs, in_nbrs = _build_neighbor_sets(edges_t)

    # 一阶：直接边
    p1 = 1.0 if v in out_nbrs.get(u, set()) else 0.0

    # 二阶：u 的出邻居 ∩ v 的入邻居
    out_u = out_nbrs.get(u, set())
    in_v = in_nbrs.get(v, set())
    p2 = float(len(out_u & in_v))

    return beta * p1 + beta * beta * p2


# ── 批量评分（供评估脚本使用）─────────────────────────────────────────────────

def batch_score(
    pairs: list[tuple[int, int, float]],
    edges: pd.DataFrame,
    method: str = "cn",
    beta: float = 0.005,
) -> np.ndarray:
    """对一批 (u, v, cutoff_time) 评分，返回 (N,) 数组。

    Args:
        pairs:  [(u, v, cutoff_time), ...]
        edges:  完整边列表（含 timestamp 列）
        method: "cn" / "aa" / "jaccard" / "katz"
        beta:   Katz 衰减系数
    """
    fn_map = {
        "cn": score_cn,
        "aa": score_aa,
        "jaccard": score_jaccard,
        "katz": lambda u, v, t, e: score_katz(u, v, t, e, beta=beta),
    }
    assert method in fn_map, f"未知方法 '{method}'，可选：{list(fn_map)}"
    fn = fn_map[method]

    scores = []
    for u, v, t in pairs:
        scores.append(fn(u, v, t, edges))
    return np.array(scores, dtype=np.float32)
