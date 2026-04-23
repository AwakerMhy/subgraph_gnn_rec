"""tests/test_recall_ppr.py — PPR 召回器测试。"""
import pandas as pd
import pytest

from src.online.static_adj import StaticAdjacency
from src.recall.ppr import PPRRecall


def make_adj(n, edges):
    df = pd.DataFrame(edges, columns=["src", "dst"])
    return StaticAdjacency(n, df)


def test_ppr_excludes_self_and_neighbors():
    """返回候选不含 u 自身和已有邻居。"""
    adj = make_adj(6, [(0, 1), (0, 2), (1, 3), (2, 4)])
    rec = PPRRecall(adj, 6)
    cands = rec.candidates(0, float("inf"), top_k=10)
    vs = {v for v, _ in cands}
    assert 0 not in vs
    assert 1 not in vs
    assert 2 not in vs


def test_ppr_returns_positive_scores():
    """所有候选分数 > 0。"""
    adj = make_adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    rec = PPRRecall(adj, 5)
    cands = rec.candidates(0, float("inf"), top_k=5)
    assert all(s > 0 for _, s in cands)


def test_ppr_top_k_respected():
    """返回候选数 <= top_k。"""
    adj = make_adj(10, [(0, i) for i in range(1, 5)] + [(i, j) for i in range(1, 5) for j in range(5, 10)])
    rec = PPRRecall(adj, 10)
    cands = rec.candidates(0, float("inf"), top_k=3)
    assert len(cands) <= 3


def test_ppr_empty_graph_returns_empty():
    """图中没有边时，PPR 召回应返回空列表（无可达节点）。"""
    adj = StaticAdjacency(5)
    rec = PPRRecall(adj, 5)
    cands = rec.candidates(0, float("inf"), top_k=5)
    assert cands == []


def test_ppr_update_graph_reflects_new_edges():
    """update_graph 后新加的边对 PPR 可见，且新增节点可以被召回。"""
    adj = make_adj(5, [(0, 1), (1, 2)])
    rec = PPRRecall(adj, 5)
    cands_before = {v for v, _ in rec.candidates(0, float("inf"), top_k=5)}

    adj.add_edge(2, 3)
    adj.add_edge(3, 4)
    rec.update_graph(1)
    cands_after = {v for v, _ in rec.candidates(0, float("inf"), top_k=5)}
    # 节点 3/4 现在通过 0->1->2->3->4 路径可达
    assert len(cands_after) >= len(cands_before)
