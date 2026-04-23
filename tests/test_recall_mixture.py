"""tests/test_recall_mixture.py — Mixture 召回器测试。"""
import pandas as pd
import pytest

from src.online.static_adj import StaticAdjacency
from src.recall.heuristic import AdamicAdarRecall
from src.recall.mixture import MixtureRecall
from src.recall.ppr import PPRRecall


def make_adj(n, edges):
    df = pd.DataFrame(edges, columns=["src", "dst"])
    return StaticAdjacency(n, df)


class FixedRecall:
    """返回固定候选列表的 mock 召回器，用于测试去重逻辑。"""

    def __init__(self, cands):
        self._cands = cands

    def candidates(self, u, cutoff_time, top_k):  # noqa: ARG002
        return self._cands[:top_k]

    def update_graph(self, round_idx):
        pass


def test_mixture_deduplicates():
    """两个子召回器返回重叠候选时，merged 结果无重复。"""
    r1 = FixedRecall([(1, 0.9), (2, 0.8), (3, 0.7)])
    r2 = FixedRecall([(2, 0.6), (3, 0.5), (4, 0.4)])
    rec = MixtureRecall([(r1, 3), (r2, 3)])
    cands = rec.candidates(0, float("inf"), top_k=10)
    vs = [v for v, _ in cands]
    assert len(vs) == len(set(vs)), "候选列表中不应有重复节点"


def test_mixture_quota_limits():
    """每个子召回器最多贡献其 quota 个候选。"""
    r1 = FixedRecall([(1, 1.0), (2, 0.9), (3, 0.8), (4, 0.7)])
    r2 = FixedRecall([(5, 0.6), (6, 0.5)])
    rec = MixtureRecall([(r1, 2), (r2, 2)])
    cands = rec.candidates(0, float("inf"), top_k=10)
    # r1 最多贡献 2，r2 最多贡献 2 → 最多 4 个（无重叠时）
    assert len(cands) <= 4


def test_mixture_scores_normalized():
    """合并后所有分数在 [0, 1] 范围内。"""
    r1 = FixedRecall([(1, 100.0), (2, 50.0)])
    r2 = FixedRecall([(3, 10.0), (4, 5.0)])
    rec = MixtureRecall([(r1, 2), (r2, 2)])
    cands = rec.candidates(0, float("inf"), top_k=10)
    for _, s in cands:
        assert 0.0 <= s <= 1.0


def test_mixture_empty_if_all_empty():
    """所有子召回器为空时，Mixture 返回空列表。"""
    r1 = FixedRecall([])
    r2 = FixedRecall([])
    rec = MixtureRecall([(r1, 5), (r2, 5)])
    assert rec.candidates(0, float("inf"), top_k=10) == []


def test_mixture_update_graph_forwarded():
    """update_graph 被转发到所有子召回器（通过 PPR + AA 验证不报错）。"""
    adj = make_adj(6, [(0, 1), (1, 2), (2, 3)])
    aa = AdamicAdarRecall(adj, 6)
    ppr = PPRRecall(adj, 6)
    rec = MixtureRecall([(aa, 30), (ppr, 10)])
    adj.add_edge(3, 4)
    rec.update_graph(1)  # 不应抛异常
    cands = rec.candidates(0, float("inf"), top_k=5)
    assert isinstance(cands, list)
