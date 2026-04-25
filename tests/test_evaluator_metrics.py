"""tests/test_evaluator_metrics.py — Hit Rate@K / Coverage@K / Novelty 测试。"""
import numpy as np
import pandas as pd
import pytest

from src.online.evaluator import RoundMetrics
from src.online.feedback import Feedback
from src.online.static_adj import StaticAdjacency


def make_adj(n, edges):
    df = pd.DataFrame(edges, columns=["src", "dst"])
    return StaticAdjacency(n, df)


def make_evaluator(n=10, k_list=None, graph_every_n=100):
    star = {(0, 1), (0, 2), (1, 3), (2, 4)}
    return RoundMetrics(star, n_nodes=n, k_list=k_list or [3], graph_every_n=graph_every_n)


# ── Hit Rate@K ────────────────────────────────────────────────────────────────

def test_hit_rate_all_miss():
    """没有用户命中推荐 → hit_rate@K = 0。"""
    ev = make_evaluator()
    adj = make_adj(10, [])
    fb = Feedback(accepted=[], rejected=[(0, 5), (0, 6)], recs={0: [5, 6]})
    row = ev.update(0, {0: [5, 6]}, fb, adj, 0.0)
    assert row["hit_rate@3"] == pytest.approx(0.0)


def test_hit_rate_all_hit():
    """每个用户都有至少 1 个命中 → hit_rate@K = 1.0。"""
    ev = make_evaluator()
    adj = make_adj(10, [])
    # 用户 0 推荐 [1,2,5]，(0,1) 和 (0,2) ∈ star
    # 用户 1 推荐 [3,6]，(1,3) ∈ star
    fb = Feedback(accepted=[(0, 1), (0, 2), (1, 3)],
                  rejected=[(0, 5), (1, 6)],
                  recs={0: [1, 2, 5], 1: [3, 6]})
    row = ev.update(0, {0: [1, 2, 5], 1: [3, 6]}, fb, adj, 0.0)
    assert row["hit_rate@3"] == pytest.approx(1.0)


def test_hit_rate_partial():
    """部分命中：hit_rate = 命中用户数 / 总用户数。"""
    ev = make_evaluator()
    adj = make_adj(10, [])
    # 用户 0 命中，用户 1 未命中
    fb = Feedback(accepted=[(0, 1)],
                  rejected=[(1, 5)],
                  recs={0: [1, 5], 1: [5, 6]})
    row = ev.update(0, {0: [1, 5], 1: [5, 6]}, fb, adj, 0.0)
    assert row["hit_rate@3"] == pytest.approx(0.5)


# ── Coverage@K ────────────────────────────────────────────────────────────────

def test_rec_coverage_distinct_targets():
    """Coverage@K = top-K 不同目标节点数 / 候选池去重大小。

    分母在 [2026-04-24] 改为候选池大小（避免被 sample_ratio 稀释）。
    本例候选池与 top-K 同为 {1,2,3,4}，故 coverage = 4/4 = 1.0。
    """
    ev = make_evaluator(n=10)
    adj = make_adj(10, [])
    fb = Feedback(accepted=[], rejected=[(0, 1), (0, 2), (1, 3), (1, 4)],
                  recs={0: [1, 2], 1: [3, 4]})
    row = ev.update(0, {0: [1, 2], 1: [3, 4]}, fb, adj, 0.0)
    assert row["rec_coverage@3"] == pytest.approx(1.0)
    assert row["unique_recs@3"] == pytest.approx(4.0)


# ── Novelty ──────────────────────────────────────────────────────────────────

def test_novelty_chain_graph():
    """链状图 0-1-2-3-4 中，推荐 (0,4) 的最短路径长度 = 4。"""
    ev = make_evaluator(n=5, graph_every_n=100)
    adj = make_adj(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    fb = Feedback(accepted=[], rejected=[(0, 4)], recs={0: [4]})
    row = ev.update(0, {0: [4]}, fb, adj, 0.0)
    assert row["novelty"] == pytest.approx(4.0, abs=0.5)


def test_novelty_disconnected_returns_nan():
    """不连通图中无路径时，novelty 应为 nan。"""
    ev = make_evaluator(n=5, graph_every_n=100)
    adj = make_adj(5, [])   # 空图
    fb = Feedback(accepted=[], rejected=[(0, 4)], recs={0: [4]})
    row = ev.update(0, {0: [4]}, fb, adj, 0.0)
    assert np.isnan(row["novelty"])
