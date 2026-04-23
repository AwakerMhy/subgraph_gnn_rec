"""tests/test_heuristic.py — heuristic baseline 单元测试

使用手工构造的小图验证 CN / AA / Jaccard / Katz 的正确性。

图结构（无向邻居视角）：
    0 → 1, 0 → 2, 1 → 3, 2 → 3, 4 → 3
  N(0) = {1, 2}, N(3) = {1, 2, 4}
  N(0) ∩ N(3) = {1, 2}  → CN = 2

时间戳均为 0.5，cutoff_time=1.0（全部包含）
"""
import math
import sys
sys.path.insert(0, ".")

import pandas as pd
import pytest

from src.baseline.heuristic import (
    score_cn,
    score_aa,
    score_jaccard,
    score_katz,
    batch_score,
)

# 构造小图边表
EDGES = pd.DataFrame({
    "src":       [0, 0, 1, 2, 4],
    "dst":       [1, 2, 3, 3, 3],
    "timestamp": [0.5, 0.5, 0.5, 0.5, 0.5],
})
T = 1.0  # cutoff_time，所有边均在截断前


# ── CN ────────────────────────────────────────────────────────────────────────

def test_cn_basic():
    # N(0)={1,2}, N(3)={0↑无, 1,2,4} → 公共={1,2}
    assert score_cn(0, 3, T, EDGES) == 2.0


def test_cn_no_common():
    # N(1)={0,3} (出邻 1,3 + 入邻 0) → N(4)={3}; 公共={3}
    # 实际：N(1)出={3}, N(1)入={0} → {0,3}; N(4)出={3}, N(4)入={} → {3}; 公共={3}
    assert score_cn(1, 4, T, EDGES) == 1.0


def test_cn_cutoff():
    # cutoff=0.3，所有边 timestamp=0.5 > 0.3 → 截断后无边 → CN=0
    assert score_cn(0, 3, 0.3, EDGES) == 0.0


# ── AA ────────────────────────────────────────────────────────────────────────

def test_aa_basic():
    # 公共节点 {1, 2}
    # N(1)={0,3} → deg=2; N(2)={0,3} → deg=2
    # AA = 1/log(2) + 1/log(2) = 2/log(2)
    expected = 2.0 / math.log(2)
    assert abs(score_aa(0, 3, T, EDGES) - expected) < 1e-9


# ── Jaccard ───────────────────────────────────────────────────────────────────

def test_jaccard_basic():
    # N(0)={1,2}, N(3)={1,2,4}（3 无出边，入邻={1,2,4}）
    # |交|=2, |并|=3  → Jaccard=2/3
    assert abs(score_jaccard(0, 3, T, EDGES) - 2.0 / 3.0) < 1e-9


def test_jaccard_no_neighbors():
    # 孤立节点对（节点99不存在）
    assert score_jaccard(99, 100, T, EDGES) == 0.0


# ── Katz ──────────────────────────────────────────────────────────────────────

def test_katz_basic():
    beta = 0.005
    # (0,3) 无直接边 → p1=0; out(0)={1,2}, in(3)={1,2,4} → 交={1,2} → p2=2
    expected = beta * 0 + beta * beta * 2
    assert abs(score_katz(0, 3, T, EDGES, beta=beta) - expected) < 1e-12


def test_katz_direct_edge():
    beta = 0.005
    # (0,1) 有直接边 → p1=1; out(0)={1,2}, in(1)={0} → 交={} → p2=0
    expected = beta * 1.0
    assert abs(score_katz(0, 1, T, EDGES, beta=beta) - expected) < 1e-12


# ── batch_score ───────────────────────────────────────────────────────────────

def test_batch_score_shapes():
    pairs = [(0, 3, T), (0, 1, T), (99, 100, T)]
    for method in ("cn", "aa", "jaccard", "katz"):
        scores = batch_score(pairs, EDGES, method=method)
        assert scores.shape == (3,), f"{method} shape wrong"
        assert scores.dtype.kind == "f"


def test_batch_score_unknown_method():
    with pytest.raises(AssertionError):
        batch_score([(0, 1, T)], EDGES, method="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
