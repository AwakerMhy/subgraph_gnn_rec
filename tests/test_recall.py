"""tests/test_recall.py — 模拟召回模块单元测试"""
import math

import pandas as pd
import pytest

from src.graph.subgraph import TimeAdjacency
from src.recall import AdamicAdarRecall, CommonNeighborsRecall, build_recall


# ── 辅助 ─────────────────────────────────────────────────────────────────────

def _make_time_adj(rows: list[tuple[int, int, float]]) -> TimeAdjacency:
    df = pd.DataFrame(rows, columns=["src", "dst", "timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return TimeAdjacency(df)


# ── 小图定义（手工可验证）────────────────────────────────────────────────────
#
#   edges（cutoff=1.0 时全可见）：
#     0→1 (0.1)   0→2 (0.2)
#     1→3 (0.3)   1→4 (0.4)
#     2→3 (0.5)   2→5 (0.6)
#
#   N_out(0) = {1, 2}
#   2-hop from 0: 1→{3,4}, 2→{3,5}  → candidates {3, 4, 5}
#   CN(0,3): z=1 (1→3) + z=2 (2→3) = 2
#   CN(0,4): z=1 only              = 1
#   CN(0,5): z=2 only              = 1
#
SMALL_EDGES = [
    (0, 1, 0.1), (0, 2, 0.2),
    (1, 3, 0.3), (1, 4, 0.4),
    (2, 3, 0.5), (2, 5, 0.6),
]


# ── CommonNeighborsRecall ────────────────────────────────────────────────────

class TestCommonNeighborsRecall:

    def _recall(self) -> CommonNeighborsRecall:
        return CommonNeighborsRecall(_make_time_adj(SMALL_EDGES), n_nodes=6)

    def test_scores_match_manual(self):
        r = self._recall()
        cands = dict(r.candidates(0, cutoff_time=1.0, top_k=10))
        assert cands[3] == pytest.approx(2.0)   # z=1 和 z=2 都经过
        assert cands[4] == pytest.approx(1.0)
        assert cands[5] == pytest.approx(1.0)

    def test_does_not_include_self(self):
        r = self._recall()
        cands_v = [v for v, _ in r.candidates(0, 1.0, 10)]
        assert 0 not in cands_v

    def test_does_not_include_direct_neighbors(self):
        r = self._recall()
        cands_v = [v for v, _ in r.candidates(0, 1.0, 10)]
        # 1 和 2 是 0 的直接 1-hop，不应出现在候选集
        assert 1 not in cands_v
        assert 2 not in cands_v

    def test_top_k_respected(self):
        r = self._recall()
        for k in [1, 2, 3]:
            cands = r.candidates(0, 1.0, k)
            assert len(cands) <= k

    def test_sorted_descending_by_score(self):
        r = self._recall()
        cands = r.candidates(0, 1.0, 10)
        scores = [s for _, s in cands]
        assert scores == sorted(scores, reverse=True)

    def test_cutoff_time_filters_future_edges(self):
        # cutoff=0.25 时只有 0→1(0.1) 和 0→2(0.2) 可见，1 的出边 0.3/0.4 不可见
        r = self._recall()
        cands = r.candidates(0, cutoff_time=0.25, top_k=10)
        # 只有 0→2(0.2) 可见（0→1 也可见），但 1 和 2 的出边不可见（0.3+）
        assert len(cands) == 0  # 2-hop 全不可见

    def test_cutoff_includes_edges_strictly_less_than(self):
        # cutoff=0.51 时 2→3(0.5) 可见，但 2→5(0.6) 不可见
        r = self._recall()
        cands = dict(r.candidates(0, cutoff_time=0.51, top_k=10))
        assert 3 in cands
        assert 5 not in cands

    def test_isolated_node_returns_empty(self):
        r = self._recall()
        cands = r.candidates(99, 1.0, 10)  # 不存在的节点
        assert cands == []

    def test_node_with_only_in_edges_has_bidir_candidates(self):
        # 节点 3 无出边，但有入边 1→3, 2→3
        # N_bidir(3)={1,2}；从 1/2 出发可到达 0,4,5 → 应有候选
        r = self._recall()
        cands_v = [v for v, _ in r.candidates(3, 1.0, 10)]
        assert len(cands_v) > 0
        assert 3 not in cands_v   # 自身不出现


# ── AdamicAdarRecall ─────────────────────────────────────────────────────────

class TestAdamicAdarRecall:

    def _recall(self) -> AdamicAdarRecall:
        return AdamicAdarRecall(_make_time_adj(SMALL_EDGES), n_nodes=6)

    def test_scores_positive(self):
        r = self._recall()
        cands = r.candidates(0, 1.0, 10)
        assert all(s > 0 for _, s in cands)

    def test_node3_has_higher_score_than_node4(self):
        # 节点 3 有 2 条路径（两个中间节点），节点 4 只有 1 条
        r = self._recall()
        cands = dict(r.candidates(0, 1.0, 10))
        assert cands[3] > cands[4]

    def test_aa_score_formula_for_node3(self):
        # z=1: N_bidir(1)={0,3,4} len=3, weight=1/log(5)
        # z=2: N_bidir(2)={0,3,5} len=3, weight=1/log(5)
        r = self._recall()
        cands = dict(r.candidates(0, 1.0, 10))
        expected = 2.0 / math.log(3 + 2)  # bidir_deg(z)=3, +2 in formula
        assert cands[3] == pytest.approx(expected, rel=1e-5)

    def test_top_k_respected(self):
        r = self._recall()
        cands = r.candidates(0, 1.0, 2)
        assert len(cands) <= 2

    def test_no_self_in_candidates(self):
        r = self._recall()
        cands_v = [v for v, _ in r.candidates(0, 1.0, 10)]
        assert 0 not in cands_v

    def test_cutoff_time_respected(self):
        r = self._recall()
        cands_before = r.candidates(0, 0.25, 10)
        assert len(cands_before) == 0  # 同 CN 分析

    def test_sorted_descending(self):
        r = self._recall()
        cands = r.candidates(0, 1.0, 10)
        scores = [s for _, s in cands]
        assert scores == sorted(scores, reverse=True)


# ── build_recall registry ────────────────────────────────────────────────────

class TestBuildRecall:

    def _ta(self) -> TimeAdjacency:
        return _make_time_adj(SMALL_EDGES)

    def test_common_neighbors(self):
        r = build_recall({"method": "common_neighbors"}, self._ta(), 6)
        assert isinstance(r, CommonNeighborsRecall)

    def test_adamic_adar(self):
        r = build_recall({"method": "adamic_adar"}, self._ta(), 6)
        assert isinstance(r, AdamicAdarRecall)

    def test_union_returns_recall_instance(self):
        r = build_recall({"method": "union"}, self._ta(), 6)
        assert isinstance(r, AdamicAdarRecall)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="未知"):
            build_recall({"method": "not_exist"}, self._ta(), 6)

    def test_default_method_is_cn(self):
        r = build_recall({}, self._ta(), 6)
        assert isinstance(r, CommonNeighborsRecall)
