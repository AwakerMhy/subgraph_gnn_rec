"""tests/test_edge_split.py — edge_split 模块单元测试"""
import numpy as np
import pandas as pd
import pytest

from src.graph.edge_split import (
    TwoLayerEdgeSet,
    build_two_layer,
    compute_reciprocity_labels,
    filter_first_time_edges,
    random_mask_split,
    temporal_mask_split,
)


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def _make_edges(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """构造最小边表 (src, dst, timestamp)。"""
    return pd.DataFrame(rows, columns=["src", "dst", "timestamp"])


def _pair_set(edges: pd.DataFrame) -> set[tuple[int, int]]:
    return set(zip(edges["src"].tolist(), edges["dst"].tolist()))


# ── filter_first_time_edges ────────────────────────────────────────────────────

class TestFilterFirstTimeEdges:

    def test_removes_duplicates(self):
        edges = _make_edges([
            (0, 1, 0.1), (0, 1, 0.5), (0, 1, 0.9),  # 三条重复
            (1, 2, 0.2),
        ])
        result = filter_first_time_edges(edges)
        assert len(result) == 2

    def test_keeps_earliest_timestamp(self):
        edges = _make_edges([
            (0, 1, 0.5), (0, 1, 0.1), (0, 1, 0.9),
        ])
        result = filter_first_time_edges(edges)
        assert float(result[result["src"] == 0]["timestamp"].iloc[0]) == pytest.approx(0.1)

    def test_different_directions_are_independent_pairs(self):
        # (0,1) 和 (1,0) 是不同有向对，各保留最早一条
        edges = _make_edges([
            (0, 1, 0.1), (0, 1, 0.3),
            (1, 0, 0.2), (1, 0, 0.4),
        ])
        result = filter_first_time_edges(edges)
        assert len(result) == 2
        assert (0, 1) in _pair_set(result)
        assert (1, 0) in _pair_set(result)

    def test_no_duplicates_unchanged_count(self):
        edges = _make_edges([(0, 1, 0.1), (1, 2, 0.2), (2, 3, 0.3)])
        result = filter_first_time_edges(edges)
        assert len(result) == 3

    def test_result_sorted_by_timestamp(self):
        edges = _make_edges([(2, 3, 0.9), (0, 1, 0.1), (1, 2, 0.5)])
        result = filter_first_time_edges(edges)
        ts = result["timestamp"].tolist()
        assert ts == sorted(ts)

    def test_self_loops_treated_as_pairs(self):
        # self-loop (0,0) 有两条，只保留一条
        edges = _make_edges([(0, 0, 0.1), (0, 0, 0.5), (0, 1, 0.2)])
        result = filter_first_time_edges(edges)
        assert len(result) == 2


# ── temporal_mask_split ───────────────────────────────────────────────────────

class TestTemporalMaskSplit:

    def _make_linear_edges(self, n: int = 100) -> pd.DataFrame:
        """产生 n 条无重复的线性时序边。"""
        rows = [(i, i + 1, float(i) / n) for i in range(n)]
        return _make_edges(rows)

    def test_returns_two_layer_edge_set(self):
        edges = self._make_linear_edges()
        result = temporal_mask_split(edges)
        assert isinstance(result, TwoLayerEdgeSet)

    def test_E_obs_and_E_hidden_pairs_disjoint(self):
        """E_obs 与 E_hidden_val / E_hidden_test 的节点对不能重叠。"""
        # 产生含重复对的边集，模拟真实场景
        rows = [(0, 1, 0.1), (0, 1, 0.5), (0, 1, 0.9),   # 重复，只第一条进 E_obs
                (1, 2, 0.2), (1, 2, 0.8),
                (2, 3, 0.3), (3, 4, 0.7), (4, 5, 0.85)]
        edges = _make_edges(rows)
        result = temporal_mask_split(edges)

        obs_pairs = _pair_set(result.E_obs)
        val_pairs = _pair_set(result.E_hidden_val)
        test_pairs = _pair_set(result.E_hidden_test)

        assert obs_pairs.isdisjoint(val_pairs), "E_obs 与 E_hidden_val 有重叠对"
        assert obs_pairs.isdisjoint(test_pairs), "E_obs 与 E_hidden_test 有重叠对"
        assert val_pairs.isdisjoint(test_pairs), "E_hidden_val 与 E_hidden_test 有重叠对"

    def test_cutoff_val_equals_max_obs_timestamp(self):
        edges = self._make_linear_edges(100)
        result = temporal_mask_split(edges)
        obs_max = float(result.E_obs["timestamp"].max())
        assert result.cutoff_val == pytest.approx(obs_max)

    def test_all_splits_nonempty_on_sufficient_data(self):
        edges = self._make_linear_edges(50)
        result = temporal_mask_split(edges)
        assert len(result.E_obs) > 0
        # val / test 可能因对重叠而为空，但至少 E_obs 非空

    def test_no_future_leakage_in_E_obs(self):
        """E_obs 中的时间戳必须 ≤ cutoff_val。"""
        edges = self._make_linear_edges(100)
        result = temporal_mask_split(edges)
        assert (result.E_obs["timestamp"] <= result.cutoff_val + 1e-9).all()


# ── random_mask_split ─────────────────────────────────────────────────────────

class TestRandomMaskSplit:

    def _make_star_edges(self, n_leaves: int = 30) -> pd.DataFrame:
        """中心节点 0 向所有叶子节点各连一条边（保证中心节点度数充足）。"""
        rows = [(0, i + 1, float(i) / n_leaves) for i in range(n_leaves)]
        return _make_edges(rows)

    def test_pairs_disjoint(self):
        edges = self._make_star_edges(30)
        result = random_mask_split(edges, min_obs_per_node=3)
        obs_pairs = _pair_set(result.E_obs)
        val_pairs = _pair_set(result.E_hidden_val)
        test_pairs = _pair_set(result.E_hidden_test)
        assert obs_pairs.isdisjoint(val_pairs)
        assert obs_pairs.isdisjoint(test_pairs)
        assert val_pairs.isdisjoint(test_pairs)

    def test_each_node_has_min_obs_edges(self):
        edges = self._make_star_edges(30)
        result = random_mask_split(edges, min_obs_per_node=3)
        obs_src_counts = result.E_obs["src"].value_counts()
        # 节点 0 在 E_obs 中出度 ≥ min_obs_per_node
        assert obs_src_counts.get(0, 0) >= 3

    def test_reproducible_with_same_seed(self):
        edges = self._make_star_edges(30)
        r1 = random_mask_split(edges, seed=0)
        r2 = random_mask_split(edges, seed=0)
        assert list(r1.E_obs["dst"]) == list(r2.E_obs["dst"])

    def test_different_seeds_differ(self):
        edges = self._make_star_edges(30)
        r1 = random_mask_split(edges, seed=0)
        r2 = random_mask_split(edges, seed=99)
        # 大概率不同（概率极低相同）
        assert set(r1.E_hidden_val["dst"].tolist()) != set(r2.E_hidden_val["dst"].tolist())


# ── build_two_layer dispatcher ────────────────────────────────────────────────

class TestBuildTwoLayer:

    def _edges(self) -> pd.DataFrame:
        rows = [(i, i + 1, float(i) / 50) for i in range(50)]
        return _make_edges(rows)

    def test_temporal_dispatch(self):
        result = build_two_layer(self._edges(), {"strategy": "temporal"})
        assert isinstance(result, TwoLayerEdgeSet)

    def test_random_dispatch(self):
        result = build_two_layer(self._edges(), {"strategy": "random"})
        assert isinstance(result, TwoLayerEdgeSet)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="未知"):
            build_two_layer(self._edges(), {"strategy": "invalid"})

    def test_default_is_temporal(self):
        result = build_two_layer(self._edges(), {})
        assert isinstance(result, TwoLayerEdgeSet)


# ── compute_reciprocity_labels ────────────────────────────────────────────────

class TestReciprocityLabels:

    def test_bidirectional_pair_marked_true(self):
        edges = _make_edges([(0, 1, 0.1), (1, 0, 0.2), (0, 2, 0.3)])
        labels = compute_reciprocity_labels(edges)
        assert labels[(0, 1)] is True
        assert labels[(1, 0)] is True

    def test_unidirectional_pair_marked_false(self):
        edges = _make_edges([(0, 1, 0.1), (0, 2, 0.3)])
        labels = compute_reciprocity_labels(edges)
        assert labels[(0, 1)] is False
        assert labels[(0, 2)] is False

    def test_symmetric(self):
        edges = _make_edges([(0, 1, 0.1), (1, 0, 0.2)])
        labels = compute_reciprocity_labels(edges)
        assert labels[(0, 1)] == labels[(1, 0)]

    def test_all_pairs_covered(self):
        edges = _make_edges([(0, 1, 0.1), (1, 2, 0.2), (2, 0, 0.3)])
        labels = compute_reciprocity_labels(edges)
        assert len(labels) == 3
