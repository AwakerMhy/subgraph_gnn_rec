"""tests/test_train.py — train.py 核心单元测试"""
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.train import RecallDataset, collate_fn


def _make_minimal_edges(n: int = 10) -> pd.DataFrame:
    """构造最小时序边集。"""
    rows = []
    for i in range(n):
        rows.append({"src": i % 5, "dst": (i + 1) % 5, "timestamp": float(i)})
    return pd.DataFrame(rows)


def _make_recall_stub():
    """返回一个 stub Recall，固定返回 candidates = [(v, 0.5)]。"""
    class _StubRecall:
        def candidates(self, u, cutoff_time, top_k):
            # 返回节点 (u+1)%10 和 (u+2)%10 作为候选
            return [((u + 1) % 10, 0.9), ((u + 2) % 10, 0.5)]
    return _StubRecall()


class TestRecallDatasetWeights:
    def test_no_weights_produces_5tuples(self):
        hidden = pd.DataFrame({"src": [0], "dst": [2], "timestamp": [5.0]})
        e_obs  = {(0, 1)}
        e_all  = {(0, 1), (0, 2)}  # (0,2) ∈ e_all → positive
        recall = _make_recall_stub()

        ds = RecallDataset(
            hidden_edges=hidden,
            e_obs_pairs=e_obs,
            e_all_pairs=e_all,
            recall=recall,
            cutoff_time=5.0,
            top_k=10,
            n_nodes=10,
            rng_seed=0,
        )
        for sample in ds.samples:
            assert len(sample) == 5, "无权重时应为 5-tuple"

    def test_with_weights_produces_6tuples(self):
        hidden = pd.DataFrame({"src": [0], "dst": [2], "timestamp": [5.0]})
        e_obs  = {(0, 1)}
        e_all  = {(0, 1), (0, 2)}
        recall = _make_recall_stub()
        recip_weights = {(0, 2): 2.0, (0, 3): 1.0}

        ds = RecallDataset(
            hidden_edges=hidden,
            e_obs_pairs=e_obs,
            e_all_pairs=e_all,
            recall=recall,
            cutoff_time=5.0,
            top_k=10,
            n_nodes=10,
            rng_seed=0,
            reciprocity_weights=recip_weights,
        )
        for sample in ds.samples:
            assert len(sample) == 6, "有权重时应为 6-tuple"

    def test_positive_sample_uses_recip_weight(self):
        hidden = pd.DataFrame({"src": [0], "dst": [2], "timestamp": [5.0]})
        e_obs  = {(0, 1)}
        e_all  = {(0, 1), (0, 2)}
        recall = _make_recall_stub()
        recip_weights = {(0, 2): 3.0}

        ds = RecallDataset(
            hidden_edges=hidden,
            e_obs_pairs=e_obs,
            e_all_pairs=e_all,
            recall=recall,
            cutoff_time=5.0,
            top_k=10,
            n_nodes=10,
            rng_seed=0,
            reciprocity_weights=recip_weights,
        )
        pos_samples = [s for s in ds.samples if s[3] == 1]
        assert len(pos_samples) > 0
        # positive (0→2) should have weight 3.0
        for s in pos_samples:
            if s[0] == 0 and s[1] == 2:
                assert s[5] == pytest.approx(3.0)

    def test_negative_sample_weight_is_1(self):
        hidden = pd.DataFrame({"src": [0], "dst": [2], "timestamp": [5.0]})
        e_obs  = {(0, 1)}
        e_all  = {(0, 1), (0, 2)}
        recall = _make_recall_stub()
        recip_weights = {(0, 2): 3.0}

        ds = RecallDataset(
            hidden_edges=hidden,
            e_obs_pairs=e_obs,
            e_all_pairs=e_all,
            recall=recall,
            cutoff_time=5.0,
            top_k=10,
            n_nodes=10,
            rng_seed=0,
            reciprocity_weights=recip_weights,
        )
        neg_samples = [s for s in ds.samples if s[3] == 0]
        for s in neg_samples:
            assert s[5] == pytest.approx(1.0)


class TestCollateFnWeights:
    def test_no_weight_sample_returns_none_weights(self):
        """5-tuple 样本 → collate_fn 返回 weights=None。"""
        batch = [(0, 1, 1.0, 1, 0), (0, 2, 1.0, 0, 0)]
        edges = _make_minimal_edges()
        bg, labels, qids, weights = collate_fn(
            batch, edges, max_hop=1, max_neighbors=5, seed=0
        )
        assert weights is None

    def test_with_weight_sample_returns_tensor(self):
        """6-tuple 样本 → collate_fn 返回 weights Tensor。"""
        batch = [(0, 1, 1.0, 1, 0, 2.0), (0, 2, 1.0, 0, 0, 1.0)]
        edges = _make_minimal_edges()
        bg, labels, qids, weights = collate_fn(
            batch, edges, max_hop=1, max_neighbors=5, seed=0
        )
        if weights is not None:
            assert isinstance(weights, torch.Tensor)
            assert weights.shape[0] == labels.shape[0]
