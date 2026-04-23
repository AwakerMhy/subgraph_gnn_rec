"""tests/test_metrics.py — 评估指标单元测试"""
import math

import numpy as np
import pytest

from src.utils.metrics import (
    compute_all_metrics,
    compute_ap,
    compute_auc,
    compute_hits_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_ranking_metrics,
)


class TestComputeAUC:
    def test_perfect_prediction(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert compute_auc(y_true, y_score) == pytest.approx(1.0)

    def test_random_prediction(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=1000)
        y_score = rng.uniform(0, 1, size=1000)
        auc = compute_auc(y_true, y_score)
        assert 0.4 < auc < 0.6  # 随机约为 0.5


class TestComputeAP:
    def test_perfect_prediction(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        assert compute_ap(y_true, y_score) == pytest.approx(1.0)


class TestHitsAtK:
    def test_all_hit(self):
        """正样本得分均高于所有负样本。"""
        n = 10
        pos_scores = np.ones(n) * 0.9
        neg_scores = np.zeros((n, 999)) + 0.1
        assert compute_hits_at_k(pos_scores, neg_scores, k=1) == pytest.approx(1.0)

    def test_none_hit(self):
        """正样本得分均低于所有负样本。"""
        n = 10
        pos_scores = np.ones(n) * 0.1
        neg_scores = np.zeros((n, 999)) + 0.9
        assert compute_hits_at_k(pos_scores, neg_scores, k=10) == pytest.approx(0.0)

    def test_k_boundary(self):
        """正样本排名恰好为 k 时应命中。"""
        # 正样本排名 = 比正样本高的负样本数 + 1 = k
        n = 5
        pos_scores = np.ones(n) * 0.5
        # 构造 k-1 个负样本分数 > pos_score，其余 <= pos_score
        k = 10
        neg_scores = np.zeros((n, 999)) + 0.3
        neg_scores[:, :k - 1] = 0.9  # k-1 个负样本比正样本高，排名 = k
        assert compute_hits_at_k(pos_scores, neg_scores, k=k) == pytest.approx(1.0)
        assert compute_hits_at_k(pos_scores, neg_scores, k=k - 1) == pytest.approx(0.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_hits_at_k(
                np.ones(5),
                np.ones((6, 999)),  # 行数不匹配
                k=10,
            )


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        y_true = np.array([1, 1, 0, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1])
        pos_scores = np.array([0.9, 0.8])
        neg_scores = np.random.rand(2, 999) * 0.5
        result = compute_all_metrics(y_true, y_score, pos_scores, neg_scores, k_list=[10, 20])
        assert "auc" in result
        assert "ap" in result
        assert "hits@10" in result
        assert "hits@20" in result


class TestComputeMRR:
    def test_rank1_gives_mrr_1(self):
        """正样本分数最高时，rank=1，MRR=1.0。"""
        pos = np.array([1.0])
        neg = np.zeros((1, 5)) + 0.5
        assert compute_mrr(pos, neg) == pytest.approx(1.0)

    def test_rank_last_gives_small_mrr(self):
        """正样本分数最低时，rank=M+1，MRR=1/(M+1)。"""
        M = 9
        pos = np.array([0.0])
        neg = np.ones((1, M)) * 0.9
        expected = 1.0 / (M + 1)
        assert compute_mrr(pos, neg) == pytest.approx(expected)

    def test_multiple_samples_averaged(self):
        pos = np.array([1.0, 0.0])
        neg = np.array([[0.5] * 9, [0.9] * 9])
        # sample 0: rank=1, mrr=1.0; sample 1: rank=10, mrr=1/10
        expected = (1.0 + 1.0 / 10) / 2
        assert compute_mrr(pos, neg) == pytest.approx(expected)

    def test_better_model_higher_mrr(self):
        neg = np.ones((4, 99)) * 0.5
        pos_good = np.ones(4) * 0.9
        pos_bad = np.ones(4) * 0.1
        assert compute_mrr(pos_good, neg) > compute_mrr(pos_bad, neg)


class TestComputeNDCGAtK:
    def test_rank1_gives_ndcg_1(self):
        pos = np.array([1.0])
        neg = np.zeros((1, 5)) + 0.5
        assert compute_ndcg_at_k(pos, neg, k=5) == pytest.approx(1.0)

    def test_rank_beyond_k_gives_0(self):
        """正样本排名超过 k 时 NDCG@k=0。"""
        pos = np.array([0.0])
        neg = np.ones((1, 5)) * 0.9  # rank=6 > k=5
        assert compute_ndcg_at_k(pos, neg, k=5) == pytest.approx(0.0)

    def test_rank2_formula(self):
        pos = np.array([0.7])
        neg = np.array([[0.9] + [0.1] * 9])   # 1 neg > pos → rank=2
        expected = 1.0 / math.log2(2 + 1)
        assert compute_ndcg_at_k(pos, neg, k=5) == pytest.approx(expected)

    def test_larger_k_ge_smaller_k(self):
        pos = np.array([0.5])
        neg = np.ones((1, 10)) * 0.3
        assert compute_ndcg_at_k(pos, neg, k=10) >= compute_ndcg_at_k(pos, neg, k=1)


class TestComputeRankingMetrics:
    def test_returns_required_keys(self):
        queries = {
            0: (np.array([0.9]), np.zeros((1, 9)) + 0.5),
            1: (np.array([0.8]), np.zeros((1, 9)) + 0.5),
        }
        result = compute_ranking_metrics(queries, k_list=[10])
        assert "mrr" in result
        assert "ndcg@10" in result
        assert "hits@10" in result

    def test_perfect_model_all_rank1(self):
        queries = {i: (np.array([1.0]), np.zeros((1, 9)) + 0.5) for i in range(5)}
        result = compute_ranking_metrics(queries, k_list=[1])
        assert result["mrr"] == pytest.approx(1.0)
        assert result["hits@1"] == pytest.approx(1.0)
