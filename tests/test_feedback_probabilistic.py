"""tests/test_feedback_probabilistic.py — 概率化反馈模型测试。"""
import numpy as np
import pytest

from src.online.feedback import Feedback, FeedbackSimulator


STAR = {(0, 1), (0, 2), (1, 3)}


def test_p_pos_1_p_neg_0_is_deterministic():
    """p_pos=1, p_neg=0 时，G* 内边全接受，G* 外边全拒绝（旧行为）。"""
    sim = FeedbackSimulator(STAR, p_pos=1.0, p_neg=0.0)
    recs = {0: [1, 2, 3], 1: [3, 0]}
    fb = sim.simulate(recs)
    assert set(fb.accepted) == {(0, 1), (0, 2), (1, 3)}
    assert set(fb.rejected) == {(0, 3), (1, 0)}


def test_p_accept_backward_compat():
    """p_accept 参数向后兼容：等效于 p_pos。"""
    sim = FeedbackSimulator(STAR, p_accept=1.0, p_neg=0.0)
    recs = {0: [1, 2], 1: [3]}
    fb = sim.simulate(recs)
    assert set(fb.accepted) == {(0, 1), (0, 2), (1, 3)}


def test_p_neg_allows_non_star_acceptance():
    """p_neg > 0 时，G* 外的边以 p_neg 概率被接受（探索性连接）。"""
    rng = np.random.default_rng(0)
    sim = FeedbackSimulator(STAR, p_pos=1.0, p_neg=1.0, rng=rng)
    recs = {0: [3, 4], 1: [0]}  # 全不在 G*
    fb = sim.simulate(recs)
    # p_neg=1.0 时所有边都被接受
    assert len(fb.accepted) == 3
    assert len(fb.rejected) == 0


def test_p_pos_probabilistic():
    """p_pos=0.8 时，G* 内边的接受率在统计上约为 0.8。"""
    rng = np.random.default_rng(42)
    sim = FeedbackSimulator(STAR, p_pos=0.8, p_neg=0.0, rng=rng)
    n_trials = 2000
    accepted_count = 0
    for _ in range(n_trials):
        fb = sim.simulate({0: [1]})  # (0,1) ∈ G*
        accepted_count += len(fb.accepted)
    rate = accepted_count / n_trials
    assert 0.73 < rate < 0.87, f"Expected ~0.8, got {rate:.3f}"


def test_p_neg_probabilistic():
    """p_neg=0.02 时，G* 外边的接受率在统计上约为 0.02。"""
    rng = np.random.default_rng(42)
    sim = FeedbackSimulator(STAR, p_pos=1.0, p_neg=0.02, rng=rng)
    n_trials = 5000
    accepted_count = 0
    for _ in range(n_trials):
        fb = sim.simulate({0: [3]})  # (0,3) ∉ G*
        accepted_count += len(fb.accepted)
    rate = accepted_count / n_trials
    assert 0.01 < rate < 0.04, f"Expected ~0.02, got {rate:.3f}"


def test_feedback_completeness():
    """每对 (u,v) 必须恰好出现在 accepted 或 rejected 其中一个。"""
    rng = np.random.default_rng(7)
    sim = FeedbackSimulator(STAR, p_pos=0.8, p_neg=0.02, rng=rng)
    recs = {0: [1, 2, 3, 4], 1: [0, 3]}
    fb = sim.simulate(recs)
    all_pairs = set(fb.accepted) | set(fb.rejected)
    expected = {(u, v) for u, vs in recs.items() for v in vs}
    assert all_pairs == expected
    assert len(set(fb.accepted) & set(fb.rejected)) == 0
