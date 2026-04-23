"""tests/test_cooldown_decay.py — 衰减式 Cooldown 测试。"""
import math
import pandas as pd
import pytest

from src.online.env import OnlineEnv


def make_env(mode="hard"):
    star_df = pd.DataFrame([(0, 1), (0, 2), (1, 2)], columns=["src", "dst"])
    env = OnlineEnv(star_df, n_nodes=5, init_edge_ratio=0.0,
                    p_pos=1.0, p_neg=0.0, cooldown_rounds=10, seed=0)
    env.set_cooldown_mode(mode)
    return env


# ── hard 模式 ─────────────────────────────────────────────────────────────────

def test_hard_blocks_until_unlock():
    """hard 模式：被拒绝的 (u,v) 在 cooldown_rounds 轮内从候选中消失。"""
    env = make_env("hard")
    env._cooldown[(0, 3)] = 5  # unlock_round = 5
    cands = [(3, 1.0), (4, 0.8)]
    # t=4：(0,3) 还在 cooldown
    result = env.mask_cooldown(0, cands, round_idx=4)
    vs = [v for v, _ in result]
    assert 3 not in vs
    # t=5：解锁
    result2 = env.mask_cooldown(0, cands, round_idx=5)
    vs2 = [v for v, _ in result2]
    assert 3 in vs2


# ── decay 模式 ────────────────────────────────────────────────────────────────

def test_decay_weight_at_reject_time():
    """t = t_reject 时，decay 权重 = 0（刚拒绝）。"""
    env = make_env("decay")
    env._cooldown[(0, 3)] = 5   # 拒绝时刻 = 5
    cands = [(3, 1.0)]
    result = env.mask_cooldown(0, cands, round_idx=5)
    _, s = result[0]
    assert s == pytest.approx(0.0, abs=1e-6)


def test_decay_weight_at_N_rounds():
    """t - t_reject = N 时，decay 权重 ≈ 0.632 (1 - 1/e)。"""
    N = 10
    env = make_env("decay")
    env._cooldown_rounds = N
    env._cooldown[(0, 3)] = 0   # 拒绝时刻 = 0
    cands = [(3, 1.0)]
    result = env.mask_cooldown(0, cands, round_idx=N)
    _, s = result[0]
    expected = 1.0 - math.exp(-1.0)  # ≈ 0.632
    assert s == pytest.approx(expected, abs=0.01)


def test_decay_weight_approaches_1():
    """t - t_reject >> N 时，decay 权重趋近于 1。"""
    N = 5
    env = make_env("decay")
    env._cooldown_rounds = N
    env._cooldown[(0, 3)] = 0
    cands = [(3, 1.0)]
    result = env.mask_cooldown(0, cands, round_idx=50)
    _, s = result[0]
    assert s > 0.98


def test_decay_non_rejected_pair_unchanged():
    """未被拒绝的 pair 分数不受影响（无 cooldown 记录）。"""
    env = make_env("decay")
    cands = [(4, 0.9)]
    result = env.mask_cooldown(0, cands, round_idx=10)
    assert result == [(4, 0.9)]


def test_set_cooldown_mode_invalid():
    """非法 mode 抛出 ValueError。"""
    env = make_env("hard")
    with pytest.raises(ValueError):
        env.set_cooldown_mode("soft_unknown")
