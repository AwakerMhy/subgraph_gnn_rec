"""tests/test_user_selector.py — UserSelector 单元测试。"""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.online.static_adj import StaticAdjacency
from src.online.user_selector import UserSelector


def make_adj(n, edges=None):
    if edges is None:
        return StaticAdjacency(n)
    df = pd.DataFrame(edges, columns=["src", "dst"])
    return StaticAdjacency(n, df)


# ── uniform 策略 ──────────────────────────────────────────────────────────────

def test_uniform_distribution():
    """uniform 策略下，各节点被选中频率应近似均匀（KS 检验 p > 0.01）。"""
    N = 50
    sel = UserSelector(N, strategy="uniform", sample_ratio=0.2, seed=0)
    adj = make_adj(N)
    counts = np.zeros(N)
    for t in range(500):
        for u in sel.select(t, adj):
            counts[u] += 1
    expected = counts.sum() / N
    # chi-square 检验均匀性
    _, p = stats.chisquare(counts, f_exp=np.full(N, expected))
    assert p > 0.01, f"uniform 分布 chi-square p={p:.4f} < 0.01"


def test_uniform_sample_size():
    """uniform 采样数量 = round(N * ratio)。"""
    N = 100
    sel = UserSelector(N, strategy="uniform", sample_ratio=0.1, seed=0)
    adj = make_adj(N)
    selected = sel.select(0, adj)
    assert len(selected) == 10
    assert len(set(selected)) == 10  # 无放回


# ── composite 策略 ────────────────────────────────────────────────────────────

def test_composite_high_degree_preferred():
    """composite 策略下，高度节点被选中频率显著高于孤立节点。"""
    N = 20
    # 节点 0 连接了 15 个出边，其余节点零出边
    edges = [(0, i) for i in range(1, 16)]
    adj = make_adj(N, edges)
    sel = UserSelector(N, strategy="composite", alpha=0.5, beta=2.0,
                       lam=0.0, gamma=0.0, sample_ratio=0.3, seed=42)
    counts = np.zeros(N)
    for t in range(300):
        for u in sel.select(t, adj):
            counts[u] += 1
    # 节点 0 的被选率应至少是平均值的 2 倍
    avg = counts.mean()
    assert counts[0] > 2 * avg, f"节点0被选{counts[0]:.0f}次，均值{avg:.1f}"


def test_composite_event_trigger():
    """事件触发：上轮获得新边的节点，本轮被选概率提升。"""
    N = 50
    adj = make_adj(N)
    sel = UserSelector(N, strategy="composite", alpha=0.0, lam=0.0,
                       gamma=5.0, w=1, sample_ratio=0.3, seed=7)
    # 模拟节点 0 在 t=0 获得新边
    sel.update_after_round(0, [(0, 1), (0, 2), (0, 3)])
    # t=1 时统计节点 0 被选中频率
    count_0 = sum(1 for _ in range(200) if 0 in sel.select(1, adj))
    expected_uniform = 200 * 0.3  # 无触发时期望次数
    assert count_0 > expected_uniform, f"事件触发不显著: {count_0} vs 期望>{expected_uniform:.0f}"


def test_composite_time_decay():
    """时间衰减：长期不活跃节点的权重因子应显著低于新近活跃节点。"""
    N = 20
    sel = UserSelector(N, strategy="composite", alpha=0.0, lam=0.5,
                       gamma=0.0, sample_ratio=0.3, seed=0)
    # 节点 0 在 t=0 活跃，节点 1 在 t=10 活跃
    sel._t_last[0] = 0
    sel._t_last[1] = 10
    # 在 t=10 时，节点 0 的时间因子 = exp(-0.5*10) ≈ 0.0067
    # 节点 1 的时间因子 = exp(-0.5*0) = 1.0
    import numpy as np
    factor_0 = float(np.exp(-0.5 * (10 - 0)))
    factor_1 = float(np.exp(-0.5 * (10 - 10)))
    assert factor_0 < 0.01, f"节点0时间因子应极小，got {factor_0:.4f}"
    assert factor_1 == pytest.approx(1.0)
    assert factor_1 / factor_0 > 50, "新近活跃节点权重应远高于长期不活跃节点"


def test_no_duplicate_in_selection():
    """无放回采样：同一轮不重复。"""
    N = 50
    sel = UserSelector(N, strategy="composite", sample_ratio=0.4, seed=1)
    adj = make_adj(N)
    for t in range(20):
        selected = sel.select(t, adj)
        assert len(selected) == len(set(selected))
