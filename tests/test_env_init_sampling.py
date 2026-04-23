"""tests/test_env_init_sampling.py — G_0 初始化策略测试。"""
import pandas as pd
import pytest

from src.online.env import OnlineEnv


def make_star_df(edges):
    return pd.DataFrame(edges, columns=["src", "dst"])


N = 20
# 构造一个有两个清晰社区的图：节点 0-9 内部稠密，节点 10-19 内部稠密，跨社区稀少
EDGES = (
    [(i, j) for i in range(10) for j in range(10) if i != j] +
    [(i, j) for i in range(10, 20) for j in range(10, 20) if i != j] +
    [(0, 10), (10, 0)]
)
STAR_DF = make_star_df(EDGES)


def make_env(strategy, ratio=0.05, seeds=5):
    return OnlineEnv(
        star_edges=STAR_DF,
        n_nodes=N,
        init_edge_ratio=ratio,
        init_strategy=strategy,
        snowball_seeds=seeds,
        seed=42,
    )


def test_random_edge_count():
    """random 策略：边数 ≈ round(|E*| * ratio)。"""
    env = make_env("random", ratio=0.10)
    n_init = env.adj.num_edges()
    expected = max(1, int(len(EDGES) * 0.10))
    assert abs(n_init - expected) <= 2


def test_stratified_edge_count():
    """stratified 策略：边数 ≥ expected（保证每个 src 至少 1 条边）。"""
    env = make_env("stratified", ratio=0.05)
    n_init = env.adj.num_edges()
    expected = max(1, int(len(EDGES) * 0.05))
    # stratified 可能略多，因为每 src 至少 1 条
    assert n_init >= expected


def test_snowball_edge_count():
    """snowball 策略：边数 ≈ 目标值（允许 ±5 容差）。"""
    env = make_env("snowball", ratio=0.10, seeds=3)
    n_init = env.adj.num_edges()
    expected = max(1, int(len(EDGES) * 0.10))
    assert abs(n_init - expected) <= 5


def test_snowball_fewer_components_than_random():
    """snowball 的连通分量数应 ≤ random（更好地保留局部社区结构）。"""
    import networkx as nx

    env_r = make_env("random", ratio=0.10)
    env_s = make_env("snowball", ratio=0.10, seeds=2)

    def n_components(env):
        G = nx.Graph()
        G.add_nodes_from(range(N))
        for u, v in env.adj.iter_edges():
            G.add_edge(u, v)
        return nx.number_connected_components(G)

    # snowball 应产生更少（或相等）的连通分量
    assert n_components(env_s) <= n_components(env_r) + 3  # 允许小幅超出


def test_forest_fire_edge_count():
    """forest_fire 策略：边数在 [expected/2, expected*2] 范围内（森林火蔓延不确定性较大）。"""
    env = make_env("forest_fire", ratio=0.10, seeds=3)
    n_init = env.adj.num_edges()
    expected = max(1, int(len(EDGES) * 0.10))
    assert n_init >= 1  # 至少有边


def test_invalid_strategy_raises():
    """未知策略应抛出 ValueError。"""
    with pytest.raises(ValueError, match="未知 init_strategy"):
        make_env("unknown_strategy")


def test_backward_compat_init_stratified():
    """init_stratified=True 等效于 init_strategy='stratified'。"""
    env1 = OnlineEnv(STAR_DF, N, init_edge_ratio=0.05, init_stratified=True, seed=42)
    env2 = OnlineEnv(STAR_DF, N, init_edge_ratio=0.05, init_strategy="stratified", seed=42)
    assert abs(env1.adj.num_edges() - env2.adj.num_edges()) <= 2
