"""tests/test_online.py — 在线学习模块单元测试（10 项）。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.online.feedback import Feedback, FeedbackSimulator
from src.online.replay import ReplayBuffer
from src.online.static_adj import StaticAdjacency


# ── 工具 ──────────────────────────────────────────────────────────────────────

def _make_edges(pairs: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(pairs, columns=["src", "dst"])


def _make_adj(n: int, pairs: list[tuple[int, int]]) -> StaticAdjacency:
    return StaticAdjacency(n, _make_edges(pairs))


# ── 1. StaticAdjacency — 基本查询 ────────────────────────────────────────────

def test_static_adj_basic():
    adj = _make_adj(5, [(0, 1), (0, 2), (1, 3)])
    assert set(adj.out_neighbors(0)) == {1, 2}
    assert set(adj.in_neighbors(3)) == {1}
    assert set(adj.neighbors(0)) == {1, 2}
    assert adj.has_edge(0, 1)
    assert not adj.has_edge(1, 0)
    assert adj.num_edges() == 3


# ── 2. StaticAdjacency — 与 TimeAdjacency 结果一致 ──────────────────────────

def test_static_adj_parity_with_time_adj():
    from src.graph.subgraph import TimeAdjacency  # noqa: PLC0415

    edges = _make_edges([(0, 1), (0, 2), (1, 3), (2, 3)])
    edges["timestamp"] = [0.1, 0.2, 0.3, 0.4]

    tadj = TimeAdjacency(edges)
    sadj = StaticAdjacency(4, edges)

    for node in range(4):
        assert set(sadj.out_neighbors(node)) == set(tadj.out_neighbors(node, 1.0))
        assert set(sadj.in_neighbors(node)) == set(tadj.in_neighbors(node, 1.0))
        assert set(sadj.neighbors(node)) == set(tadj.neighbors(node, 1.0))


# ── 3. StaticAdjacency — 动态增边 ───────────────────────────────────────────

def test_static_adj_dynamic():
    adj = StaticAdjacency(5)
    assert adj.num_edges() == 0
    adj.add_edge(0, 1)
    adj.add_edge(0, 2)
    assert adj.num_edges() == 2
    assert adj.has_edge(0, 1)
    # 重复加不计数
    adj.add_edge(0, 1)
    assert adj.num_edges() == 2
    # iter_edges
    edges = list(adj.iter_edges())
    assert (0, 1) in edges and (0, 2) in edges


# ── 4. extract_subgraph 兼容 StaticAdjacency ────────────────────────────────

def test_extract_subgraph_with_static_adj():
    from src.graph.subgraph import extract_subgraph  # noqa: PLC0415

    adj = _make_adj(6, [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5)])
    g = extract_subgraph(0, 3, cutoff_time=float("inf"), edges=None,
                         max_hop=2, time_adj=adj)
    assert g is not None
    import torch  # noqa: PLC0415
    assert "_u_flag" in g.ndata or "feat" in g.ndata  # 子图已构建


# ── 5. cooldown 逻辑 ─────────────────────────────────────────────────────────

def test_cooldown_logic():
    from src.online.env import OnlineEnv  # noqa: PLC0415

    star = _make_edges([(0, 1), (0, 2), (1, 2)])
    env = OnlineEnv(star, n_nodes=3, init_edge_ratio=0.0,
                    cooldown_rounds=3, p_accept=1.0, seed=0)

    # 拒绝 (0, 2)（不在 star 中才会被拒绝，这里 (0,2) 在 star 中但先测 cooldown 逻辑）
    # 手动插入 cooldown
    env._cooldown[(0, 2)] = 5  # unlock_round = 5

    cands = [(2, 1.0), (1, 0.5)]
    # round 3 < 5：(0,2) 被屏蔽
    filtered = env.mask_cooldown(0, cands, round_idx=3)
    assert all(v != 2 for v, _ in filtered)
    # round 5 == unlock_round：解锁
    filtered2 = env.mask_cooldown(0, cands, round_idx=5)
    assert any(v == 2 for v, _ in filtered2)


# ── 6. FeedbackSimulator — p_accept=1.0 ─────────────────────────────────────

def test_feedback_p_accept_1():
    star = {(0, 1), (0, 2)}
    sim = FeedbackSimulator(star, p_accept=1.0)
    recs = {0: [1, 2, 3]}
    fb = sim.simulate(recs)
    assert set(fb.accepted) == {(0, 1), (0, 2)}
    assert fb.rejected == [(0, 3)]


# ── 7. FeedbackSimulator — p_accept=0.0 ─────────────────────────────────────

def test_feedback_p_accept_0():
    star = {(0, 1), (0, 2)}
    sim = FeedbackSimulator(star, p_accept=0.0, rng=np.random.default_rng(0))
    recs = {0: [1, 2]}
    fb = sim.simulate(recs)
    assert fb.accepted == []
    assert len(fb.rejected) == 2


# ── 8. env.step 更新 G_t，coverage 单调非减 ─────────────────────────────────

def test_env_step_updates_gt():
    from src.online.env import OnlineEnv as _Env  # noqa: PLC0415
    star = _make_edges([(0, 1), (0, 2), (1, 2)])
    env = _Env(star, n_nodes=3, init_edge_ratio=0.0,
                    cooldown_rounds=2, p_accept=1.0, seed=0)

    prev_cov = env.coverage()
    recs = {0: [1]}
    fb = env.step(recs, round_idx=0)
    assert (0, 1) in [(u, v) for u, v in fb.accepted]
    assert env.adj.has_edge(0, 1)
    assert env.coverage() >= prev_cov


# ── 9. ReplayBuffer — 禁用时 no-op ──────────────────────────────────────────

def test_replay_disabled():
    buf = ReplayBuffer(capacity=0)
    buf.push([(0, 1)], [(0, 2)], round_idx=0)
    pos, neg = buf.sample(10)
    assert pos == [] and neg == []
    assert len(buf) == 0


# ── 10. loop 烟测（3 轮 SBM，不崩溃，有输出）─────────────────────────────────

def test_loop_smoke(tmp_path):
    cfg = {
        "dataset": {"type": "sbm", "params": {
            "n_nodes": 50, "n_communities": 3,
            "p_in": 0.3, "p_out": 0.02,
            "T": 10, "edges_per_step": 3, "seed": 42,
        }},
        "init_edge_ratio": 0.20,
        "user_sample_ratio": 0.30,
        "total_rounds": 3,
        "recall": {"method": "common_neighbors", "top_k_recall": 10},
        "recommend": {"top_k": 3},
        "feedback": {"p_accept": 1.0, "cooldown_rounds": 1},
        "trainer": {
            "update_every_n_rounds": 1,
            "batch_subgraph_max_hop": 1,
            "max_neighbors": 10,
            "lr": 1e-3,
            "min_batch_size": 1,
            "grad_clip": 1.0,
            "scheduler": {"strategy": "constant"},
        },
        "model": {"hidden_dim": 16, "num_layers": 2, "encoder_type": "last", "node_feat_dim": 0},
        "replay": {"capacity": 0, "sample_n": 0},
        "eval": {"k_list": [3], "graph_every_n": 1, "degree_bins": 10},
        "runtime": {"seed": 42, "device": "cpu",
                    "out_dir": str(tmp_path), "log_every": 1},
    }
    from src.online.loop import run_online_simulation  # noqa: PLC0415
    df = run_online_simulation(cfg)
    assert len(df) == 3
    assert "coverage" in df.columns
    assert "precision_k" in df.columns
