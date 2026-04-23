"""tests/test_subgraph.py — 子图提取与标记的单元测试（不依赖 DGL）"""
import numpy as np
import pandas as pd
import pytest

from src.graph.labeling import drnl_label, build_undirected_adj
from src.graph.negative_sampling import sample_negatives


# ── 测试数据：小型有向图 ──────────────────────────────────────────────
#
#   0 → 1 → 2 → 4
#   0 → 3 → 4
#   时间戳：t=1,2,3,4,5
#
def _small_graph() -> pd.DataFrame:
    return pd.DataFrame({
        "src": [0, 1, 2, 0, 3],
        "dst": [1, 2, 4, 3, 4],
        "timestamp": [1.0, 2.0, 3.0, 4.0, 5.0],
    })


class TestDRNLLabel:
    def test_u_v_get_label_1(self):
        """u 和 v 本身标签应为 1。"""
        node_list = [0, 1, 2, 3, 4]
        adj = build_undirected_adj(_small_graph())
        labels = drnl_label(node_list, u_global=0, v_global=4, adj_undirected=adj)
        assert labels[0] == 1  # u=0
        assert labels[4] == 1  # v=4

    def test_unreachable_gets_label_0(self):
        """不可达节点标签为 0。"""
        # 创建一个有孤立节点的图
        edges = pd.DataFrame({
            "src": [0, 1],
            "dst": [1, 2],
            "timestamp": [1.0, 2.0],
        })
        adj = build_undirected_adj(edges)
        node_list = [0, 1, 2, 5]  # 节点 5 孤立，不在边中
        labels = drnl_label(node_list, u_global=0, v_global=2, adj_undirected=adj)
        assert labels[3] == 0  # 节点 5 不可达

    def test_intermediate_node_label(self):
        """中间节点标签 > 1。"""
        adj = build_undirected_adj(_small_graph())
        node_list = [0, 1, 2, 3, 4]
        labels = drnl_label(node_list, u_global=0, v_global=4, adj_undirected=adj)
        # 节点 1,2,3 都是中间节点，标签 > 1
        assert all(labels[i] > 1 for i in [1, 2, 3])

    def test_label_shape(self):
        adj = build_undirected_adj(_small_graph())
        node_list = [0, 1, 2, 3, 4]
        labels = drnl_label(node_list, u_global=0, v_global=4, adj_undirected=adj)
        assert labels.shape == (5,)
        assert labels.dtype == np.int64


class TestBuildUndirectedAdj:
    def test_both_directions(self):
        """有向边 u→v 应该在无向邻接表中双向出现。"""
        edges = pd.DataFrame({"src": [0], "dst": [1], "timestamp": [1.0]})
        adj = build_undirected_adj(edges)
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_empty_edges(self):
        edges = pd.DataFrame({"src": [], "dst": [], "timestamp": []})
        adj = build_undirected_adj(edges)
        assert adj == {}


class TestNegativeSampling:
    def test_random_strategy_excludes_existing_edges(self):
        edges = _small_graph()
        negs = sample_negatives(
            u=0, cutoff_time=10.0, edges=edges,
            n_nodes=5, strategy="random", k=2, seed=0,
        )
        # 节点 0 的出边目标：1, 3；负样本不应包含 1 或 3（或 0 本身）
        existing = {0, 1, 3}
        assert all(n not in existing for n in negs)

    def test_random_strategy_returns_k_samples(self):
        edges = _small_graph()
        negs = sample_negatives(
            u=0, cutoff_time=10.0, edges=edges,
            n_nodes=5, strategy="random", k=2, seed=1,
        )
        assert len(negs) == 2

    def test_degree_strategy(self):
        edges = _small_graph()
        negs = sample_negatives(
            u=0, cutoff_time=10.0, edges=edges,
            n_nodes=5, strategy="degree", k=2, seed=2,
        )
        existing = {0, 1, 3}
        assert all(n not in existing for n in negs)

    def test_hard_2hop_strategy(self):
        edges = _small_graph()
        # u=0 的二跳：0→1→2 和 0→3→4，候选：2, 4
        negs = sample_negatives(
            u=0, cutoff_time=10.0, edges=edges,
            n_nodes=5, strategy="hard_2hop", k=2, seed=3,
        )
        assert len(negs) > 0
        existing = {0, 1, 3}
        assert all(n not in existing for n in negs)

    def test_cutoff_time_respected(self):
        """只有 t < cutoff_time 的边才算已有出边。"""
        edges = _small_graph()
        # cutoff=1.5：只有边 (0→1, t=1) 生效
        negs = sample_negatives(
            u=0, cutoff_time=1.5, edges=edges,
            n_nodes=5, strategy="random", k=3, seed=4,
        )
        # 此时 u=0 的已有出边只有 1（t=4 的 0→3 在截断后），负样本不含 0 或 1
        assert 1 not in negs
        assert 0 not in negs

    def test_invalid_strategy_raises(self):
        with pytest.raises(AssertionError, match="不支持的策略"):
            sample_negatives(
                u=0, cutoff_time=10.0, edges=_small_graph(),
                n_nodes=5, strategy="unknown", k=1,
            )
