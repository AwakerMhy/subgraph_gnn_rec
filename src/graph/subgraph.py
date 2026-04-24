"""src/graph/subgraph.py — 局部子图提取

子图设计（2026-04-22 更新）：
  对于链接预测的节点对 (u, v)，子图由以下部分组成（都从可观测图 E_obs 中提取）：
  - u 的一度邻居 N(u)
  - u 和 v 的共同一度邻居 CN(u,v) = N(u) ∩ N(v)
  - u 和 v 本身
  子图节点 = {u} ∪ N(u) ∪ CN(u,v) ∪ {v}

核心约束：
- 严格 t < cutoff_time（防止时间泄露），函数内部含断言检查
- 保留边的方向性
- max_hop 参数（保留向后兼容，但当前设计下实际值为 1）
- max_neighbors_per_node 应用于 N(u)（防止高度节点爆炸）
- 返回 DGLGraph，子图节点特征需调用方挂载
"""
from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import dgl
    import torch
    HAS_DGL = True
except ImportError:
    HAS_DGL = False


class TimeAdjacency:
    """时序邻接表：预构建一次，任意截断时刻的邻居查询 O(log degree)。

    假设传入的 edges 已按 timestamp 升序排列（train.py 中已确保）。
    构建时间 O(|E|)，每次查询 O(log degree)，显著优于慢路径 O(|E|)。

    用法：
        time_adj = TimeAdjacency(all_edges)
        # 在 extract_subgraph 中传入 time_adj 参数即可使用快路径
    """

    def __init__(self, edges: pd.DataFrame) -> None:
        # 每个节点存两个并行 list（已按 timestamp 排序）
        # _out[u] = ([v1, v2, ...], [t1, t2, ...])
        out: dict[int, tuple[list[int], list[float]]] = {}
        inp: dict[int, tuple[list[int], list[float]]] = {}
        for u, v, t in zip(
            edges["src"].to_numpy(),
            edges["dst"].to_numpy(),
            edges["timestamp"].to_numpy(),
        ):
            u, v, t = int(u), int(v), float(t)
            if u not in out:
                out[u] = ([], [])
            out[u][0].append(v)
            out[u][1].append(t)
            if v not in inp:
                inp[v] = ([], [])
            inp[v][0].append(u)
            inp[v][1].append(t)
        self._out = out
        self._in  = inp

    def out_neighbors(self, node: int, cutoff: float) -> list[int]:
        if node not in self._out:
            return []
        nbrs, times = self._out[node]
        return nbrs[: bisect_left(times, cutoff)]

    def in_neighbors(self, node: int, cutoff: float) -> list[int]:
        if node not in self._in:
            return []
        nbrs, times = self._in[node]
        return nbrs[: bisect_left(times, cutoff)]

    def neighbors(self, node: int, cutoff: float) -> list[int]:
        return list(set(self.out_neighbors(node, cutoff)) | set(self.in_neighbors(node, cutoff)))

    def iter_out_neighbors(self, node: int, cutoff: float) -> list[int]:
        """返回截断时刻前的出边邻居列表，无 timestamp，用于只读迭代。"""
        if node not in self._out:
            return []
        nbrs, times = self._out[node]
        return nbrs[: bisect_left(times, cutoff)]

    def out_edges_at(self, node: int, cutoff: float) -> list[tuple[int, float]]:
        """返回 [(neighbor, timestamp), ...] 供 TGAT 使用。"""
        if node not in self._out:
            return []
        nbrs, times = self._out[node]
        idx = bisect_left(times, cutoff)
        return list(zip(nbrs[:idx], times[:idx]))


def _get_neighbors(
    node: int,
    adj_out: dict[int, list[int]],
    adj_in: dict[int, list[int]],
) -> set[int]:
    """返回节点的所有直接邻居（出边+入边方向）。"""
    return set(adj_out.get(node, [])) | set(adj_in.get(node, []))


def _build_adj(edges_t: pd.DataFrame) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """从截断边列表构建邻接表（出边和入边分别存储）。使用 numpy 向量化避免 iterrows。"""
    adj_out: dict[int, list[int]] = {}
    adj_in: dict[int, list[int]] = {}
    for u, v in zip(edges_t["src"].to_numpy(), edges_t["dst"].to_numpy()):
        u, v = int(u), int(v)
        adj_out.setdefault(u, []).append(v)
        adj_in.setdefault(v, []).append(u)
    return adj_out, adj_in


def build_graph_adj(
    edges: pd.DataFrame,
    cutoff_time: float | None = None,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """预构建邻接表，供 extract_subgraph 复用，避免每次重建。

    Args:
        edges:        完整边列表
        cutoff_time:  若指定，只包含 timestamp < cutoff_time 的边

    Returns:
        (adj_out, adj_in)
    """
    if cutoff_time is not None:
        edges = edges[edges["timestamp"] < cutoff_time]
    return _build_adj(edges)


def _bfs_neighbors(
    start: int,
    adj_out: dict[int, list[int]],
    adj_in: dict[int, list[int]],
    max_hop: int,
    max_neighbors_per_node: int,
    rng: np.random.Generator,
) -> set[int]:
    """BFS 采集 start 节点的 max_hop 跳邻居。

    若某节点邻居数超过 max_neighbors_per_node，随机采样。
    """
    visited = {start}
    frontier = {start}

    for _ in range(max_hop):
        next_frontier: set[int] = set()
        for node in frontier:
            neighbors = list(_get_neighbors(node, adj_out, adj_in))
            if len(neighbors) > max_neighbors_per_node:
                idx = rng.choice(len(neighbors), size=max_neighbors_per_node, replace=False)
                neighbors = [neighbors[i] for i in idx]
            next_frontier.update(neighbors)
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    return visited


def _get_one_hop_neighbors(
    node: int,
    adj_out: dict[int, list[int]],
    adj_in: dict[int, list[int]],
) -> set[int]:
    """获取节点的一度邻居（入度+出度邻接）。"""
    return set(_get_neighbors(node, adj_out, adj_in))


def _time_adj_bfs(
    node: int,
    time_adj: "TimeAdjacency",
    cutoff: float,
    max_hop: int,
    max_nbrs: int,
    rng: np.random.Generator,
) -> set[int]:
    """TimeAdjacency 上的 BFS，返回 node 的 max_hop 跳邻居（含自身）。"""
    visited: set[int] = {node}
    frontier: set[int] = {node}
    for _ in range(max_hop):
        next_f: set[int] = set()
        for n in frontier:
            nbrs = time_adj.neighbors(n, cutoff)
            if len(nbrs) > max_nbrs:
                nbrs = [nbrs[i] for i in rng.choice(len(nbrs), max_nbrs, replace=False)]
            next_f.update(nbrs)
        next_f -= visited
        visited |= next_f
        frontier = next_f
    return visited


def extract_subgraph(
    u: int,
    v: int,
    cutoff_time: float,
    edges: pd.DataFrame,
    max_hop: int = 2,
    max_neighbors_per_node: int = 30,
    seed: int = 42,
    prebuilt_adj_out: "dict[int, list[int]] | None" = None,
    prebuilt_adj_in: "dict[int, list[int]] | None" = None,
    store_edge_time: bool = False,
    time_adj: "TimeAdjacency | None" = None,
    node_feat: "torch.Tensor | None" = None,
    subgraph_type: str = "ego_cn",
    precomputed_nbrs_u: "set[int] | None" = None,
) -> "dgl.DGLGraph | None":
    """提取节点对 (u, v) 的局部子图。

    subgraph_type:
      "ego_cn"   （默认）: {u} ∪ N(u) ∪ CN(u,v) ∪ {v}
      "bfs_2hop"          : BFS(u, max_hop) ∪ BFS(v, max_hop)

    快路径优先级：time_adj > prebuilt_adj > 慢路径（DataFrame 全扫）。
    """
    assert HAS_DGL, "需要安装 DGL：pip install dgl"

    rng = np.random.default_rng(seed)

    # ── 路径选择 ──────────────────────────────────────────────────────────────
    if time_adj is not None:
        if subgraph_type == "bfs_2hop":
            nodes_u = _time_adj_bfs(u, time_adj, cutoff_time, max_hop, max_neighbors_per_node, rng)
            nodes_v = _time_adj_bfs(v, time_adj, cutoff_time, max_hop, max_neighbors_per_node, rng)
            subgraph_nodes: set[int] = nodes_u | nodes_v
        else:  # ego_cn
            if precomputed_nbrs_u is not None:
                # 使用调用方预计算并已采样的 N(u)，直接计算公共邻居
                nbrs_u_set = precomputed_nbrs_u
                nbrs_v_list = time_adj.neighbors(v, cutoff_time)
                nbrs_v_set = set(nbrs_v_list)
                common_neighbors = nbrs_u_set & nbrs_v_set
            else:
                nbrs_u_list = time_adj.neighbors(u, cutoff_time)
                nbrs_v_list = time_adj.neighbors(v, cutoff_time)
                nbrs_u_set = set(nbrs_u_list)
                nbrs_v_set = set(nbrs_v_list)
                common_neighbors = nbrs_u_set & nbrs_v_set
                if len(nbrs_u_set) > max_neighbors_per_node:
                    idx = rng.choice(len(nbrs_u_list), size=max_neighbors_per_node, replace=False)
                    nbrs_u_set = {nbrs_u_list[i] for i in idx}
            subgraph_nodes = {u, v} | nbrs_u_set | common_neighbors

        # 边枚举：只取子图内部且 t < cutoff_time 的边
        node_list = sorted(subgraph_nodes)
        global_to_local = {gid: lid for lid, gid in enumerate(node_list)}
        src_local, dst_local, edge_times_list = [], [], []
        _has_iter = hasattr(time_adj, "iter_out_neighbors")
        if not store_edge_time and _has_iter:
            # 避免创建 (dst, timestamp) tuple，直接迭代邻居 ID
            for s in subgraph_nodes:
                for d in time_adj.iter_out_neighbors(s, cutoff_time):
                    if d in subgraph_nodes:
                        src_local.append(global_to_local[s])
                        dst_local.append(global_to_local[d])
        else:
            for s in subgraph_nodes:
                for d, t_e in time_adj.out_edges_at(s, cutoff_time):
                    if d in subgraph_nodes:
                        src_local.append(global_to_local[s])
                        dst_local.append(global_to_local[d])
                        edge_times_list.append(t_e)

    elif prebuilt_adj_out is not None and prebuilt_adj_in is not None:
        # 旧快路径（固定截断，仅用于兼容）
        if store_edge_time:
            raise ValueError("store_edge_time=True 不支持 prebuilt_adj 快路径")
        adj_out = prebuilt_adj_out
        adj_in = prebuilt_adj_in
        has_edges = bool(adj_out or adj_in)

        if not has_edges:
            g = dgl.graph(([], []))
            g.add_nodes(2)
            g.ndata["global_id"] = torch.tensor([u, v], dtype=torch.long)
            g.ndata["_u_flag"] = torch.tensor([True, False])
            g.ndata["_v_flag"] = torch.tensor([False, True])
            if node_feat is not None:
                g.ndata["node_feat"] = node_feat[torch.tensor([u, v], dtype=torch.long)]
            g.u_local_idx = 0
            g.v_local_idx = 1
            return g

        if subgraph_type == "bfs_2hop":
            subgraph_nodes = (
                _bfs_neighbors(u, adj_out, adj_in, max_hop, max_neighbors_per_node, rng) |
                _bfs_neighbors(v, adj_out, adj_in, max_hop, max_neighbors_per_node, rng)
            )
        else:  # ego_cn
            nbrs_u = _get_one_hop_neighbors(u, adj_out, adj_in)
            nbrs_v = _get_one_hop_neighbors(v, adj_out, adj_in)
            common_neighbors = nbrs_u & nbrs_v
            if len(nbrs_u) > max_neighbors_per_node:
                nbrs_u_list = list(nbrs_u)
                idx = rng.choice(len(nbrs_u_list), size=max_neighbors_per_node, replace=False)
                nbrs_u = {nbrs_u_list[i] for i in idx}
            subgraph_nodes = {u, v} | nbrs_u | common_neighbors

        node_list = sorted(subgraph_nodes)
        global_to_local = {gid: lid for lid, gid in enumerate(node_list)}
        src_local, dst_local, edge_times_list = [], [], []
        for s in subgraph_nodes:
            for d in adj_out.get(s, []):
                if d in subgraph_nodes:
                    src_local.append(global_to_local[s])
                    dst_local.append(global_to_local[d])

    else:
        # 慢路径：逐样本过滤 DataFrame
        assert "timestamp" in edges.columns
        edges_t = edges[edges["timestamp"] < cutoff_time].copy()
        if len(edges_t) > 0:
            assert edges_t["timestamp"].max() < cutoff_time, \
                f"时间泄露：截断后仍有 timestamp >= {cutoff_time}"
        if len(edges_t) == 0:
            g = dgl.graph(([], []))
            g.add_nodes(2)
            g.ndata["global_id"] = torch.tensor([u, v], dtype=torch.long)
            g.ndata["_u_flag"] = torch.tensor([True, False])
            g.ndata["_v_flag"] = torch.tensor([False, True])
            if node_feat is not None:
                g.ndata["node_feat"] = node_feat[torch.tensor([u, v], dtype=torch.long)]
            g.u_local_idx = 0
            g.v_local_idx = 1
            return g

        adj_out, adj_in = _build_adj(edges_t)
        if subgraph_type == "bfs_2hop":
            subgraph_nodes = (
                _bfs_neighbors(u, adj_out, adj_in, max_hop, max_neighbors_per_node, rng) |
                _bfs_neighbors(v, adj_out, adj_in, max_hop, max_neighbors_per_node, rng)
            )
        else:  # ego_cn
            nbrs_u = _get_one_hop_neighbors(u, adj_out, adj_in)
            nbrs_v = _get_one_hop_neighbors(v, adj_out, adj_in)
            common_neighbors = nbrs_u & nbrs_v
            if len(nbrs_u) > max_neighbors_per_node:
                nbrs_u_list = list(nbrs_u)
                idx = rng.choice(len(nbrs_u_list), size=max_neighbors_per_node, replace=False)
                nbrs_u = {nbrs_u_list[i] for i in idx}
            subgraph_nodes = {u, v} | nbrs_u | common_neighbors
        node_list = sorted(subgraph_nodes)
        global_to_local = {gid: lid for lid, gid in enumerate(node_list)}
        src_local, dst_local, edge_times_list = [], [], []
        mask = (
            edges_t["src"].isin(subgraph_nodes) &
            edges_t["dst"].isin(subgraph_nodes)
        )
        sub_edges = edges_t[mask]
        for s_val, d_val, t_val in zip(
            sub_edges["src"].to_numpy(),
            sub_edges["dst"].to_numpy(),
            sub_edges["timestamp"].to_numpy(),
        ):
            src_local.append(global_to_local[int(s_val)])
            dst_local.append(global_to_local[int(d_val)])
            edge_times_list.append(float(t_val))

    # ── 统一图构建（三条路径共用） ───────────────────────────────────────────
    n_nodes = len(node_list)

    if len(src_local) == 0:
        g = dgl.graph(([], []))
        g.add_nodes(n_nodes)
    else:
        g = dgl.graph(
            (torch.tensor(src_local, dtype=torch.long),
             torch.tensor(dst_local, dtype=torch.long)),
            num_nodes=n_nodes,
        )

    u_local = global_to_local[u]
    v_local = global_to_local[v]

    g.ndata["global_id"] = torch.tensor(node_list, dtype=torch.long)
    g.ndata["_u_flag"] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata["_v_flag"] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata["_u_flag"][u_local] = True
    g.ndata["_v_flag"][v_local] = True

    if store_edge_time and len(src_local) > 0:
        dt = np.float32(cutoff_time) - np.array(edge_times_list, dtype=np.float32)
        g.edata["dt"] = torch.tensor(dt, dtype=torch.float32)

    if node_feat is not None:
        g.ndata["node_feat"] = node_feat[torch.tensor(node_list, dtype=torch.long)]

    g.u_local_idx = u_local
    g.v_local_idx = v_local

    return g


# ── 离线缓存工具 ──────────────────────────────────────────────────────────────

def cache_subgraphs(
    pairs: list[tuple[int, int, float]],
    edges: pd.DataFrame,
    cache_dir: str | Path,
    dataset_name: str,
    max_hop: int = 2,
    max_neighbors_per_node: int = 30,
    seed: int = 42,
) -> None:
    """预计算并缓存子图到磁盘。

    Args:
        pairs:      [(u, v, cutoff_time), ...] 列表
        edges:      完整边列表
        cache_dir:  缓存目录根（子图存到 cache_dir/<dataset_name>/subgraphs/）
        dataset_name: 数据集名称
    """
    assert HAS_DGL, "需要安装 DGL"

    out_dir = Path(cache_dir) / dataset_name / "subgraphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    graphs = []
    labels = []  # 用于 dgl.save_graphs 的 label dict

    for u, v, t_q in pairs:
        g = extract_subgraph(u, v, t_q, edges, max_hop, max_neighbors_per_node, seed)
        if g is not None:
            graphs.append(g)

    bin_path = str(out_dir / f"hop{max_hop}_k{max_neighbors_per_node}.bin")
    dgl.save_graphs(bin_path, graphs)
    print(f"[cache_subgraphs] 保存 {len(graphs)} 个子图到 {bin_path}")


def load_cached_subgraphs(
    cache_path: str | Path,
) -> list["dgl.DGLGraph"]:
    """从缓存文件加载子图列表。"""
    assert HAS_DGL
    graphs, _ = dgl.load_graphs(str(cache_path))
    return graphs
