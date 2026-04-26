"""src/online/trainer.py — 在线训练器（梯度更新 + 批量打分）。"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except (ImportError, OSError):
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    nn = None     # type: ignore[assignment]

from src.graph.subgraph import extract_subgraph
from src.online.static_adj import StaticAdjacency

# ── Numba 并行加速（可选）────────────────────────────────────────────────────
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
    def njit(*a, **kw):          # type: ignore[misc]
        return (lambda f: f) if not a or callable(a[0]) is False else a[0]
    prange = range               # type: ignore[assignment]


@njit(cache=True, parallel=True, boundscheck=False)
def _count_edges_batched_nb(
    indptr: np.ndarray,        # int32, shape (N+1,)
    indices: np.ndarray,       # int32, shape (E,)
    nodes_flat: np.ndarray,    # int32, all sub-node IDs concatenated
    nodes_offsets: np.ndarray, # int64, shape (P+1,)
) -> np.ndarray:               # int64, shape (P,)
    """统计每个 pair 子图内的有向边数（并行）。"""
    P = nodes_offsets.shape[0] - 1
    counts = np.zeros(P, dtype=np.int64)
    for p in prange(P):
        s = nodes_offsets[p]
        e = nodes_offsets[p + 1]
        cnt = np.int64(0)
        for li in range(e - s):
            g_u = nodes_flat[s + li]
            rs = indptr[g_u]
            re = indptr[g_u + 1]
            for k in range(rs, re):
                g_v = indices[k]
                lo, hi = s, e
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if nodes_flat[mid] < g_v:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < e and nodes_flat[lo] == g_v:
                    cnt += np.int64(1)
        counts[p] = cnt
    return counts


@njit(cache=True, parallel=True, boundscheck=False)
def _fill_edges_batched_nb(
    indptr: np.ndarray,         # int32
    indices: np.ndarray,        # int32
    nodes_flat: np.ndarray,     # int32
    nodes_offsets: np.ndarray,  # int64, shape (P+1,)
    out_src: np.ndarray,        # int64, pre-allocated (total_edges,)
    out_dst: np.ndarray,        # int64, pre-allocated (total_edges,)
    out_gid: np.ndarray,        # int64, pre-allocated (total_edges,)
    edge_offsets: np.ndarray,   # int64, shape (P,) — start pos per pair
) -> None:
    """填充 (src_local, dst_local, gid) 到预分配 buffer（并行）。"""
    P = nodes_offsets.shape[0] - 1
    for p in prange(P):
        s = nodes_offsets[p]
        e = nodes_offsets[p + 1]
        cur = edge_offsets[p]
        for li in range(e - s):
            g_u = nodes_flat[s + li]
            rs = indptr[g_u]
            re = indptr[g_u + 1]
            for k in range(rs, re):
                g_v = indices[k]
                lo, hi = s, e
                while lo < hi:
                    mid = (lo + hi) >> 1
                    if nodes_flat[mid] < g_v:
                        lo = mid + 1
                    else:
                        hi = mid
                if lo < e and nodes_flat[lo] == g_v:
                    out_src[cur] = li
                    out_dst[cur] = lo - s
                    out_gid[cur] = np.int64(p)
                    cur += np.int64(1)


def _extract_edges_csr_fast(
    indptr: np.ndarray,
    indices: np.ndarray,
    sub_nodes_sorted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """单 pair CSR 边提取（fallback，_build_flat_batched_graph 的 update() 路径仍使用）。"""
    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    n_sub = len(sub_nodes_sorted)
    for local_s, global_s in enumerate(sub_nodes_sorted):
        start = int(indptr[global_s])
        end = int(indptr[global_s + 1])
        if start == end:
            continue
        nbrs = indices[start:end]
        pos = np.searchsorted(sub_nodes_sorted, nbrs)
        safe_pos = np.clip(pos, 0, n_sub - 1)
        valid = (pos < n_sub) & (sub_nodes_sorted[safe_pos] == nbrs)
        n_e = int(valid.sum())
        if n_e > 0:
            src_parts.append(np.full(n_e, local_s, dtype=np.int64))
            dst_parts.append(pos[valid].astype(np.int64))

    if not src_parts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(src_parts), np.concatenate(dst_parts)


_EMPTY_INT32: np.ndarray = np.empty(0, dtype=np.int32)


class OnlineTrainer:
    """封装子图提取 → 批量前向 → BCE loss → 梯度更新。

    score() 调用时使用 no_grad + eval mode，update() 切回 train mode。
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        device: torch.device | str,
        max_hop: int = 2,
        max_neighbors: int = 30,
        node_feat: "torch.Tensor | None" = None,
        min_batch_size: int = 4,
        grad_clip: float = 1.0,
        score_chunk_size: int = 512,
        use_amp: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.max_hop = max_hop
        self.max_neighbors = max_neighbors
        # extract_subgraph 在 CPU 上构建图，node_feat 必须保持在 CPU
        self.node_feat = node_feat.cpu() if node_feat is not None else None
        self.min_batch_size = min_batch_size
        self.grad_clip = grad_clip
        self.score_chunk_size = score_chunk_size
        self.use_amp = use_amp and str(torch.device(device)) != "cpu"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None
        # (nbrs_array, len_out, len_in) — 跨轮复用，仅当度变化时失效
        self._u_nbrs_cache: dict[int, tuple[np.ndarray, int, int]] = {}

    # ── 子图构建（update() 使用，pair 数量少，仍走逐对路径）────────────────────

    def _build_subgraphs(
        self,
        pairs: list[tuple[int, int]],
        adj: StaticAdjacency,
        seed: int = 42,
        precomputed_u_nbrs: "dict[int, set[int]] | None" = None,
    ) -> tuple["list[dgl.DGLGraph]", list[int]]:
        """为 pairs 提取子图，返回 (graphs, valid_indices)。"""
        import dgl  # noqa: PLC0415

        graphs, valid_idx = [], []
        for i, (u, v) in enumerate(pairs):
            nbrs_u = precomputed_u_nbrs.get(u) if precomputed_u_nbrs else None
            g = extract_subgraph(
                u, v,
                cutoff_time=float("inf"),
                edges=None,
                max_hop=self.max_hop,
                max_neighbors_per_node=self.max_neighbors,
                seed=seed + i,
                time_adj=adj,
                node_feat=self.node_feat,
                precomputed_nbrs_u=nbrs_u,
            )
            if g is not None:
                graphs.append(g)
                valid_idx.append(i)
        return graphs, valid_idx

    def _precompute_u_nbrs(
        self,
        user_cand_list: list[tuple[int, list[int]]],
        adj: StaticAdjacency,
        rng: np.random.Generator,
    ) -> dict[int, np.ndarray]:
        """为 user_cand_list 中每个唯一 u 预计算并采样 N(u)，返回 sorted int32 ndarray。

        用 np.union1d 合并出/入边集合（归并 O(d)），比 sorted(set|set) 快 5-10×。
        """
        u_nbrs: dict[int, np.ndarray] = {}
        for u, cands in user_cand_list:
            if u not in u_nbrs and cands:
                cur_out = adj.out_degree(u)
                cur_in  = adj.in_degree(u)
                cached = self._u_nbrs_cache.get(u)
                if cached is not None and cached[1] == cur_out and cached[2] == cur_in:
                    u_nbrs[u] = cached[0]
                    continue
                out_arr = np.fromiter(adj.out_neighbors_set(u), dtype=np.int32, count=cur_out)
                in_arr  = np.fromiter(adj.in_neighbors_set(u),  dtype=np.int32, count=cur_in)
                raw: np.ndarray = np.union1d(out_arr, in_arr)   # sorted, unique
                if len(raw) > self.max_neighbors:
                    idx = rng.choice(len(raw), self.max_neighbors, replace=False)
                    raw = np.sort(raw[idx])
                self._u_nbrs_cache[u] = (raw, cur_out, cur_in)
                u_nbrs[u] = raw
        return u_nbrs

    # ── 扁平化批量建图（score_batch 核心，只调用一次 dgl.graph()）──────────────

    def _build_flat_batched_graph(
        self,
        pairs: list[tuple[int, int]],
        adj: StaticAdjacency,
        u_nbrs: dict[int, np.ndarray],
    ) -> "dgl.DGLGraph | None":
        """将 pairs 中所有 (u,v) 对的子图拼成一张扁平 DGL 图，只调用一次 dgl.graph()。

        子图节点集 = {u,v} ∪ N(u)（ego_cn 设计，CN⊆N(u) 故无需再取交集）。
        边提取走 numba 并行双循环（有 numba 时），否则 fallback 到 numpy searchsorted。
        """
        import dgl  # noqa: PLC0415

        indptr, indices = adj.get_csr()

        # ── Phase 1: 逐 pair 计算子图节点集（Python set→ndarray，不可避免）──────
        all_sub_nodes: list[np.ndarray] = []
        pair_node_offset: list[int] = []
        batch_num_nodes: list[int] = []
        u_abs: list[int] = []
        v_abs: list[int] = []
        _uv = np.empty(2, dtype=np.int32)
        node_offset = 0

        for u, v in pairs:
            nbrs_u: np.ndarray = u_nbrs.get(u, _EMPTY_INT32)
            _uv[0], _uv[1] = u, v
            sub_nodes = np.union1d(_uv, nbrs_u)   # sorted int32，O(|N(u)|)
            n = len(sub_nodes)

            pair_node_offset.append(node_offset)
            all_sub_nodes.append(sub_nodes)
            batch_num_nodes.append(n)
            u_abs.append(node_offset + int(np.searchsorted(sub_nodes, u)))
            v_abs.append(node_offset + int(np.searchsorted(sub_nodes, v)))
            node_offset += n

        if node_offset == 0:
            return None

        P = len(pairs)
        nodes_flat = np.concatenate(all_sub_nodes)  # int32
        nodes_offsets = np.empty(P + 1, dtype=np.int64)
        nodes_offsets[0] = 0
        nodes_offsets[1:] = np.cumsum(
            np.array([len(s) for s in all_sub_nodes], dtype=np.int64)
        )

        # ── Phase 2: 边提取（numba 并行 or numpy fallback）────────────────────
        if _HAS_NUMBA:
            counts = _count_edges_batched_nb(indptr, indices, nodes_flat, nodes_offsets)
            edge_offsets = np.empty(P + 1, dtype=np.int64)
            edge_offsets[0] = 0
            edge_offsets[1:] = np.cumsum(counts)
            total = int(edge_offsets[-1])

            if total > 0:
                out_src = np.empty(total, dtype=np.int64)
                out_dst = np.empty(total, dtype=np.int64)
                out_gid = np.empty(total, dtype=np.int64)
                _fill_edges_batched_nb(
                    indptr, indices, nodes_flat, nodes_offsets,
                    out_src, out_dst, out_gid, edge_offsets[:-1],
                )
                pno = np.array(pair_node_offset, dtype=np.int64)
                offsets_per_edge = pno[out_gid]
                src_t = torch.from_numpy(out_src + offsets_per_edge)
                dst_t = torch.from_numpy(out_dst + offsets_per_edge)
            else:
                src_t = torch.zeros(0, dtype=torch.long)
                dst_t = torch.zeros(0, dtype=torch.long)
            batch_num_edges = counts.tolist()
        else:
            all_src_parts: list[np.ndarray] = []
            all_dst_parts: list[np.ndarray] = []
            batch_num_edges = []
            for i in range(P):
                ss, sd = _extract_edges_csr_fast(indptr, indices, all_sub_nodes[i])
                all_src_parts.append(ss + pair_node_offset[i])
                all_dst_parts.append(sd + pair_node_offset[i])
                batch_num_edges.append(len(ss))
            nonempty_s = [s for s in all_src_parts if len(s) > 0]
            nonempty_d = [d for d in all_dst_parts if len(d) > 0]
            if nonempty_s:
                src_t = torch.from_numpy(np.concatenate(nonempty_s))
                dst_t = torch.from_numpy(np.concatenate(nonempty_d))
            else:
                src_t = torch.zeros(0, dtype=torch.long)
                dst_t = torch.zeros(0, dtype=torch.long)

        # ── Phase 3: 一次 dgl.graph() ────────────────────────────────────────
        g = dgl.graph((src_t, dst_t), num_nodes=node_offset)
        g.set_batch_num_nodes(torch.tensor(batch_num_nodes, dtype=torch.long))
        g.set_batch_num_edges(torch.tensor(batch_num_edges, dtype=torch.long))

        u_flag = torch.zeros(node_offset, dtype=torch.bool)
        v_flag = torch.zeros(node_offset, dtype=torch.bool)
        u_flag[u_abs] = True
        v_flag[v_abs] = True
        g.ndata["_u_flag"] = u_flag
        g.ndata["_v_flag"] = v_flag
        g.ndata["_node_id"] = torch.from_numpy(nodes_flat.astype(np.int64))

        if self.node_feat is not None:
            nid_t = torch.from_numpy(nodes_flat.astype(np.int64))
            # pin_memory 让 H→D 拷贝与 GPU forward 重叠（CPU gather 后固定内存）
            gathered = self.node_feat[nid_t]
            if str(self.device) != "cpu":
                gathered = gathered.pin_memory()
            g.ndata["node_feat"] = gathered

        return g

    # ── 打分（推理模式）───────────────────────────────────────────────────────

    def score(
        self,
        u: int,
        candidates: list[int],
        adj: StaticAdjacency,
    ) -> list[float]:
        """对 (u, v) for v in candidates 批量打分，返回 float 列表。"""
        if not candidates:
            return []

        import dgl  # noqa: PLC0415

        pairs = [(u, v) for v in candidates]
        graphs, valid_idx = self._build_subgraphs(pairs, adj)
        if not graphs:
            return [0.0] * len(candidates)

        bg = dgl.batch(graphs).to(self.device)
        self.model.eval()
        with torch.no_grad():
            scores_tensor = self.model.forward_batch(bg)
        del bg

        scores_list = [0.0] * len(candidates)
        for rank, orig_i in enumerate(valid_idx):
            scores_list[orig_i] = scores_tensor[rank].item()
        return scores_list

    def score_batch(
        self,
        user_cand_list: list[tuple[int, list[int]]],
        adj: StaticAdjacency,
        chunk_size: int | None = None,
    ) -> list[list[float]]:
        """多用户候选扁平化建图后批量打分：每 chunk 只调用一次 dgl.graph()。

        Returns list of score lists，与 user_cand_list 顺序一致。
        """
        # 展平 (u, v) pairs
        all_pairs: list[tuple[int, int]] = []
        user_offsets: list[tuple[int, int]] = []
        for u, cands in user_cand_list:
            start = len(all_pairs)
            all_pairs.extend((u, v) for v in cands)
            user_offsets.append((start, len(cands)))

        if not all_pairs:
            return [[] for _ in user_cand_list]

        rng = np.random.default_rng(42)
        precomputed = self._precompute_u_nbrs(user_cand_list, adj, rng)

        global_scores = [0.0] * len(all_pairs)
        self.model.eval()
        _chunk = chunk_size if chunk_size is not None else self.score_chunk_size

        for chunk_start in range(0, len(all_pairs), _chunk):
            chunk = all_pairs[chunk_start: chunk_start + _chunk]
            g = self._build_flat_batched_graph(chunk, adj, precomputed)
            if g is None:
                continue
            g_gpu = g.to(self.device)
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        chunk_scores = self.model.forward_batch(g_gpu)
                else:
                    chunk_scores = self.model.forward_batch(g_gpu)
            del g_gpu
            for i, s in enumerate(chunk_scores.tolist()):
                global_scores[chunk_start + i] = s

        return [
            global_scores[start: start + length]
            for start, length in user_offsets
        ]

    # ── 梯度更新（训练模式）──────────────────────────────────────────────────

    def update(
        self,
        pos_pairs: list[tuple[int, int]],
        neg_pairs: list[tuple[int, int]],
        adj: StaticAdjacency,
    ) -> dict[str, float]:
        """用本轮正负样本做一步梯度更新。返回 {'loss': float}。"""
        all_pairs = pos_pairs + neg_pairs
        if len(all_pairs) < self.min_batch_size:
            return {"loss": float("nan"), "skipped": 1}

        # 复用 score_batch 的快速批量路径，避免逐对调用 extract_subgraph
        _rng = np.random.default_rng(0)
        u_nbrs = self._precompute_u_nbrs([(u, [v]) for u, v in all_pairs], adj, _rng)
        bg = self._build_flat_batched_graph(all_pairs, adj, u_nbrs)
        if bg is None:
            return {"loss": float("nan"), "skipped": 1}

        labels = torch.tensor(
            [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs),
            dtype=torch.float32, device=self.device,
        )

        bg = bg.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                preds = self.model.forward_batch(bg)
            del bg
            loss = nn.functional.binary_cross_entropy(preds.float(), labels)
            loss_val = loss.item()
            self.scaler.scale(loss).backward()
            del preds, loss
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            preds = self.model.forward_batch(bg)
            del bg
            loss = nn.functional.binary_cross_entropy(preds, labels)
            loss_val = loss.item()
            loss.backward()
            del preds, loss
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": loss_val}
