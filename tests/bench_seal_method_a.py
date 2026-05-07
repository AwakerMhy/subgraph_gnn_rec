"""tests/bench_seal_method_a.py — 方案 A 速度对比：快速路径 vs 降级路径。"""
import sys, time, random
sys.path.insert(0, ".")

import numpy as np
import torch
import dgl

from src.online.trainer import OnlineTrainer
from src.baseline.seal import SEALModel


class FakeAdj:
    def __init__(self, n, edges):
        self._n = n
        self._out = [set() for _ in range(n)]
        self._in  = [set() for _ in range(n)]
        for s, d in edges:
            self._out[s].add(d)
            self._in[d].add(s)
        self._csr = None
    def out_neighbors_set(self, u): return self._out[u]
    def in_neighbors_set(self,  u): return self._in[u]
    def out_degree(self, u): return len(self._out[u])
    def in_degree(self,  u): return len(self._in[u])
    def get_csr(self):
        if self._csr: return self._csr
        n = self._n
        indptr = np.zeros(n + 1, dtype=np.int32)
        for u in range(n): indptr[u+1] = indptr[u] + len(self._out[u])
        total = int(indptr[n])
        indices = np.empty(total, dtype=np.int32)
        for u in range(n):
            s = int(indptr[u])
            nbrs = np.array(sorted(self._out[u]), dtype=np.int32)
            indices[s:s+len(nbrs)] = nbrs
        self._csr = (indptr, indices)
        return self._csr


def make_adj_pairs(n, n_edges, n_pairs, seed=0):
    rng = random.Random(seed)
    edges = set()
    while len(edges) < n_edges:
        s, d = rng.randint(0, n-1), rng.randint(0, n-1)
        if s != d: edges.add((s, d))
    adj = FakeAdj(n, list(edges))
    pairs = [(rng.randint(0, n-1), rng.randint(0, n-1)) for _ in range(n_pairs*2)]
    pairs = [(u, v) for u, v in pairs if u != v][:n_pairs]
    return adj, pairs


def run_bench(device_str, batch_sizes, n_repeat=20):
    device = torch.device(device_str)
    model = SEALModel(hidden_dim=32, num_layers=3, label_dim=16, max_label=50).to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(
        model=model, optimizer=optimizer, scheduler=None,
        device=device_str, max_hop=2, max_neighbors=30,
    )
    rng_np = np.random.default_rng(0)

    print(f"\n=== device={device_str} ===")
    print(f"{'batch':>8} {'旧串行(ms)':>14} {'新批量(ms)':>14} {'加速':>8}")
    print("-" * 50)

    for n_pairs in batch_sizes:
        adj, pairs = make_adj_pairs(n=200, n_edges=600, n_pairs=n_pairs, seed=42)
        u_nbrs = trainer._precompute_u_nbrs([(u, [v]) for u, v in pairs], adj, rng_np)

        # ── 预热 ──────────────────────────────────────────────────────────────
        bg = trainer._build_flat_batched_graph(pairs, adj, u_nbrs)
        if bg is None:
            print(f"{n_pairs:>8}  skip (empty graph)")
            continue
        bg_dev = bg.to(device)
        with torch.no_grad():
            model.forward_batch(bg_dev)          # 快速路径预热
        # 降级预热：删掉 _drnl
        bg2 = trainer._build_flat_batched_graph(pairs, adj, u_nbrs)
        bg2.ndata.pop("_drnl")
        bg2_dev = bg2.to(device)
        with torch.no_grad():
            model.forward_batch(bg2_dev)

        # ── 旧串行路径（无 _drnl）──────────────────────────────────────────────
        if device_str.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            bg_old = trainer._build_flat_batched_graph(pairs, adj, u_nbrs)
            bg_old.ndata.pop("_drnl")
            bg_old_dev = bg_old.to(device)
            with torch.no_grad():
                model.forward_batch(bg_old_dev)
            if device_str.startswith("cuda"):
                torch.cuda.synchronize()
        old_ms = (time.perf_counter() - t0) / n_repeat * 1000

        # ── 新批量路径（含 _drnl）──────────────────────────────────────────────
        if device_str.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            bg_new = trainer._build_flat_batched_graph(pairs, adj, u_nbrs)
            bg_new_dev = bg_new.to(device)
            with torch.no_grad():
                model.forward_batch(bg_new_dev)
            if device_str.startswith("cuda"):
                torch.cuda.synchronize()
        new_ms = (time.perf_counter() - t0) / n_repeat * 1000

        speedup = old_ms / new_ms if new_ms > 0 else float("inf")
        print(f"{n_pairs:>8} {old_ms:>14.1f} {new_ms:>14.1f} {speedup:>7.1f}x")


BATCH_SIZES = [16, 64, 128, 256, 512]

run_bench("cpu", BATCH_SIZES, n_repeat=10)

if torch.cuda.is_available():
    run_bench("cuda", BATCH_SIZES, n_repeat=30)
else:
    print("\nCUDA 不可用，跳过 GPU 测试")
