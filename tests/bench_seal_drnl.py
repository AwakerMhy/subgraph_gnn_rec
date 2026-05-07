"""tests/bench_seal_drnl.py — 对比新旧 _compute_drnl 速度。"""
import sys, time, random
sys.path.insert(0, ".")

from collections import deque
import numpy as np
import torch
import dgl

from src.baseline.seal import _compute_drnl as new_drnl


def _bfs_old(start, adj, n):
    dist = {start: 0}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in adj.get(cur, []):
            if nb not in dist:
                dist[nb] = dist[cur] + 1
                q.append(nb)
    return dist


def old_drnl(g):
    src, dst = g.edges()
    adj = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        adj.setdefault(s, []).append(d)
        adj.setdefault(d, []).append(s)
    n = g.num_nodes()
    u = int(g.ndata["_u_flag"].nonzero(as_tuple=False)[0].item())
    v = int(g.ndata["_v_flag"].nonzero(as_tuple=False)[0].item())
    du = _bfs_old(u, adj, n)
    dv = _bfs_old(v, adj, n)
    labels = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        if i == u or i == v:
            labels[i] = 1; continue
        a = du.get(i); b = dv.get(i)
        if a is None or b is None:
            labels[i] = 0
        else:
            d = a + b
            labels[i] = 1 + min(a, b) + (d // 2) * ((d - 1) // 2)
    return labels


def make_random_graph(n, n_edges, u_idx=0, v_idx=None, seed=0):
    random.seed(seed)
    if v_idx is None:
        v_idx = n - 1
    src_list, dst_list = [], []
    seen = set()
    while len(src_list) < n_edges:
        s = random.randint(0, n - 1)
        d = random.randint(0, n - 1)
        if s != d and (s, d) not in seen:
            src_list.append(s); dst_list.append(d)
            seen.add((s, d))
    g = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=n)
    uf = torch.zeros(n, dtype=torch.bool); uf[u_idx] = True
    vf = torch.zeros(n, dtype=torch.bool); vf[v_idx] = True
    g.ndata["_u_flag"] = uf
    g.ndata["_v_flag"] = vf
    return g


def bench(graphs, n_repeat=200):
    # 旧
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        for g in graphs:
            old_drnl(g)
    old_ms = (time.perf_counter() - t0) / n_repeat * 1000

    # 新
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        for g in graphs:
            new_drnl(g)
    new_ms = (time.perf_counter() - t0) / n_repeat * 1000

    return old_ms, new_ms


print(f"{'场景':<30} {'旧(ms)':>10} {'新(ms)':>10} {'加速':>8}")
print("-" * 62)

configs = [
    ("n=10, edges=20,  batch=1",   10,  20,   1),
    ("n=32, edges=60,  batch=1",   32,  60,   1),
    ("n=32, edges=60,  batch=32",  32,  60,  32),
    ("n=32, edges=60,  batch=512", 32,  60, 512),
    ("n=32, edges=100, batch=512", 32, 100, 512),
]

for label, n, ne, batch in configs:
    graphs = [make_random_graph(n, ne, seed=i) for i in range(batch)]
    n_repeat = max(5, 500 // batch)
    old_ms, new_ms = bench(graphs, n_repeat=n_repeat)
    speedup = old_ms / new_ms if new_ms > 0 else float("inf")
    print(f"{label:<30} {old_ms:>10.2f} {new_ms:>10.2f} {speedup:>7.1f}x")
