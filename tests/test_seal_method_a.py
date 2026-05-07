"""tests/test_seal_method_a.py — 验证方案 A：快速路径 forward_batch 与降级路径输出等价。"""
import sys, random
sys.path.insert(0, ".")

import numpy as np
import torch
import dgl

# ── 构造小型 StaticAdjacency（内联，不依赖完整环境）──────────────────────────

class FakeAdj:
    """最小化 StaticAdjacency，只实现 get_csr / out_neighbors_set / in_neighbors_set。"""
    def __init__(self, n, edges):
        self._n = n
        self._out = [set() for _ in range(n)]
        self._in  = [set() for _ in range(n)]
        for s, d in edges:
            self._out[s].add(d)
            self._in[d].add(s)
        self._csr = None

    def out_neighbors_set(self, u): return self._out[u]
    def in_neighbors_set(self, u):  return self._in[u]
    def out_degree(self, u): return len(self._out[u])
    def in_degree(self,  u): return len(self._in[u])

    def get_csr(self):
        if self._csr is not None:
            return self._csr
        n = self._n
        indptr = np.zeros(n + 1, dtype=np.int32)
        for u in range(n):
            indptr[u + 1] = indptr[u] + len(self._out[u])
        total = int(indptr[n])
        indices = np.empty(total, dtype=np.int32)
        for u in range(n):
            s = int(indptr[u])
            nbrs = np.array(sorted(self._out[u]), dtype=np.int32)
            indices[s: s + len(nbrs)] = nbrs
        self._csr = (indptr, indices)
        return self._csr


# ── 构造测试图 ────────────────────────────────────────────────────────────────

def make_test_case(n=20, n_edges=50, n_pairs=32, seed=0):
    """生成随机图和 pairs，返回 (adj, pairs)。"""
    rng = random.Random(seed)
    edges = set()
    while len(edges) < n_edges:
        s = rng.randint(0, n - 1)
        d = rng.randint(0, n - 1)
        if s != d:
            edges.add((s, d))
    adj = FakeAdj(n, list(edges))
    pairs = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(n_pairs)]
    pairs = [(u, v) for u, v in pairs if u != v]
    return adj, pairs


# ── 测试主体 ──────────────────────────────────────────────────────────────────

from src.online.trainer import OnlineTrainer
from src.baseline.seal import SEALModel

torch.manual_seed(42)
model = SEALModel(hidden_dim=16, num_layers=2, label_dim=8, max_label=50)
model.eval()

adj, pairs = make_test_case(n=20, n_edges=60, n_pairs=32, seed=1)

# 用 OnlineTrainer 构建 flat batch 图（含 _drnl）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = OnlineTrainer(
    model=model, optimizer=optimizer, scheduler=None,
    device="cpu", max_hop=2, max_neighbors=10,
)

import numpy as np
rng = np.random.default_rng(0)
u_nbrs = trainer._precompute_u_nbrs([(u, [v]) for u, v in pairs], adj, rng)
bg_with_drnl = trainer._build_flat_batched_graph(pairs, adj, u_nbrs)

assert bg_with_drnl is not None, "batch 图构建失败"
assert "_drnl" in bg_with_drnl.ndata, "_drnl 未写入 ndata"

# 快速路径输出
with torch.no_grad():
    scores_fast = model.forward_batch(bg_with_drnl)

# 降级路径：手动删除 _drnl，走 unbatch 串行路径
bg_no_drnl = bg_with_drnl
_ = bg_no_drnl.ndata.pop("_drnl")
with torch.no_grad():
    scores_slow = model.forward_batch(bg_no_drnl)

# 比较（允许极小浮点误差）
max_diff = (scores_fast - scores_slow).abs().max().item()
print(f"快速路径 vs 降级路径 最大差异: {max_diff:.2e}")
ok = max_diff < 1e-5
print(f"[{'OK' if ok else 'FAIL'}] forward_batch 快慢路径输出等价")

# 验证 _drnl 与逐图 _compute_drnl 标签一致性
bg_with_drnl2 = trainer._build_flat_batched_graph(pairs, adj, u_nbrs)
from src.baseline.seal import _compute_drnl
graphs = dgl.unbatch(bg_with_drnl2)
batch_labels = bg_with_drnl2.ndata["_drnl"]
offset = 0
label_ok = True
for i, g in enumerate(graphs):
    n_g = g.num_nodes()
    per_graph_lbl = _compute_drnl(g)
    batch_lbl = batch_labels[offset: offset + n_g]
    if not torch.equal(per_graph_lbl, batch_lbl):
        print(f"  [FAIL] pair {i}: per_graph={per_graph_lbl.tolist()} batch={batch_lbl.tolist()}")
        label_ok = False
    offset += n_g

print(f"[{'OK' if label_ok else 'FAIL'}] _drnl 标签与逐图 _compute_drnl 一致（{len(graphs)} 个子图）")

all_ok = ok and label_ok
print()
print("全部通过" if all_ok else "存在失败！")
sys.exit(0 if all_ok else 1)
