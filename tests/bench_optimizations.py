"""
正确性 + 速度测试
模拟 facebook_ego 规模：4039 节点, ~15000 边, 400 活跃用户, 50 候选/用户
"""
import sys, time, os
import numpy as np

# Ensure src module can be found when run from different directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.online.static_adj import StaticAdjacency
from src.recall.heuristic import CommonNeighborsRecall, AdamicAdarRecall, _two_hop_scores

# Try to import trainer components; defer torch import for later
try:
    from src.online.trainer import _HAS_NUMBA, _EMPTY_INT32, _extract_edges_csr_fast
except (ImportError, OSError) as import_err:
    # Fallback if trainer module fails to import due to torch issues
    _HAS_NUMBA = False
    _EMPTY_INT32 = np.array([], dtype=np.int32)

N_NODES = 4039
N_EDGES = 15000
N_USERS = 400
TOP_K   = 50
REPS    = 5

rng = np.random.default_rng(42)
print(f"=== 规模：{N_NODES} 节点 / {N_EDGES} 边 / {N_USERS} 活跃用户 / top_k={TOP_K} ===")
print(f"numba 可用：{_HAS_NUMBA}\n")

# ── 构造随机图 ────────────────────────────────────────────────────────────────
adj = StaticAdjacency(N_NODES)
srcs = rng.integers(0, N_NODES, N_EDGES)
dsts = rng.integers(0, N_NODES, N_EDGES)
for u, v in zip(srcs, dsts):
    if u != v:
        adj.add_edge(int(u), int(v))
print(f"图构建完成：实际边数={adj.num_edges()}")

active_users = rng.integers(0, N_NODES, N_USERS).tolist()

# ══════════════════════════════════════════════════════════════════════════════
# 1. CN/AA 召回：正确性验证
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [1] CN/AA 召回正确性 ──")

cn_new = CommonNeighborsRecall(adj, N_NODES)
cn_new.precompute_for_users(active_users)

n_ok_cn = 0
for u in active_users[:10]:
    new_cands = set(v for v, _ in cn_new.candidates(u, float("inf"), TOP_K))
    fb = _two_hop_scores(u, float("inf"), adj, use_adamic_adar=False)
    fb_cands = set(v for v, _ in sorted(fb.items(), key=lambda x: -x[1])[:TOP_K])
    if new_cands == fb_cands:
        n_ok_cn += 1
    else:
        miss  = fb_cands - new_cands
        extra = new_cands - fb_cands
        print(f"  CN MISMATCH u={u}: missing={miss}, extra={extra}")

print(f"  CN 正确性：{n_ok_cn}/10 通过")

aa_new = AdamicAdarRecall(adj, N_NODES)
aa_new.precompute_for_users(active_users)

n_ok_aa = 0
for u in active_users[:10]:
    new_cands = set(v for v, _ in aa_new.candidates(u, float("inf"), TOP_K))
    fb = _two_hop_scores(u, float("inf"), adj, use_adamic_adar=True)
    fb_cands = set(v for v, _ in sorted(fb.items(), key=lambda x: -x[1])[:TOP_K])
    if new_cands == fb_cands:
        n_ok_aa += 1
    else:
        miss  = fb_cands - new_cands
        extra = new_cands - fb_cands
        print(f"  AA MISMATCH u={u}: missing={miss}, extra={extra}")

print(f"  AA 正确性：{n_ok_aa}/10 通过")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CN/AA 召回：速度对比
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [2] CN 召回速度（400 用户 × top_k=50）──")

t0 = time.perf_counter()
for _ in range(REPS):
    for u in active_users:
        fb = _two_hop_scores(u, float("inf"), adj, use_adamic_adar=False)
        sorted(fb.items(), key=lambda x: -x[1])[:TOP_K]
t_old_cn = (time.perf_counter() - t0) / REPS

t0 = time.perf_counter()
for _ in range(REPS):
    cn_b = CommonNeighborsRecall(adj, N_NODES)
    cn_b.precompute_for_users(active_users)
    for u in active_users:
        cn_b.candidates(u, float("inf"), TOP_K)
t_new_cn = (time.perf_counter() - t0) / REPS

print(f"  旧（逐用户 set intersection）：{t_old_cn*1000:.1f} ms")
print(f"  新（sparse matmul + cache）  ：{t_new_cn*1000:.1f} ms")
print(f"  加速比：{t_old_cn/t_new_cn:.1f}×")

print("\n── [2b] AA 召回速度 ──")

t0 = time.perf_counter()
for _ in range(REPS):
    for u in active_users:
        fb = _two_hop_scores(u, float("inf"), adj, use_adamic_adar=True)
        sorted(fb.items(), key=lambda x: -x[1])[:TOP_K]
t_old_aa = (time.perf_counter() - t0) / REPS

t0 = time.perf_counter()
for _ in range(REPS):
    aa_b = AdamicAdarRecall(adj, N_NODES)
    aa_b.precompute_for_users(active_users)
    for u in active_users:
        aa_b.candidates(u, float("inf"), TOP_K)
t_new_aa = (time.perf_counter() - t0) / REPS

print(f"  旧：{t_old_aa*1000:.1f} ms    新：{t_new_aa*1000:.1f} ms    加速比：{t_old_aa/t_new_aa:.1f}×")

# ══════════════════════════════════════════════════════════════════════════════
# 3. _precompute_u_nbrs：类型验证 + 速度
# ══════════════════════════════════════════════════════════════════════════════
print("\n── [3] _precompute_u_nbrs 速度 ──")

try:
    import torch.nn as nn, torch
    from src.online.trainer import OnlineTrainer

    class DummyModel(nn.Module):
        def forward_batch(self, g): return torch.sigmoid(torch.zeros(g.batch_size))

    model = DummyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = OnlineTrainer(model=model, optimizer=opt, scheduler=None,
                            device="cpu", max_hop=1, max_neighbors=30)
    _HAS_TORCH = True
except (ImportError, RuntimeError, OSError) as e:
    print(f"  ⚠ Torch import failed ({type(e).__name__}): 跳过 torch 依赖的测试")
    _HAS_TORCH = False

if _HAS_TORCH:
    pairs_sample = []
    for u, v in zip(rng.integers(0,N_NODES,600), rng.integers(0,N_NODES,600)):
        if u != v:
            pairs_sample.append((int(u), int(v)))
        if len(pairs_sample) == N_USERS:
            break
    ucl = [(u, [v]) for u, v in pairs_sample]

    rng4 = np.random.default_rng(7)

    t0 = time.perf_counter()
    for _ in range(REPS):
        u_nbrs = trainer._precompute_u_nbrs(ucl, adj, rng4)
    t_precomp = (time.perf_counter() - t0) / REPS

    assert all(isinstance(v, np.ndarray) for v in u_nbrs.values()), "类型错误：应为 ndarray"
    assert all(v.dtype == np.int32 for v in u_nbrs.values()), "dtype 错误：应为 int32"
    print(f"  _precompute_u_nbrs 类型验证通过（ndarray int32），耗时={t_precomp*1000:.1f} ms")
else:
    print("  (跳过：需要 torch 环境)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. _build_flat_batched_graph：正确性 + 速度
# ══════════════════════════════════════════════════════════════════════════════
if _HAS_TORCH:
    print("\n── [4] _build_flat_batched_graph 正确性 ──")

    pairs_20 = [(int(u), int(v)) for u, v in zip(
        rng.integers(0,N_NODES,40), rng.integers(0,N_NODES,40)) if u != v][:20]
    u_nbrs_20 = trainer._precompute_u_nbrs([(u,[v]) for u,v in pairs_20], adj, np.random.default_rng(1))

    g = trainer._build_flat_batched_graph(pairs_20, adj, u_nbrs_20)
    assert g is not None
    assert g.batch_size == len(pairs_20), f"batch_size={g.batch_size}"
    assert "_u_flag" in g.ndata and "_v_flag" in g.ndata
    uf = g.ndata["_u_flag"].sum().item()
    vf = g.ndata["_v_flag"].sum().item()
    assert uf == len(pairs_20), f"u_flag count={uf}"
    assert vf == len(pairs_20), f"v_flag count={vf}"
    print(f"  图构建通过：batch={g.batch_size}, nodes={g.num_nodes()}, edges={g.num_edges()}, u_flag={uf}, v_flag={vf}")

    print("\n── [4b] _build_flat_batched_graph 速度（512 pairs）──")

    pairs_512 = [(int(u), int(v)) for u, v in zip(
        rng.integers(0,N_NODES,600), rng.integers(0,N_NODES,600)) if u != v][:512]
    u_nbrs_512 = trainer._precompute_u_nbrs([(u,[v]) for u,v in pairs_512], adj, np.random.default_rng(2))

    indptr, indices_arr = adj.get_csr()

    def old_build(pairs, u_nbrs_dict):
        """旧实现：set ops + 逐 pair searchsorted loop"""
        node_offset = 0
        for u, v in pairs:
            nbrs_u_arr = u_nbrs_dict.get(u, _EMPTY_INT32)
            nbrs_u_set = set(nbrs_u_arr.tolist())
            nbrs_v = adj._out[v] | adj._in[v]
            cn = nbrs_u_set & nbrs_v
            sub_nodes = np.array(sorted({u, v} | nbrs_u_set | cn), dtype=np.int32)
            ss, sd = _extract_edges_csr_fast(indptr, indices_arr, sub_nodes)
            node_offset += len(sub_nodes)
        return node_offset

    # 热身
    _ = trainer._build_flat_batched_graph(pairs_512[:32], adj, u_nbrs_512)
    _ = old_build(pairs_512[:32], u_nbrs_512)

    t0 = time.perf_counter()
    for _ in range(REPS):
        old_build(pairs_512, u_nbrs_512)
    t_old_bg = (time.perf_counter() - t0) / REPS

    t0 = time.perf_counter()
    for _ in range(REPS):
        trainer._build_flat_batched_graph(pairs_512, adj, u_nbrs_512)
    t_new_bg = (time.perf_counter() - t0) / REPS

    method = "numba" if _HAS_NUMBA else "numpy fallback"
    print(f"  旧（set ops + searchsorted loop）：{t_old_bg*1000:.1f} ms")
    print(f"  新（union1d + {method}）         ：{t_new_bg*1000:.1f} ms")
    print(f"  加速比：{t_old_bg/t_new_bg:.1f}×")
else:
    print("\n── [4] _build_flat_batched_graph 正确性 ──")
    print("  (跳过：需要 torch 环境)")

# ══════════════════════════════════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════════════════════════════════
print("\n══ 汇总 ══════════════════════════════════")
print(f"  CN 召回    旧={t_old_cn*1000:.0f}ms  新={t_new_cn*1000:.0f}ms  加速={t_old_cn/t_new_cn:.1f}×")
print(f"  AA 召回    旧={t_old_aa*1000:.0f}ms  新={t_new_aa*1000:.0f}ms  加速={t_old_aa/t_new_aa:.1f}×")
if _HAS_TORCH:
    print(f"  子图建图   旧={t_old_bg*1000:.0f}ms  新={t_new_bg*1000:.0f}ms  加速={t_old_bg/t_new_bg:.1f}×")
else:
    print(f"  子图建图   (跳过：需要 torch 环境)")
print(f"  numba: {'启用' if _HAS_NUMBA else '未启用（Smart App Control 阻止）'}")
