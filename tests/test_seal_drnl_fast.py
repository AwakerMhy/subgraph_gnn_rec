"""tests/test_seal_drnl_fast.py — 验证新 _compute_drnl 与旧 deque 版输出一致。"""
import sys
sys.path.insert(0, ".")

from collections import deque
import numpy as np
import torch
import dgl

from src.baseline.seal import _compute_drnl as new_drnl


# ── 旧实现（内联，用于对比）──────────────────────────────────────────────────

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
            labels[i] = 1
            continue
        a = du.get(i)
        b = dv.get(i)
        if a is None or b is None:
            labels[i] = 0
        else:
            d = a + b
            labels[i] = 1 + min(a, b) + (d // 2) * ((d - 1) // 2)
    return labels


# ── 辅助函数 ─────────────────────────────────────────────────────────────────

def make_graph(src_list, dst_list, u_idx, v_idx, n):
    g = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=n)
    uf = torch.zeros(n, dtype=torch.bool); uf[u_idx] = True
    vf = torch.zeros(n, dtype=torch.bool); vf[v_idx] = True
    g.ndata["_u_flag"] = uf
    g.ndata["_v_flag"] = vf
    return g


def check(name, g):
    old = old_drnl(g)
    new = new_drnl(g)
    ok = torch.equal(old, new)
    print(f"[{'OK' if ok else 'FAIL'}] {name}")
    if not ok:
        print(f"  old: {old.tolist()}")
        print(f"  new: {new.tolist()}")
    return ok


# ── 测试用例 ──────────────────────────────────────────────────────────────────

all_ok = True

# 1. 基础路径：u=0, v=5，中间有多种路径
g1 = make_graph([0,0,1,2,3,4,0], [1,2,3,3,4,5,5], u_idx=0, v_idx=5, n=6)
all_ok &= check("基础路径图", g1)

# 2. ego_cn 风格：u=0, v=5, N(u)={1,2,3,4}，部分 N(u) 也是 N(v) 的邻居
g2 = make_graph(
    [0,0,0,0, 5,5, 1,2],
    [1,2,3,4, 1,2, 3,4],
    u_idx=0, v_idx=5, n=6
)
all_ok &= check("ego_cn 风格", g2)

# 3. u 和 v 无公共邻居（v 孤立，只通过 u 可达）
g3 = make_graph([0,0,0], [1,2,3], u_idx=0, v_idx=4, n=5)
all_ok &= check("v 孤立（无 v 出边）", g3)

# 4. 稀疏子图（u 和 v 几乎孤立，只有一条不相关边）
g4 = make_graph([2], [2], u_idx=0, v_idx=1, n=3)  # 自环，u/v 不可达对方
all_ok &= check("稀疏子图（u-v 不可达）", g4)

# 5. u 和 v 直接相连
g5 = make_graph([0,0,1], [1,2,2], u_idx=0, v_idx=1, n=3)
all_ok &= check("u-v 直接相连", g5)

# 6. 较大子图（模拟 max_neighbors=30）
import random
random.seed(0)
n6 = 32
edges_s, edges_d = [], []
for _ in range(80):
    s = random.randint(0, n6-1)
    d = random.randint(0, n6-1)
    if s != d:
        edges_s.append(s)
        edges_d.append(d)
g6 = make_graph(edges_s, edges_d, u_idx=0, v_idx=31, n=n6)
all_ok &= check("随机大子图(n=32)", g6)

# 7. GPU 上的图（如果 CUDA 可用）
if torch.cuda.is_available():
    g7 = g1.to("cuda")
    # new_drnl 应能处理 GPU 上的图（内部做 .cpu()）
    try:
        res = new_drnl(g7)
        ref = old_drnl(g1)  # old 在 CPU 上算
        ok7 = torch.equal(res, ref)
        print(f"[{'OK' if ok7 else 'FAIL'}] GPU 图输入")
        all_ok &= ok7
    except Exception as e:
        print(f"[FAIL] GPU 图输入: {e}")
        all_ok = False
else:
    print("[SKIP] GPU 不可用，跳过 GPU 测试")

print()
print("全部通过" if all_ok else "存在失败用例！")
sys.exit(0 if all_ok else 1)
