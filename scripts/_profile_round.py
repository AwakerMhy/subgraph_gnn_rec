"""单轮各阶段耗时诊断（在主循环关键点插桩）。"""
import time, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib) and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(_torch_lib)

import yaml, numpy as np, torch
from src.online.env import OnlineEnv
from src.online.static_adj import StaticAdjacency
from src.online.trainer import OnlineTrainer
from src.online.schedule import build_scheduler
from src.recall.registry import build_recall
from src.model.model import LinkPredModel
from src.utils.seed import set_seed

with open("configs/online/college_msg_full.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

set_seed(cfg["runtime"]["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import pandas as pd
edges_df = pd.read_csv(cfg["dataset"]["path"])[["src","dst"]].drop_duplicates()
n_nodes = int(max(edges_df["src"].max(), edges_df["dst"].max())) + 1

env = OnlineEnv(
    star_edges=edges_df, n_nodes=n_nodes,
    init_edge_ratio=cfg["init_edge_ratio"],
    user_sample_ratio=0.10,
    cooldown_rounds=cfg["feedback"]["cooldown_rounds"],
    p_pos=cfg["feedback"]["p_pos"], p_neg=cfg["feedback"]["p_neg"],
    seed=cfg["runtime"]["seed"],
    init_strategy=cfg.get("init_strategy"),
    user_selector_cfg=cfg["user_selector"],
)
env.set_cooldown_mode(cfg["feedback"]["cooldown_mode"])
adj = env.get_adjacency()

recall_cfg = cfg["recall"]
recall = build_recall({"method": recall_cfg["method"], "top_k": recall_cfg["top_k_recall"],
                       "components": recall_cfg["components"]}, adj, n_nodes)

model = LinkPredModel(hidden_dim=cfg["model"]["hidden_dim"],
                      num_layers=cfg["model"]["num_layers"]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["trainer"]["lr"])
from src.online.schedule import build_scheduler
sc = cfg["trainer"]["scheduler"]
scheduler = build_scheduler(optimizer, total_steps=100,
                            warmup_steps=sc["warmup_rounds"], min_lr=sc["min_lr"])
trainer = OnlineTrainer(model=model, optimizer=optimizer, scheduler=scheduler,
                        device=device, max_neighbors=cfg["trainer"]["max_neighbors"],
                        score_chunk_size=cfg["trainer"]["score_chunk_size"],
                        use_amp=cfg["trainer"]["use_amp"])

# ── 跑 5 轮，每轮计各阶段时间 ──────────────────────────────────────────────
for t in range(5):
    t0 = time.perf_counter()

    recall.update_graph(t)
    t1 = time.perf_counter()

    U = env.sample_active_users(t)
    if hasattr(recall, "precompute_for_users"):
        recall.precompute_for_users(list(U))
    t2 = time.perf_counter()

    user_cand_nodes = {}
    for u in U:
        cands = recall.candidates(u, cutoff_time=float("inf"), top_k=recall_cfg["top_k_recall"])
        cands = env.mask_existing_edges(u, cands)
        cands = env.mask_cooldown(u, cands, t)
        user_cand_nodes[u] = [v for v, _ in cands] if cands else []
    t3 = time.perf_counter()

    gnn_inputs = [(u, user_cand_nodes[u]) for u in U if user_cand_nodes[u]]
    batch_scores = trainer.score_batch(gnn_inputs, adj)
    t4 = time.perf_counter()

    # dummy feedback
    recs = {u: user_cand_nodes[u][:10] for u in U}
    feedback = env.step(recs, t)
    t5 = time.perf_counter()

    # CSR rebuild timing
    adj._csr_dirty = True
    tc0 = time.perf_counter()
    adj.get_csr()
    tc1 = time.perf_counter()

    print(f"Round {t+1}: recall_update={1000*(t1-t0):.1f}ms  ppr_precompute={1000*(t2-t1):.1f}ms  "
          f"candidates={1000*(t3-t2):.1f}ms  score_batch={1000*(t4-t3):.1f}ms  "
          f"env_step={1000*(t5-t4):.1f}ms  csr_rebuild={1000*(tc1-tc0):.1f}ms  "
          f"total={1000*(t5-t0):.1f}ms", flush=True)
