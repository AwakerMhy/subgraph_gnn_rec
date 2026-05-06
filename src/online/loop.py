"""src/online/loop.py — 在线学习主循环。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import dgl  # noqa: F401 — 先于 numpy/torch 导入，触发 Windows DLL 目录注册
import torch
import numpy as np
import pandas as pd

from src.model.model import LinkPredModel
from src.online.env import OnlineEnv
from src.online.evaluator import RoundMetrics
from src.online.replay import ReplayBuffer
from src.online.schedule import build_scheduler
from src.online.trainer import OnlineTrainer
from src.recall.registry import build_recall
from src.utils.seed import set_seed


def _drop_isolated_nodes(
    edges_df: pd.DataFrame,
    n_nodes: int,
    node_feat: "torch.Tensor | None",
) -> tuple[pd.DataFrame, int, "torch.Tensor | None"]:
    """删除在 G* 中没有任何边的孤立节点，并将节点 ID 重映射为连续整数。"""
    active = sorted(set(edges_df["src"].tolist()) | set(edges_df["dst"].tolist()))
    if len(active) == n_nodes:
        return edges_df, n_nodes, node_feat
    remap = {old: new for new, old in enumerate(active)}
    edges_df = edges_df.copy()
    edges_df["src"] = edges_df["src"].map(remap)
    edges_df["dst"] = edges_df["dst"].map(remap)
    new_n = len(active)
    if node_feat is not None:
        node_feat = node_feat[active]
    removed = n_nodes - new_n
    print(f"[preprocess] 删除 {removed} 个孤立节点，剩余 {new_n} 个节点", flush=True)
    return edges_df, new_n, node_feat


def _load_dataset(cfg: dict) -> tuple[pd.DataFrame, int, "torch.Tensor | None"]:
    """加载数据集，丢弃时间戳，返回 (star_edges_df, n_nodes, node_feat)。"""
    dtype = cfg["dataset"]["type"]

    if dtype in ("sbm", "dcsbm", "triadic"):
        params = cfg["dataset"].get("params", {})
        if dtype == "sbm":
            from src.dataset.synthetic.sbm import SBMGenerator  # noqa: PLC0415
            gen = SBMGenerator(**params)
        elif dtype == "dcsbm":
            from src.dataset.synthetic.dcsbm import DCSBMGenerator  # noqa: PLC0415
            gen = DCSBMGenerator(**params)
        else:
            from src.dataset.synthetic.triadic import TriadicGenerator  # noqa: PLC0415
            gen = TriadicGenerator(**params)
        edges_df = gen.generate()
        edges_df = edges_df[["src", "dst"]].drop_duplicates().reset_index(drop=True)
        n_nodes = int(max(edges_df["src"].max(), edges_df["dst"].max())) + 1
        feats = gen.get_node_features()
        node_feat = torch.tensor(feats, dtype=torch.float32) if feats is not None else None
        return _drop_isolated_nodes(edges_df, n_nodes, node_feat)

    if dtype == "college_msg":
        path = cfg["dataset"].get("path", "data/processed/college_msg/edges.csv")
        edges_df = pd.read_csv(path)
        edges_df = edges_df[["src", "dst"]].drop_duplicates().reset_index(drop=True)
        n_nodes = int(max(edges_df["src"].max(), edges_df["dst"].max())) + 1
        return _drop_isolated_nodes(edges_df, n_nodes, None)

    # 通用 path 模式
    path = cfg["dataset"]["path"]
    edges_df = pd.read_csv(path)
    edges_df = edges_df[["src", "dst"]].drop_duplicates().reset_index(drop=True)
    n_nodes = int(max(edges_df["src"].max(), edges_df["dst"].max())) + 1
    return _drop_isolated_nodes(edges_df, n_nodes, None)


_HEURISTIC_TYPES: frozenset[str] = frozenset({"cn", "aa", "jaccard", "pa"})
_NO_MODEL_TYPES: frozenset[str] = frozenset({"random", "ground_truth"}) | _HEURISTIC_TYPES
# seal / graphsage_emb / gat_emb 均通过 OnlineTrainer 走同一 GNN 打分路径
_GNN_LIKE_TYPES: frozenset[str] = frozenset({"gnn", "seal", "graphsage_emb", "gat_emb"})


def _score_heuristic(
    method: str,
    u: int,
    cands: list[int],
    adj: "StaticAdjacency",
) -> list[float]:
    """为候选列表打启发式分数（无向邻居集合）。"""
    nu = adj.out_neighbors_set(u) | adj.in_neighbors_set(u)
    nu_size = len(nu)
    scores: list[float] = []
    for v in cands:
        nv = adj.out_neighbors_set(v) | adj.in_neighbors_set(v)
        if method == "pa":
            scores.append(float(nu_size * len(nv)))
            continue
        common = nu & nv
        cn = len(common)
        if method == "cn":
            scores.append(float(cn))
        elif method == "aa":
            aa_score = 0.0
            for w in common:
                dw = len(adj.out_neighbors_set(w) | adj.in_neighbors_set(w))
                aa_score += 1.0 / np.log(max(dw, 2))
            scores.append(aa_score)
        else:  # jaccard
            union_size = len(nu | nv)
            scores.append(float(cn) / union_size if union_size > 0 else 0.0)
    return scores


def _build_optimizer(model: torch.nn.Module, trainer_cfg: dict) -> torch.optim.Optimizer:
    opt_name = trainer_cfg.get("optimizer", "adam").lower()
    lr = trainer_cfg.get("lr", 1e-3)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=trainer_cfg.get("momentum", 0.9))
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=trainer_cfg.get("weight_decay", 0.0))


def run_online_simulation(cfg: dict) -> pd.DataFrame:
    """执行在线学习仿真，返回 per-round 指标 DataFrame。"""
    seed = cfg.get("runtime", {}).get("seed", 42)
    set_seed(seed)
    device_str = cfg.get("runtime", {}).get("device", "cpu")
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        import warnings as _warnings
        _warnings.warn(
            f"config 指定 device={device_str!r}，但当前环境无 CUDA，已自动 fallback 到 CPU。",
            RuntimeWarning,
            stacklevel=1,
        )
        device_str = "cpu"
    device = torch.device(device_str)
    out_dir = Path(cfg.get("runtime", {}).get("out_dir", "results/online"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_every = cfg.get("runtime", {}).get("log_every", 1)

    # ── 数据集 ────────────────────────────────────────────────────────────────
    star_edges, n_nodes, node_feat = _load_dataset(cfg)
    if node_feat is not None:
        node_feat = node_feat.to(device)

    # ── 环境 ──────────────────────────────────────────────────────────────────
    fb_cfg = cfg.get("feedback", {})
    sel_cfg = cfg.get("user_selector", {})
    # user_sample_ratio 作为 selector.sample_ratio 的同义词（向后兼容）
    if "sample_ratio" not in sel_cfg and "user_sample_ratio" in cfg:
        sel_cfg = {**sel_cfg, "sample_ratio": cfg["user_sample_ratio"]}
    env = OnlineEnv(
        star_edges=star_edges,
        n_nodes=n_nodes,
        init_edge_ratio=cfg.get("init_edge_ratio", 0.05),
        user_sample_ratio=cfg.get("user_sample_ratio", 0.10),
        cooldown_rounds=fb_cfg.get("cooldown_rounds", 5),
        p_accept=fb_cfg.get("p_accept", 1.0),
        p_pos=fb_cfg.get("p_pos", None),
        p_neg=fb_cfg.get("p_neg", 0.0),
        seed=seed,
        init_stratified=cfg.get("init_stratified", False),
        init_strategy=cfg.get("init_strategy", None),
        snowball_seeds=cfg.get("snowball_seeds", 5),
        user_selector_cfg=sel_cfg,
    )
    cooldown_mode = fb_cfg.get("cooldown_mode", "hard")
    env.set_cooldown_mode(cooldown_mode)
    adj = env.get_adjacency()

    # ── 召回 ──────────────────────────────────────────────────────────────────
    recall_cfg = cfg.get("recall", {})
    # 整体透传 recall_cfg：保留 components/alpha/recompute_every_n 等子字段
    # 同时把 top_k_recall 提升为 top_k 的兜底
    _recall_full = dict(recall_cfg)
    _recall_full.setdefault("method", "adamic_adar")
    _recall_full.setdefault("top_k", recall_cfg.get("top_k_recall", 50))
    recall = build_recall(_recall_full, adj, n_nodes)

    # ── 模型 ──────────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "gnn")
    node_feat_dim = model_cfg.get("node_feat_dim", 0)
    if node_feat is not None and node_feat_dim == 0:
        node_feat_dim = node_feat.shape[1]

    trainer_cfg = cfg.get("trainer", {})
    total_rounds = cfg.get("total_rounds", 100)
    update_every = trainer_cfg.get("update_every_n_rounds", 1)

    if model_type in ("random", "ground_truth"):
        model = None
        optimizer = None
        trainer = None
        scheduler = None
    elif model_type == "node_emb":
        from src.model.node_emb_model import NodeEmbModel  # noqa: PLC0415
        model = NodeEmbModel(
            n_nodes=n_nodes,
            emb_dim=model_cfg.get("emb_dim", 64),
            hidden_dim=model_cfg.get("hidden_dim", 64),
        ).to(device)
        optimizer = _build_optimizer(model, trainer_cfg)
        sched_cfg = trainer_cfg.get("scheduler", {})
        scheduler = build_scheduler(
            optimizer,
            total_steps=max(total_rounds // update_every, 1),
            warmup_steps=sched_cfg.get("warmup_rounds", 5),
            min_lr=sched_cfg.get("min_lr", 1e-5),
            strategy=sched_cfg.get("strategy", "cosine_warmup"),
            cycle_steps=sched_cfg.get("cycle_rounds", 25),
        )
        trainer = None
    elif model_type == "mlp":
        from src.baseline.mlp_link import MLPLinkScorer, extract_topo_features  # noqa: PLC0415
        _mlp_topo_dim = 3 + node_feat_dim
        model = MLPLinkScorer(in_dim=_mlp_topo_dim, hidden_dim=model_cfg.get("hidden_dim", 64)).to(device)
        optimizer = _build_optimizer(model, trainer_cfg)
        sched_cfg = trainer_cfg.get("scheduler", {})
        scheduler = build_scheduler(
            optimizer,
            total_steps=max(total_rounds // update_every, 1),
            warmup_steps=sched_cfg.get("warmup_rounds", 5),
            min_lr=sched_cfg.get("min_lr", 1e-5),
            strategy=sched_cfg.get("strategy", "cosine_warmup"),
            cycle_steps=sched_cfg.get("cycle_rounds", 25),
        )
        trainer = None
    elif model_type in _HEURISTIC_TYPES:
        model = None
        optimizer = None
        trainer = None
        scheduler = None
    elif model_type == "seal":
        from src.baseline.seal import SEALModel  # noqa: PLC0415
        model = SEALModel(
            hidden_dim=model_cfg.get("hidden_dim", 32),
            num_layers=model_cfg.get("num_layers", 2),
            label_dim=model_cfg.get("label_dim", 16),
            max_label=model_cfg.get("max_label", 50),
            scorer_hidden_dim=model_cfg.get("scorer_hidden_dim", None),
            node_feat_dim=node_feat_dim,
        ).to(device)
        optimizer = _build_optimizer(model, trainer_cfg)
        sched_cfg = trainer_cfg.get("scheduler", {})
        scheduler = build_scheduler(
            optimizer,
            total_steps=max(total_rounds // update_every, 1),
            warmup_steps=sched_cfg.get("warmup_rounds", 5),
            min_lr=sched_cfg.get("min_lr", 1e-5),
            strategy=sched_cfg.get("strategy", "cosine_warmup"),
            cycle_steps=sched_cfg.get("cycle_rounds", 25),
        )
        trainer = OnlineTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_hop=trainer_cfg.get("batch_subgraph_max_hop", 2),
            max_neighbors=trainer_cfg.get("max_neighbors", 30),
            node_feat=node_feat,
            min_batch_size=trainer_cfg.get("min_batch_size", 4),
            grad_clip=trainer_cfg.get("grad_clip", 1.0),
            score_chunk_size=trainer_cfg.get("score_chunk_size", 512),
            use_amp=trainer_cfg.get("use_amp", False),
        )
    elif model_type == "graphsage_emb":
        from src.baseline.graphsage_emb import GraphSAGEEmbModel  # noqa: PLC0415
        model = GraphSAGEEmbModel(
            n_nodes=n_nodes,
            emb_dim=model_cfg.get("emb_dim", 32),
            hidden_dim=model_cfg.get("hidden_dim", 32),
            num_layers=model_cfg.get("num_layers", 2),
            scorer_hidden_dim=model_cfg.get("scorer_hidden_dim", None),
            aggregator_type=model_cfg.get("aggregator_type", "mean"),
        ).to(device)
        optimizer = _build_optimizer(model, trainer_cfg)
        sched_cfg = trainer_cfg.get("scheduler", {})
        scheduler = build_scheduler(
            optimizer,
            total_steps=max(total_rounds // update_every, 1),
            warmup_steps=sched_cfg.get("warmup_rounds", 5),
            min_lr=sched_cfg.get("min_lr", 1e-5),
            strategy=sched_cfg.get("strategy", "cosine_warmup"),
            cycle_steps=sched_cfg.get("cycle_rounds", 25),
        )
        trainer = OnlineTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_hop=trainer_cfg.get("batch_subgraph_max_hop", 2),
            max_neighbors=trainer_cfg.get("max_neighbors", 30),
            node_feat=node_feat,
            min_batch_size=trainer_cfg.get("min_batch_size", 4),
            grad_clip=trainer_cfg.get("grad_clip", 1.0),
            score_chunk_size=trainer_cfg.get("score_chunk_size", 512),
            use_amp=trainer_cfg.get("use_amp", False),
        )
    elif model_type == "gat_emb":
        from src.baseline.gat_emb import GATEmbModel  # noqa: PLC0415
        model = GATEmbModel(
            n_nodes=n_nodes,
            emb_dim=model_cfg.get("emb_dim", 32),
            hidden_dim=model_cfg.get("hidden_dim", 32),
            num_layers=model_cfg.get("num_layers", 2),
            num_heads=model_cfg.get("num_heads", 4),
            scorer_hidden_dim=model_cfg.get("scorer_hidden_dim", None),
        ).to(device)
        optimizer = _build_optimizer(model, trainer_cfg)
        sched_cfg = trainer_cfg.get("scheduler", {})
        scheduler = build_scheduler(
            optimizer,
            total_steps=max(total_rounds // update_every, 1),
            warmup_steps=sched_cfg.get("warmup_rounds", 5),
            min_lr=sched_cfg.get("min_lr", 1e-5),
            strategy=sched_cfg.get("strategy", "cosine_warmup"),
            cycle_steps=sched_cfg.get("cycle_rounds", 25),
        )
        trainer = OnlineTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_hop=trainer_cfg.get("batch_subgraph_max_hop", 2),
            max_neighbors=trainer_cfg.get("max_neighbors", 30),
            node_feat=node_feat,
            min_batch_size=trainer_cfg.get("min_batch_size", 4),
            grad_clip=trainer_cfg.get("grad_clip", 1.0),
            score_chunk_size=trainer_cfg.get("score_chunk_size", 512),
            use_amp=trainer_cfg.get("use_amp", False),
        )
    else:
        model = LinkPredModel(
            hidden_dim=model_cfg.get("hidden_dim", 64),
            num_layers=model_cfg.get("num_layers", 3),
            scorer_hidden_dim=model_cfg.get("scorer_hidden_dim", None),
            encoder_type=model_cfg.get("encoder_type", "last"),
            node_feat_dim=node_feat_dim,
            n_nodes=n_nodes,
            node_emb_dim=model_cfg.get("node_emb_dim", 0),
        ).to(device)
        optimizer = _build_optimizer(model, trainer_cfg)
        sched_cfg = trainer_cfg.get("scheduler", {})
        scheduler = build_scheduler(
            optimizer,
            total_steps=max(total_rounds // update_every, 1),
            warmup_steps=sched_cfg.get("warmup_rounds", 5),
            min_lr=sched_cfg.get("min_lr", 1e-5),
            strategy=sched_cfg.get("strategy", "cosine_warmup"),
            cycle_steps=sched_cfg.get("cycle_rounds", 25),
        )
        trainer = OnlineTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_hop=trainer_cfg.get("batch_subgraph_max_hop", 2),
            max_neighbors=trainer_cfg.get("max_neighbors", 30),
            node_feat=node_feat,
            min_batch_size=trainer_cfg.get("min_batch_size", 4),
            grad_clip=trainer_cfg.get("grad_clip", 1.0),
            score_chunk_size=trainer_cfg.get("score_chunk_size", 512),
            use_amp=trainer_cfg.get("use_amp", False),
        )

    replay_cfg = cfg.get("replay", {})
    replay = ReplayBuffer(replay_cfg.get("capacity", 0))

    # ── 评估器 ────────────────────────────────────────────────────────────────
    eval_cfg = cfg.get("eval", {})
    evaluator = RoundMetrics(
        star_set=env.star_set,
        n_nodes=n_nodes,
        k_list=eval_cfg.get("k_list", [5, 10, 20]),
        graph_every_n=eval_cfg.get("graph_every_n", 10),
        degree_bins=eval_cfg.get("degree_bins", 50),
    )

    # 保存 config snapshot
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    top_k_rec = cfg.get("recommend", {}).get("top_k", 10)
    top_k_recall = recall_cfg.get("top_k_recall", 50)
    cold_fill = cfg.get("recommend", {}).get("cold_start_random_fill", True)
    cold_fill_k = cfg.get("recommend", {}).get("cold_start_k", top_k_recall)
    _rng = np.random.default_rng(seed + 1)

    # ── 主循环 ────────────────────────────────────────────────────────────────
    for t in range(total_rounds):
        t0 = time.time()
        recall.update_graph(t)
        U = env.sample_active_users(t)
        # 批量预计算本轮活跃用户的 PPR 向量（比逐用户快 50-100×）
        if hasattr(recall, "precompute_for_users"):
            recall.precompute_for_users(list(U))
        recs: dict[int, list[int]] = {}

        # ── Phase 1：收集所有用户候选（CPU） ────────────────────────────────
        cold_start_users: set[int] = set()
        user_cand_nodes: dict[int, list[int]] = {}
        for u in U:
            cands = recall.candidates(u, cutoff_time=float("inf"), top_k=top_k_recall)
            cands = env.mask_existing_edges(u, cands)
            cands = env.mask_cooldown(u, cands, t)

            if not cands and cold_fill:
                exclude = set(adj.out_neighbors(u)) | {u}
                exclude |= env.cooldown_excluded_nodes(u, t)
                pool = [v for v in range(n_nodes) if v not in exclude]
                if pool:
                    sample_n = min(cold_fill_k, len(pool))
                    chosen = _rng.choice(len(pool), size=sample_n, replace=False)
                    cands = [(pool[int(i)], 0.0) for i in chosen]
                    cold_start_users.add(u)

            user_cand_nodes[u] = [v for v, _ in cands] if cands else []

        # ── Phase 2：批量打分（GNN 模式下一次 GPU forward） ─────────────────
        if model_type in _GNN_LIKE_TYPES and trainer is not None:
            gnn_inputs = [(u, user_cand_nodes[u]) for u in U if user_cand_nodes[u]]
            batch_scores = trainer.score_batch(gnn_inputs, adj)
            gnn_score_map: dict[int, list[float]] = {
                u: s for (u, _), s in zip(gnn_inputs, batch_scores)
            }
        elif model_type == "node_emb":
            model.eval()
        elif model_type == "mlp":
            from src.baseline.mlp_link import extract_topo_features  # noqa: PLC0415
            # 打分用 t 轮开始时的 adj（env.step 之前）
            feat = extract_topo_features(adj, n_nodes, node_feat, device)
            _feat_edge_count = adj.num_edges()  # 记录当前边数，训练时按需重算
        elif model_type in _HEURISTIC_TYPES:
            heuristic_scores: dict[int, list[float]] = {
                u: _score_heuristic(model_type, u, user_cand_nodes[u], adj)
                for u in U if user_cand_nodes[u]
            }

        # ── Phase 3：构建推荐（top-K 精排） ─────────────────────────────────
        # 探索策略：exploit_ratio 控制 GNN top-k 保留比例，剩余从尾部候选随机补入
        # exploit_ratio=1.0 表示纯利用（默认），<1.0 时强制探索尾部候选
        _exploit_ratio = trainer_cfg.get("exploit_ratio", 1.0)
        _exploit_k = max(1, int(top_k_rec * _exploit_ratio))   # GNN 保留位数
        _explore_k = top_k_rec - _exploit_k                    # 随机探索位数
        rec_scores: dict[int, list[float]] = {}  # u -> 推荐候选的打分（与 recs[u] 对齐）
        for u in U:
            cand_nodes = user_cand_nodes[u]
            if not cand_nodes:
                recs[u] = []
                continue
            if model_type == "random":
                perm = _rng.permutation(len(cand_nodes))[:top_k_rec]
                recs[u] = [cand_nodes[int(i)] for i in perm]
                continue
            elif model_type == "ground_truth":
                _star_set = env.star_set
                true_cands = [v for v in cand_nodes if (u, v) in _star_set]
                if len(true_cands) >= top_k_rec:
                    chosen = _rng.choice(len(true_cands), top_k_rec, replace=False)
                    recs[u] = [true_cands[int(i)] for i in chosen]
                else:
                    false_cands = [v for v in cand_nodes if (u, v) not in _star_set]
                    fill_n = top_k_rec - len(true_cands)
                    if false_cands and fill_n > 0:
                        fill_n = min(fill_n, len(false_cands))
                        chosen_fill = _rng.choice(len(false_cands), fill_n, replace=False)
                        recs[u] = true_cands + [false_cands[int(i)] for i in chosen_fill]
                    else:
                        recs[u] = true_cands
                continue
            elif model_type == "node_emb":
                u_t = torch.tensor([u] * len(cand_nodes), dtype=torch.long, device=device)
                v_t = torch.tensor(cand_nodes, dtype=torch.long, device=device)
                with torch.no_grad():
                    scores = model(u_t, v_t).cpu().numpy()
            elif model_type == "mlp":
                u_feat = feat[[u]].expand(len(cand_nodes), -1)
                v_feat = feat[cand_nodes]
                with torch.no_grad():
                    scores = model(u_feat, v_feat).cpu().numpy()
            elif model_type in _HEURISTIC_TYPES:
                scores = heuristic_scores.get(u, [0.0] * len(cand_nodes))
            else:
                scores = gnn_score_map.get(u, [0.0] * len(cand_nodes))
            order = np.argsort(scores)[::-1]
            exploit_idx = order[:_exploit_k].tolist()
            if _explore_k > 0 and len(order) > _exploit_k:
                tail_idx = order[_exploit_k:]  # GNN 未选中的候选
                chosen = _rng.choice(len(tail_idx), min(_explore_k, len(tail_idx)), replace=False)
                explore_idx = tail_idx[chosen].tolist()
            else:
                explore_idx = []
            all_idx = exploit_idx + explore_idx
            recs[u] = [cand_nodes[i] for i in all_idx]
            rec_scores[u] = [float(scores[i]) for i in all_idx]

        feedback = env.step(recs, t)

        # 在线更新：冷启动用户的 rejected 不参与训练（随机负样本信噪比低）
        if model_type in _NO_MODEL_TYPES:
            train_result = {}
        elif t % update_every == 0:
            recall_rejected = [(u, v) for u, v in feedback.rejected
                               if u not in cold_start_users]
            pos_r, neg_r = replay.sample(replay_cfg.get("sample_n", 0))
            _use_oracle = trainer_cfg.get("oracle_labels", False)
            if _use_oracle:
                # Oracle 模式：用 G* 直接标注召回池中全部候选，消除假负例与选择偏差
                _star_set = env.star_set
                oracle_pos: list[tuple[int, int]] = []
                oracle_neg: list[tuple[int, int]] = []
                for _u in U:
                    if _u in cold_start_users:
                        continue
                    for _v in user_cand_nodes.get(_u, []):
                        if adj.has_edge(_u, _v) or _v == _u:
                            continue
                        if (_u, _v) in _star_set:
                            oracle_pos.append((_u, _v))
                        else:
                            oracle_neg.append((_u, _v))
                # 对负样本按 neg_ratio:1 随机下采样，防止批次爆炸
                _neg_ratio = trainer_cfg.get("oracle_neg_ratio", 4)
                _max_neg = max(len(oracle_pos) * _neg_ratio, 8)
                if len(oracle_neg) > _max_neg:
                    _idx = _rng.choice(len(oracle_neg), _max_neg, replace=False)
                    oracle_neg = [oracle_neg[int(i)] for i in _idx]
                train_pos: list[tuple[int, int]] = oracle_pos
                train_neg: list[tuple[int, int]] = oracle_neg
            else:
                train_pos = feedback.accepted + pos_r
                train_neg = recall_rejected + neg_r
            if model_type == "node_emb":
                train_result = {}
                # 只用精排推荐过去被拒绝的作为负例，避免召回池中真实正样本造成假负例
                eff_neg = train_neg
                if train_pos and eff_neg:
                    pos_u = torch.tensor([u for u, _ in train_pos], dtype=torch.long, device=device)
                    pos_v = torch.tensor([v for _, v in train_pos], dtype=torch.long, device=device)
                    neg_u = torch.tensor([u for u, _ in eff_neg], dtype=torch.long, device=device)
                    neg_v = torch.tensor([v for _, v in eff_neg], dtype=torch.long, device=device)
                    model.train()
                    logits = torch.cat([model(pos_u, pos_v), model(neg_u, neg_v)])
                    labels = torch.cat([
                        torch.ones(len(train_pos), device=device),
                        torch.zeros(len(eff_neg), device=device),
                    ])
                    loss = torch.nn.functional.binary_cross_entropy(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    _gc = trainer_cfg.get("grad_clip", 1.0)
                    if _gc > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    train_result = {"loss": loss.item()}
            elif model_type == "mlp":
                from src.baseline.mlp_link import extract_topo_features  # noqa: PLC0415
                train_result = {}
                if train_pos and train_neg:
                    # 训练用 t 轮结束后的 adj（env.step 已加入新接受边）；仅边数变化时重算
                    if adj.num_edges() != _feat_edge_count:
                        feat = extract_topo_features(adj, n_nodes, node_feat, device)
                    pos_u = torch.tensor([u for u, _ in train_pos], device=device)
                    pos_v = torch.tensor([v for _, v in train_pos], device=device)
                    neg_u = torch.tensor([u for u, _ in train_neg], device=device)
                    neg_v = torch.tensor([v for _, v in train_neg], device=device)
                    logits_pos = model(feat[pos_u], feat[pos_v])
                    logits_neg = model(feat[neg_u], feat[neg_v])
                    logits = torch.cat([logits_pos, logits_neg])
                    labels = torch.cat([
                        torch.ones(len(train_pos), device=device),
                        torch.zeros(len(train_neg), device=device),
                    ])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_result = {"loss": loss.item()}
            else:
                train_result = trainer.update(train_pos, train_neg, adj)
            replay.push(feedback.accepted, recall_rejected, t)
        else:
            train_result = {}

        metrics = evaluator.update(t, recs, feedback, adj, env.coverage(),
                                   rec_scores=rec_scores if rec_scores else None)
        metrics["replay_size"] = float(len(replay))

        # 定期释放 CUDA 缓存，防止显存碎片积累
        if torch.cuda.is_available() and t % 10 == 9:
            torch.cuda.empty_cache()

        if t % log_every == 0:
            elapsed = time.time() - t0
            auc_str = (
                (f"auc={metrics['auc_feedback']:.4f}  " if "auc_feedback" in metrics else "") +
                (f"uauc={metrics['uauc_feedback']:.4f}  " if "uauc_feedback" in metrics else "")
            )
            print(
                f"Round {t+1:>4}/{total_rounds}  "
                f"coverage={metrics['coverage']:.4f}  "
                f"prec@K={metrics.get('precision_k', 0):.4f}  "
                f"accepted={int(metrics['n_accepted'])}  "
                f"{auc_str}"
                f"loss={train_result.get('loss', float('nan')):.4f}  "
                f"({elapsed:.1f}s)",
                flush=True,
            )

    df = evaluator.history_df()
    df.to_csv(out_dir / "rounds.csv", index=False)
    print(f"\n结果已写入 {out_dir}/rounds.csv", flush=True)
    return df


if __name__ == "__main__":
    import argparse, yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_online_simulation(cfg)
