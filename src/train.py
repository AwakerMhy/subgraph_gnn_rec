"""src/train.py — 训练主循环

用法：
    conda run -n gnn python src/train.py --dataset synthetic --data_dir data/synthetic/sbm \
        --epochs 50 --batch_size 32 --lr 0.001 --hidden_dim 64 --num_layers 2 --seed 42
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.graph.edge_split import build_two_layer, filter_first_time_edges
from src.graph.negative_sampling import build_adj_out, sample_negatives, sample_negatives_mixed
from src.graph.subgraph import extract_subgraph, TimeAdjacency
from src.model.model import LinkPredModel
from src.baseline.graphsage import GraphSAGEModel
from src.baseline.seal import SEALModel
from src.baseline.tgat import TGATModel
from src.recall import build_recall
from src.utils.metrics import (
    compute_auc, compute_ap,
    compute_mrr, compute_ndcg_at_k, compute_hits_at_k,
)
from src.utils.seed import set_seed
from src.utils.split import temporal_split


def build_model(args: argparse.Namespace, node_feat_dim: int = 0) -> nn.Module:
    """根据 --model_type 实例化对应模型。"""
    if args.model_type == "gin":
        return LinkPredModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            encoder_type=args.encoder_type,
            node_feat_dim=node_feat_dim,
        )
    elif args.model_type == "graphsage":
        return GraphSAGEModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            node_feat_dim=node_feat_dim,
        )
    elif args.model_type == "seal":
        return SEALModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            node_feat_dim=node_feat_dim,
        )
    elif args.model_type == "tgat":
        return TGATModel(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    else:
        raise ValueError(f"未知 model_type: {args.model_type!r}")


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def parse_neg_strategy(s: str) -> dict[str, float]:
    """解析 --neg_strategy 参数为策略权重字典。

    支持：
      "random"                               → {"random": 1.0}
      "random:0.5,hard_2hop:0.3,degree:0.2" → {"random": 0.5, "hard_2hop": 0.3, "degree": 0.2}
    """
    if ":" not in s:
        return {s: 1.0}
    mix: dict[str, float] = {}
    for part in s.split(","):
        strat, w = part.strip().split(":")
        mix[strat.strip()] = float(w.strip())
    return mix


# ── Dataset ──────────────────────────────────────────────────────────────────

class LinkPredDataset(Dataset):
    """正负样本对数据集。

    每个样本：(u, v, cutoff_time, label)
    label=1 为正样本（真实边），label=0 为负样本。
    """

    def __init__(
        self,
        edges: pd.DataFrame,
        all_edges: pd.DataFrame,
        n_nodes: int,
        neg_ratio: int = 1,
        strategy: str = "random",
        strategy_mix: dict[str, float] | None = None,
        seed: int = 42,
        time_adj: "TimeAdjacency | None" = None,
        inductive_pool: list[int] | None = None,
    ) -> None:
        """
        strategy_mix 不为 None 时使用混合策略采样（忽略 strategy 参数）。
        strategy_mix: {strategy_name: weight}，例如 {"random": 0.5, "hard_2hop": 0.3, "degree": 0.2}
        """
        self.all_edges = all_edges
        self.n_nodes = n_nodes
        self.neg_ratio = neg_ratio
        self.seed = seed

        # 全时段出边邻接表：排除 u 在整个数据集中曾连接过的节点，防止假负样本
        all_time_adj_out, _ = build_adj_out(all_edges, cutoff_time=None)

        use_mixed = strategy_mix is not None and len(strategy_mix) > 1

        # 构造正样本列表
        self.samples: list[tuple[int, int, float, int]] = []
        for u_val, v_val, t_val in zip(
            edges["src"].to_numpy(), edges["dst"].to_numpy(), edges["timestamp"].to_numpy()
        ):
            u, v, t = int(u_val), int(v_val), float(t_val)
            self.samples.append((u, v, t, 1))
            if use_mixed:
                neg_vs = sample_negatives_mixed(
                    u, t, all_edges, n_nodes,
                    strategy_mix=strategy_mix, k=neg_ratio, seed=seed,
                    all_time_adj_out=all_time_adj_out,
                    time_adj=time_adj,
                    inductive_pool=inductive_pool,
                )
            else:
                neg_vs = sample_negatives(
                    u, t, all_edges, n_nodes,
                    strategy=strategy, k=neg_ratio, seed=seed,
                    all_time_adj_out=all_time_adj_out,
                    time_adj=time_adj,
                    inductive_pool=inductive_pool,
                )
            for nv in neg_vs:
                self.samples.append((u, nv, t, 0))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]


class RecallDataset(Dataset):
    """simulated_recall 协议的候选集数据集。

    每个样本：(u, v, cutoff_time, label, query_id)
        label=1  → v ∈ E_hidden（正样本）
        label=0  → v ∉ E_all（真负样本）
        discard  → v ∈ E_obs（已知关系，不加入样本）

    query_id 用于 eval_mrr_epoch 按 u 分组计算 MRR / NDCG。
    """

    def __init__(
        self,
        hidden_edges: pd.DataFrame,
        e_obs_pairs: set[tuple[int, int]],
        e_all_pairs: set[tuple[int, int]],
        recall,
        cutoff_time: float,
        top_k: int,
        n_nodes: int,
        rng_seed: int = 42,
        reciprocity_weights: "dict[tuple[int,int], float] | None" = None,
        difficulty_range: "tuple[int, int] | None" = None,
    ) -> None:
        self.samples: list[tuple] = []
        self._recip_weights = reciprocity_weights or {}
        self._diff_range = difficulty_range
        self._build(hidden_edges, e_obs_pairs, e_all_pairs,
                    recall, cutoff_time, top_k, n_nodes, rng_seed)

    def _build(self, hidden_edges, e_obs_pairs, e_all_pairs,
               recall, cutoff_time, top_k, n_nodes, rng_seed):
        rng = np.random.default_rng(rng_seed)
        pos_by_u: dict[int, set[int]] = {}
        for u, v in zip(hidden_edges["src"].tolist(), hidden_edges["dst"].tolist()):
            pos_by_u.setdefault(int(u), set()).add(int(v))

        use_weights = bool(self._recip_weights)
        diff_start, diff_end = self._diff_range if self._diff_range else (0, top_k)

        query_id = 0
        for u, pos_vs in pos_by_u.items():
            cands = recall.candidates(u, cutoff_time, top_k)
            # curriculum: slice candidates to [diff_start, diff_end) for negatives
            cands_neg_pool = cands[diff_start:diff_end]
            positives, negatives = [], []
            # positives come from full candidate list regardless of difficulty
            for v, _ in cands:
                if v in pos_vs:
                    positives.append(v)
            for v, _ in cands_neg_pool:
                if v not in pos_vs and (u, v) not in e_all_pairs:
                    negatives.append(v)

            # 候选池不足时随机补充负样本
            if not negatives:
                attempts = 0
                while len(negatives) < max(1, top_k // 4) and attempts < top_k * 4:
                    rv = int(rng.integers(0, n_nodes))
                    if rv != u and (u, rv) not in e_all_pairs:
                        negatives.append(rv)
                    attempts += 1

            if not positives or not negatives:
                continue

            for v in positives:
                w = self._recip_weights.get((u, v), 1.0) if use_weights else 1.0
                if use_weights:
                    self.samples.append((u, v, cutoff_time, 1, query_id, w))
                else:
                    self.samples.append((u, v, cutoff_time, 1, query_id))
            for v in negatives:
                if use_weights:
                    self.samples.append((u, v, cutoff_time, 0, query_id, 1.0))
                else:
                    self.samples.append((u, v, cutoff_time, 0, query_id))
            query_id += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]


class EgoCNOfflineDataset(Dataset):
    """ego_cn_offline 协议数据集。

    正样本：E_hidden 中存在的边 (u→v)。
    负样本：(u→v) 满足 (u,v) ∉ exclusion_pairs，其中
        exclusion_pairs = E_obs pairs ∪ E_hidden pairs（当前切分期）。
    每个样本：(u, v, cutoff_time, label, query_id)，兼容 collate_fn 和 eval_mrr_epoch。
    """

    def __init__(
        self,
        hidden_edges: pd.DataFrame,
        exclusion_pairs: set[tuple[int, int]],
        n_nodes: int,
        cutoff_time: float,
        neg_ratio: int = 1,
        seed: int = 42,
        max_samples: int = 0,
    ) -> None:
        self.samples: list[tuple] = []
        self._build(hidden_edges, exclusion_pairs, n_nodes, cutoff_time, neg_ratio, seed, max_samples)

    def _build(self, hidden_edges, exclusion_pairs, n_nodes, cutoff_time, neg_ratio, seed, max_samples):
        rng = np.random.default_rng(seed)

        # 按 u 分组收集正样本
        pos_by_u: dict[int, list[int]] = {}
        for u, v in zip(hidden_edges["src"].tolist(), hidden_edges["dst"].tolist()):
            pos_by_u.setdefault(int(u), []).append(int(v))

        all_nodes = list(range(n_nodes))
        query_id = 0

        for u, pos_vs in pos_by_u.items():
            if max_samples > 0 and len(self.samples) >= max_samples:
                break

            n_neg = len(pos_vs) * neg_ratio
            negatives: list[int] = []
            attempts = 0
            max_attempts = n_neg * 200
            while len(negatives) < n_neg and attempts < max_attempts:
                v_neg = int(rng.integers(0, n_nodes))
                if v_neg != u and (u, v_neg) not in exclusion_pairs:
                    negatives.append(v_neg)
                attempts += 1

            if not pos_vs or not negatives:
                continue

            for v in pos_vs:
                self.samples.append((u, v, cutoff_time, 1, query_id))
            for v in negatives:
                self.samples.append((u, v, cutoff_time, 0, query_id))
            query_id += 1

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]


def collate_fn(
    batch: list[tuple],
    all_edges: pd.DataFrame,
    max_hop: int,
    max_neighbors: int,
    seed: int,
    store_edge_time: bool = False,
    time_adj: "TimeAdjacency | None" = None,
    node_feat: "torch.Tensor | None" = None,
    subgraph_type: str = "ego_cn",
) -> tuple:
    """将一个 batch 的 (u, v, t, label[, query_id[, weight]]) 打包为批量图。

    Returns:
        (bg, labels, query_ids_or_None, sample_weights_or_None)
    """
    graphs, labels, query_ids, weights = [], [], [], []
    has_qid    = len(batch[0]) > 4
    has_weight = len(batch[0]) > 5
    for item in batch:
        u, v, t, label = item[0], item[1], item[2], item[3]
        qid = item[4] if has_qid else -1
        w   = float(item[5]) if has_weight else 1.0
        g = extract_subgraph(
            u, v, t, all_edges, max_hop,
            max_neighbors_per_node=max_neighbors, seed=seed,
            store_edge_time=store_edge_time,
            time_adj=time_adj,
            node_feat=node_feat,
            subgraph_type=subgraph_type,
        )
        if g is not None:
            graphs.append(g)
            labels.append(label)
            query_ids.append(qid)
            weights.append(w)
    if not graphs:
        return None, None, None, None
    bg = dgl.batch(graphs)
    sample_weights = torch.tensor(weights, dtype=torch.float32) if has_weight else None
    return bg, torch.tensor(labels, dtype=torch.float32), query_ids if has_qid else None, sample_weights


# ── 训练 / 验证 ───────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    all_edges: pd.DataFrame,
    max_hop: int = 2,
    max_neighbors: int = 30,
    seed: int = 42,
    store_edge_time: bool = False,
    time_adj: "TimeAdjacency | None" = None,
    node_feat: "torch.Tensor | None" = None,
    subgraph_type: str = "ego_cn",
) -> tuple[float, float, float]:
    """单 epoch 训练或验证。

    time_adj 不为 None 时，每个样本的子图提取走 TimeAdjacency 快路径。

    Returns:
        (loss, auc, ap)
    """
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.BCELoss()

    all_scores, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    criterion_none = nn.BCELoss(reduction='none')

    for batch in loader:
        bg, labels, _, sample_weights = collate_fn(
            batch, all_edges, max_hop, max_neighbors, seed,
            store_edge_time=store_edge_time,
            time_adj=time_adj,
            node_feat=node_feat,
            subgraph_type=subgraph_type,
        )
        if bg is None:
            continue

        bg = bg.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            scores = model.forward_batch(bg)
            if sample_weights is not None:
                loss = (criterion_none(scores, labels) * sample_weights.to(device)).mean()
            else:
                loss = criterion(scores, labels)

        if is_train:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1
        all_scores.append(scores.detach().cpu())
        all_labels.append(labels.detach().cpu())

    all_scores = torch.cat(all_scores).numpy()
    all_labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / max(n_batches, 1)
    auc = compute_auc(all_labels, all_scores)
    ap = compute_ap(all_labels, all_scores)
    return avg_loss, auc, ap


# ── MRR 评估（simulated_recall 协议专用）─────────────────────────────────────

def eval_mrr_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    all_edges: pd.DataFrame,
    max_hop: int = 2,
    max_neighbors: int = 30,
    seed: int = 42,
    time_adj: "TimeAdjacency | None" = None,
    node_feat: "torch.Tensor | None" = None,
    k_list: list[int] | None = None,
    subgraph_type: str = "ego_cn",
) -> dict[str, float]:
    """按 query 分组计算 MRR / NDCG@K / Hits@K（simulated_recall 协议）。"""
    if k_list is None:
        k_list = [10, 20, 50]

    model.eval()
    per_query: dict[int, dict[str, list]] = {}

    with torch.no_grad():
        for batch in loader:
            bg, labels, query_ids, _ = collate_fn(
                batch, all_edges, max_hop, max_neighbors, seed,
                time_adj=time_adj, node_feat=node_feat,
                subgraph_type=subgraph_type,
            )
            if bg is None:
                continue
            scores = model.forward_batch(bg.to(device)).cpu().numpy()
            for qid, score, label in zip(query_ids, scores, labels.numpy()):
                if qid not in per_query:
                    per_query[qid] = {"pos": [], "neg": []}
                if int(label) == 1:
                    per_query[qid]["pos"].append(float(score))
                else:
                    per_query[qid]["neg"].append(float(score))

    pos_rows: list[float] = []
    neg_rows: list[list[float]] = []
    for v in per_query.values():
        if not v["pos"] or not v["neg"]:
            continue
        for p in v["pos"]:
            pos_rows.append(p)
            neg_rows.append(v["neg"])

    if not pos_rows:
        return {"mrr": 0.0, **{f"ndcg@{k}": 0.0 for k in k_list},
                **{f"hits@{k}": 0.0 for k in k_list}}

    pos_arr = np.array(pos_rows, dtype=np.float32)
    max_neg = max(len(n) for n in neg_rows)
    neg_mat = np.zeros((len(neg_rows), max_neg), dtype=np.float32)
    for i, n in enumerate(neg_rows):
        neg_mat[i, :len(n)] = n

    result: dict[str, float] = {"mrr": compute_mrr(pos_arr, neg_mat)}
    for k in k_list:
        result[f"ndcg@{k}"] = compute_ndcg_at_k(pos_arr, neg_mat, k)
        result[f"hits@{k}"] = compute_hits_at_k(pos_arr, neg_mat, k)
    return result


# ── simulated_recall 协议实现 ─────────────────────────────────────────────────

def _run_simulated_recall(
    args: argparse.Namespace,
    edges: pd.DataFrame,
    n_nodes: int,
    node_feat: "torch.Tensor | None",
    node_feat_dim: int,
    device: torch.device,
) -> None:
    """simulated_recall 协议完整训练流程。"""
    # Step 1: 首次边过滤（可选）
    if args.first_time_only:
        edges = filter_first_time_edges(edges)
        print(f"首次边过滤后: {len(edges)} 条边", flush=True)

    # Step 3: 两层图构造
    cfg_split = {
        "edge_split": {
            "strategy": args.edge_split_strategy,
            "mask_ratio_val": 0.15,
            "mask_ratio_test": 0.15,
        }
    }
    two_layer = build_two_layer(edges, cfg_split)
    E_obs = two_layer.E_obs
    E_hidden_train = two_layer.E_hidden_val   # 训练目标
    E_hidden_val   = two_layer.E_hidden_test  # 验证/选模型目标
    print(
        f"两层图: E_obs={len(E_obs)}  E_hidden_train={len(E_hidden_train)}  "
        f"E_hidden_val={len(E_hidden_val)}",
        flush=True,
    )

    # 反泄露断言：E_obs 与 E_hidden 不重叠
    obs_pairs  = set(zip(E_obs["src"].tolist(), E_obs["dst"].tolist()))
    htr_pairs  = set(zip(E_hidden_train["src"].tolist(), E_hidden_train["dst"].tolist()))
    hval_pairs = set(zip(E_hidden_val["src"].tolist(), E_hidden_val["dst"].tolist()))
    assert obs_pairs.isdisjoint(htr_pairs),  "E_obs ∩ E_hidden_train ≠ ∅ — 数据泄露！"
    assert obs_pairs.isdisjoint(hval_pairs), "E_obs ∩ E_hidden_val ≠ ∅ — 数据泄露！"

    # !! 关键：TimeAdjacency 只基于 E_obs 构建，绝对不能用全量 edges
    print("构建时序邻接表（仅 E_obs）...", flush=True)
    time_adj = TimeAdjacency(E_obs)
    assert len(time_adj._out) <= n_nodes, "TimeAdjacency 节点数超出预期"
    print("时序邻接表构建完成", flush=True)

    # Step 4: 构建召回模型
    cfg_recall = {"method": args.recall_method, "top_k": args.recall_top_k}
    recall = build_recall(cfg_recall, time_adj, n_nodes)

    # 全量边对集合：用于 RecallDataset 的标签分配（避免假负样本）
    e_all_pairs = obs_pairs | htr_pairs | hval_pairs

    # Step 5: 互惠性加权（可选）
    recip_weights: dict[tuple[int, int], float] | None = None
    if args.reciprocity_weighting:
        from src.graph.edge_split import compute_reciprocity_labels
        all_recip = compute_reciprocity_labels(edges)
        recip_weights = {
            pair: (args.reciprocity_bidir_weight if is_bidir else args.reciprocity_unidir_weight)
            for pair, is_bidir in all_recip.items()
        }
        print(f"互惠性加权: bidir={args.reciprocity_bidir_weight}  unidir={args.reciprocity_unidir_weight}", flush=True)

    # Step 8: Curriculum 调度器（可选）
    curriculum: "CurriculumScheduler | None" = None
    if args.curriculum:
        from src.recall.curriculum import CurriculumScheduler
        curriculum = CurriculumScheduler(
            total_epochs=args.epochs,
            schedule=args.curriculum_schedule,
            warmup_epochs=args.curriculum_warmup,
        )
        print(f"课程学习: schedule={args.curriculum_schedule}  warmup={args.curriculum_warmup}", flush=True)

    # ── 验证集固定不变，训练集按 epoch 重建（仅 curriculum 模式）─────────────
    def _make_train_ds(epoch: int = 1) -> RecallDataset:
        diff_range = curriculum.top_k_range(epoch, args.recall_top_k) if curriculum else None
        return RecallDataset(
            hidden_edges=E_hidden_train,
            e_obs_pairs=obs_pairs,
            e_all_pairs=e_all_pairs,
            recall=recall,
            cutoff_time=float(two_layer.cutoff_val),
            top_k=args.recall_top_k,
            n_nodes=n_nodes,
            rng_seed=args.seed,
            reciprocity_weights=recip_weights,
            difficulty_range=diff_range,
        )

    print("构建 RecallDataset（训练集）...", flush=True)
    train_ds = _make_train_ds(epoch=1)
    print(f"训练集 {len(train_ds)} 样本", flush=True)

    print("构建 RecallDataset（验证集）...", flush=True)
    val_ds = RecallDataset(
        hidden_edges=E_hidden_val,
        e_obs_pairs=obs_pairs,
        e_all_pairs=e_all_pairs,
        recall=recall,
        cutoff_time=float(two_layer.cutoff_test),
        top_k=args.recall_top_k,
        n_nodes=n_nodes,
        rng_seed=args.seed + 1,
    )
    print(f"验证集 {len(val_ds)} 样本", flush=True)

    if len(train_ds) == 0 or len(val_ds) == 0:
        print("警告：训练集或验证集为空，检查数据集或召回配置", flush=True)
        return

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)

    # 模型 & 优化器
    model = build_model(args, node_feat_dim=node_feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 日志目录
    run_dir = Path("results/logs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("results/checkpoints") / f"{args.run_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_records: list[dict] = []

    best_val_mrr = 0.0
    no_improve = 0
    k_list = [10, 20, 50]

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # curriculum: 每 epoch 重建训练集（难度渐进）
        if curriculum and epoch > 1:
            train_ds = _make_train_ds(epoch=epoch)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: b)

        tr_loss, tr_auc, tr_ap = run_epoch(
            model, train_loader, optimizer, device,
            all_edges=E_obs, max_hop=args.max_hop,
            max_neighbors=args.max_neighbors, seed=args.seed,
            time_adj=time_adj, node_feat=node_feat,
            subgraph_type=args.subgraph_type,
        )
        val_metrics = eval_mrr_epoch(
            model, val_loader, device,
            all_edges=E_obs, max_hop=args.max_hop,
            max_neighbors=args.max_neighbors, seed=args.seed,
            time_adj=time_adj, node_feat=node_feat,
            k_list=k_list,
            subgraph_type=args.subgraph_type,
        )
        val_mrr = val_metrics["mrr"]
        elapsed = time.time() - t0

        diff_str = f"  difficulty={curriculum.difficulty(epoch):.2f}" if curriculum else ""
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items())
        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  tr_auc={tr_auc:.4f}  "
            f"{metrics_str}{diff_str}  ({elapsed:.1f}s)"
        )

        rec = {"epoch": epoch, "tr_loss": tr_loss, "tr_auc": tr_auc, **val_metrics}
        if curriculum:
            rec["difficulty"] = curriculum.difficulty(epoch)
        log_records.append(rec)
        with open(run_dir / "train.json", "w") as f:
            json.dump(log_records, f, indent=2)

        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            no_improve = 0
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_mrr": val_mrr, **{k: v for k, v in val_metrics.items()},
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"早停：val MRR 连续 {args.patience} epoch 未提升")
                break

    print(f"训练完成。最佳 val MRR: {best_val_mrr:.4f}  checkpoint: {ckpt_path}")


# ── ego_cn_offline 协议实现 ───────────────────────────────────────────────────

def _run_ego_cn_offline(
    args: argparse.Namespace,
    edges: pd.DataFrame,
    n_nodes: int,
    node_feat: "torch.Tensor | None",
    node_feat_dim: int,
    device: torch.device,
) -> None:
    """ego_cn_offline 协议完整训练流程。

    切分：70% E_obs / 中15% 训练 / 后15% 测试。
    正样本：E_hidden 中存在的边；负样本：排除 E_obs ∪ E_hidden 后随机采样。
    子图：ego_cn，背景图始终为 E_obs（TimeAdjacency）。
    """
    from src.graph.edge_split import temporal_mask_split

    # 首次边过滤（可选）
    if args.first_time_only:
        edges = filter_first_time_edges(edges)
        print(f"首次边过滤后: {len(edges)} 条边", flush=True)

    # 70/15/15 时间切分
    split = temporal_mask_split(edges)
    E_obs        = split.E_obs
    E_hidden_tr  = split.E_hidden_val   # 中15%，训练集
    E_hidden_te  = split.E_hidden_test  # 后15%，测试集
    print(
        f"切分: E_obs={len(E_obs)}  train={len(E_hidden_tr)}  test={len(E_hidden_te)}",
        flush=True,
    )

    # 反泄露断言
    obs_pairs = set(zip(E_obs["src"].tolist(), E_obs["dst"].tolist()))
    htr_pairs = set(zip(E_hidden_tr["src"].tolist(), E_hidden_tr["dst"].tolist()))
    hte_pairs = set(zip(E_hidden_te["src"].tolist(), E_hidden_te["dst"].tolist()))
    assert obs_pairs.isdisjoint(htr_pairs), "E_obs ∩ E_hidden_tr ≠ ∅ — 数据泄露！"
    assert obs_pairs.isdisjoint(hte_pairs), "E_obs ∩ E_hidden_te ≠ ∅ — 数据泄露！"

    # 训练背景图：E_obs（前70%）
    # 测试背景图：E_obs ∪ E_hidden_tr（前85%）——测试时中15%已写入网络
    print("构建时序邻接表（训练背景: E_obs）...", flush=True)
    time_adj_tr = TimeAdjacency(E_obs)
    print("构建时序邻接表（测试背景: E_obs ∪ 中15%）...", flush=True)
    E_obs_plus_tr = pd.concat([E_obs, E_hidden_tr], ignore_index=True).sort_values("timestamp")
    time_adj_te = TimeAdjacency(E_obs_plus_tr)
    print("时序邻接表构建完成", flush=True)

    # cutoff_time = 2.0：高于所有归一化时间戳（[0,1]），确保 TimeAdjacency 内全量边可见
    _CUTOFF = 2.0

    # 构建排除集：负样本不得出现在 E_obs 或当前切分期 E_hidden 中
    train_excl = obs_pairs | htr_pairs
    test_excl  = obs_pairs | htr_pairs | hte_pairs  # 测试负样本额外排除中15%

    print("构建训练集...", flush=True)
    train_ds = EgoCNOfflineDataset(
        hidden_edges=E_hidden_tr,
        exclusion_pairs=train_excl,
        n_nodes=n_nodes,
        cutoff_time=_CUTOFF,
        neg_ratio=args.neg_ratio,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    print(f"训练集 {len(train_ds)} 样本", flush=True)

    print("构建测试集...", flush=True)
    test_ds = EgoCNOfflineDataset(
        hidden_edges=E_hidden_te,
        exclusion_pairs=test_excl,
        n_nodes=n_nodes,
        cutoff_time=_CUTOFF,
        neg_ratio=args.neg_ratio,
        seed=args.seed + 1,
        max_samples=args.max_samples,
    )
    print(f"测试集 {len(test_ds)} 样本", flush=True)

    if len(train_ds) == 0 or len(test_ds) == 0:
        print("警告：训练集或测试集为空，检查数据集配置", flush=True)
        return

    # collate_fn=lambda: 透传原始 tuple 列表，由训练循环 / eval_mrr_epoch 自行调用 collate_fn
    _passthrough = lambda batch: batch  # noqa: E731
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_passthrough, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              collate_fn=_passthrough, num_workers=0)

    # 模型与优化器
    model = build_model(args, node_feat_dim=node_feat_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    criterion = nn.BCELoss()

    # 结果目录
    run_dir = Path("results/offline_ego_cn") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"
    log_records: list[dict] = []
    best_test_mrr = 0.0

    print(f"开始训练，共 {args.epochs} epoch，设备: {device}", flush=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, total_auc = 0.0, 0.0
        n_batches = 0

        for batch in train_loader:
            bg, labels, _, _ = collate_fn(
                batch, E_obs,
                max_hop=args.max_hop, max_neighbors=args.max_neighbors, seed=args.seed,
                time_adj=time_adj_tr, node_feat=node_feat, subgraph_type="ego_cn",
            )
            if bg is None:
                continue
            bg = bg.to(device)
            labels_f = labels.float().to(device)
            scores = model.forward_batch(bg)
            loss = criterion(scores, labels_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            auc = compute_auc(
                labels.numpy(),
                scores.detach().cpu().numpy(),
            )
            total_auc += auc
            n_batches += 1

        tr_loss = total_loss / max(n_batches, 1)
        tr_auc  = total_auc  / max(n_batches, 1)

        # 测试集 MRR（每 epoch 评估，小数据集可接受；大数据集可改为每 N epoch）
        test_metrics = eval_mrr_epoch(
            model, test_loader, device, E_obs_plus_tr,
            max_hop=args.max_hop, max_neighbors=args.max_neighbors,
            seed=args.seed, time_adj=time_adj_te, node_feat=node_feat,
            k_list=[5, 10, 20], subgraph_type="ego_cn",
        )
        test_mrr = test_metrics.get("mrr", 0.0)

        elapsed = time.time() - t0
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in test_metrics.items())
        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  tr_auc={tr_auc:.4f}  "
            f"{metrics_str}  ({elapsed:.1f}s)",
            flush=True,
        )

        rec = {"epoch": epoch, "tr_loss": tr_loss, "tr_auc": tr_auc, **test_metrics}
        log_records.append(rec)
        with open(run_dir / "train.json", "w") as f:
            json.dump(log_records, f, indent=2)

        if test_mrr > best_test_mrr:
            best_test_mrr = test_mrr
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "test_mrr": test_mrr, **test_metrics}, ckpt_path)

    print(f"训练完成。最佳 test MRR: {best_test_mrr:.4f}  checkpoint: {ckpt_path}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="processed 数据目录（含 edges.csv / meta.json）")
    parser.add_argument("--run_name", type=str, default="run",
                        help="实验名称，用于保存日志和 checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_hop", type=int, default=2)
    parser.add_argument("--max_neighbors", type=int, default=30)
    parser.add_argument("--subgraph_type", type=str, default="ego_cn",
                        choices=["ego_cn", "bfs_2hop"],
                        help="子图构建策略：ego_cn={u}∪N(u)∪CN(u,v)∪{v}；bfs_2hop=BFS(u)∪BFS(v)")
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--neg_strategy", type=str, default="random:0.5,hard_2hop:0.3,degree:0.2",
                        help="训练负样本策略。单策略: 'random'；混合策略: 'random:0.5,hard_2hop:0.3,degree:0.2'")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="训练/验证集最大样本数（0=不限制，用于 smoke test）")
    parser.add_argument("--patience", type=int, default=10,
                        help="早停 patience（val AUC 不提升的 epoch 数）")
    parser.add_argument("--encoder_type", type=str, default="last",
                        choices=["last", "layer_concat", "layer_sum"],
                        help="GIN 编码器类型（仅 model_type=gin 时生效）")
    parser.add_argument("--model_type", type=str, default="gin",
                        choices=["gin", "graphsage", "seal", "tgat"],
                        help="模型类型：gin / graphsage / seal / tgat")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # ── simulated_recall 协议参数 ──────────────────────────────────────────────
    parser.add_argument("--protocol", type=str, default="legacy",
                        choices=["legacy", "simulated_recall", "ego_cn_offline"],
                        help="训练协议：legacy / simulated_recall / ego_cn_offline")
    parser.add_argument("--first_time_only", action="store_true",
                        help="是否过滤为首次边（每 (u,v) 对只保留最早一条）")
    parser.add_argument("--edge_split_strategy", type=str, default="temporal",
                        choices=["temporal", "random"],
                        help="两层图切分策略")
    parser.add_argument("--recall_method", type=str, default="common_neighbors",
                        choices=["common_neighbors", "adamic_adar", "union"],
                        help="召回模型类型")
    parser.add_argument("--recall_top_k", type=int, default=100,
                        help="每个查询节点的召回候选数")
    # ── 互惠性加权（Step 5）──────────────────────────────────────────────────
    parser.add_argument("--reciprocity_weighting", action="store_true",
                        help="启用互惠性样本加权（正样本中双向边权重更高）")
    parser.add_argument("--reciprocity_bidir_weight", type=float, default=2.0)
    parser.add_argument("--reciprocity_unidir_weight", type=float, default=1.0)
    # ── Curriculum Learning（Step 8）─────────────────────────────────────────
    parser.add_argument("--curriculum", action="store_true",
                        help="启用课程学习：负样本难度随 epoch 线性提升")
    parser.add_argument("--curriculum_schedule", type=str, default="linear",
                        choices=["linear", "cosine", "step"])
    parser.add_argument("--curriculum_warmup", type=int, default=0,
                        help="前 N epoch 保持最低难度（easy 阶段）")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    edges = pd.read_csv(data_dir / "edges.csv")
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n_nodes = meta["n_nodes"]

    edges = edges.sort_values("timestamp").reset_index(drop=True)
    train_edges, val_edges, test_edges = temporal_split(edges)

    # smoke test：限制样本数
    if args.max_samples > 0:
        train_edges = train_edges.head(args.max_samples).reset_index(drop=True)
        val_edges   = val_edges.head(args.max_samples // 4).reset_index(drop=True)

    print(f"数据集: {data_dir.name}  节点={n_nodes}  "
          f"训练={len(train_edges)}  验证={len(val_edges)}  测试={len(test_edges)}")

    # ── 加载节点属性（nodes.csv）──────────────────────────────────────────────
    nodes_path = data_dir / "nodes.csv"
    node_feat: torch.Tensor | None = None
    node_feat_dim = 0
    if nodes_path.exists():
        nodes_df = pd.read_csv(nodes_path)
        feat_cols = [c for c in nodes_df.columns if c != "node_id"]
        if feat_cols:
            node_feat = torch.tensor(
                nodes_df.sort_values("node_id")[feat_cols].to_numpy(),
                dtype=torch.float32,
            )
            node_feat_dim = len(feat_cols)
            print(f"节点属性: {feat_cols}  维度={node_feat_dim}", flush=True)

    # ── simulated_recall 协议分支 ─────────────────────────────────────────────
    if args.protocol == "simulated_recall":
        _run_simulated_recall(args, edges, n_nodes, node_feat, node_feat_dim, device)
        return

    # ── ego_cn_offline 协议分支 ───────────────────────────────────────────────
    if args.protocol == "ego_cn_offline":
        _run_ego_cn_offline(args, edges, n_nodes, node_feat, node_feat_dim, device)
        return

    # ── 构建 TimeAdjacency（一次性预构建，供所有 epoch 复用）──────────────────
    print("构建时序邻接表...", flush=True)
    time_adj = TimeAdjacency(edges)
    print("时序邻接表构建完成", flush=True)

    # ── 解析训练负样本策略 ────────────────────────────────────────────────────
    neg_strategy_mix = parse_neg_strategy(args.neg_strategy)
    is_mixed = len(neg_strategy_mix) > 1
    print(f"训练负样本策略: {neg_strategy_mix}", flush=True)

    # ── 预计算 inductive_pool（训练期从未出现的节点）────────────────────────
    train_nodes = set(train_edges["src"].tolist()) | set(train_edges["dst"].tolist())
    inductive_pool = [v for v in range(n_nodes) if v not in train_nodes]
    print(f"新实体池大小: {len(inductive_pool)}", flush=True)

    # ── 构建训练集（混合策略）────────────────────────────────────────────────
    print("构建训练集...", flush=True)
    train_ds = LinkPredDataset(
        train_edges, edges, n_nodes, args.neg_ratio,
        strategy=args.neg_strategy if not is_mixed else "random",
        strategy_mix=neg_strategy_mix if is_mixed else None,
        seed=args.seed,
        time_adj=time_adj,
        inductive_pool=inductive_pool,
    )
    print(f"训练集 {len(train_ds)} 样本", flush=True)

    # ── 构建三路验证集（分策略独立评估）─────────────────────────────────────
    # random: 基础判别能力上界
    # hard_2hop: 结构区分能力下界
    # historical: 历史续期识别能力
    print("构建验证集（hard_2hop / historical）...", flush=True)
    val_ds_hard2hop = LinkPredDataset(
        val_edges, edges, n_nodes, args.neg_ratio,
        strategy="hard_2hop", seed=args.seed,
        time_adj=time_adj,
    )
    val_ds_historical = LinkPredDataset(
        val_edges, edges, n_nodes, args.neg_ratio,
        strategy="historical", seed=args.seed,
        time_adj=time_adj,
    )
    print(f"验证集大小: {len(val_ds_hard2hop)} 样本（各策略相同）", flush=True)

    train_loader      = DataLoader(train_ds,         batch_size=args.batch_size, shuffle=True,  collate_fn=lambda b: b)
    val_loader_hard2h = DataLoader(val_ds_hard2hop,  batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)
    val_loader_hist   = DataLoader(val_ds_historical, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: b)

    # ── 模型 & 优化器 ─────────────────────────────────────────────────────────
    model = build_model(args, node_feat_dim=node_feat_dim).to(device)
    store_edge_time = (args.model_type == "tgat")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    # ── 日志目录 ──────────────────────────────────────────────────────────────
    run_dir = Path("results/logs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("results/checkpoints") / f"{args.run_name}_best.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_records = []

    best_val_auc = 0.0
    no_improve = 0

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_auc, tr_ap = run_epoch(
            model, train_loader, optimizer, device,
            all_edges=edges, max_hop=args.max_hop,
            max_neighbors=args.max_neighbors, seed=args.seed,
            store_edge_time=store_edge_time, time_adj=time_adj,
            node_feat=node_feat, subgraph_type=args.subgraph_type,
        )
        _kw = dict(all_edges=edges, max_hop=args.max_hop,
                   max_neighbors=args.max_neighbors, seed=args.seed,
                   store_edge_time=store_edge_time, time_adj=time_adj,
                   node_feat=node_feat, subgraph_type=args.subgraph_type)
        _, val_auc_hard,  val_ap_hard  = run_epoch(model, val_loader_hard2h, None, device, **_kw)
        _, val_auc_hist,  val_ap_hist  = run_epoch(model, val_loader_hist,   None, device, **_kw)
        val_auc_mean = (val_auc_hard + val_auc_hist) / 2
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  loss={tr_loss:.4f}  tr_auc={tr_auc:.4f}  "
            f"val_hard2h={val_auc_hard:.4f}  val_hist={val_auc_hist:.4f}  mean={val_auc_mean:.4f}  "
            f"({elapsed:.1f}s)"
        )

        rec = dict(
            epoch=epoch, tr_loss=tr_loss, tr_auc=tr_auc,
            val_auc_hard2hop=val_auc_hard, val_ap_hard2hop=val_ap_hard,
            val_auc_historical=val_auc_hist, val_ap_historical=val_ap_hist,
            val_auc_mean=val_auc_mean,
        )
        log_records.append(rec)
        with open(run_dir / "train.json", "w") as f:
            json.dump(log_records, f, indent=2)

        # checkpoint 以 hard_2hop + historical 均值为准
        if val_auc_mean > best_val_auc:
            best_val_auc = val_auc_mean
            no_improve = 0
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_auc_hard2hop": val_auc_hard,
                        "val_auc_historical": val_auc_hist,
                        "val_auc_mean": val_auc_mean}, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"早停：val AUC(mean) 连续 {args.patience} epoch 未提升")
                break

    print(f"训练完成。最佳 val AUC(mean): {best_val_auc:.4f}  checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
