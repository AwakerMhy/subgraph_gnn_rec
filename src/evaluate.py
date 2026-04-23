"""src/evaluate.py — 测试集评估

用法：
    conda run -n gnn python src/evaluate.py \
        --data_dir data/synthetic/sbm \
        --ckpt results/checkpoints/run_best.pt \
        --hidden_dim 64 --num_layers 2
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
from torch.utils.data import DataLoader

from src.graph.negative_sampling import build_adj_out, sample_negatives
from src.graph.subgraph import build_graph_adj, extract_subgraph
from src.model.model import LinkPredModel
from src.utils.metrics import compute_all_metrics
from src.utils.seed import set_seed
from src.utils.split import temporal_split


def evaluate(
    model: LinkPredModel,
    test_edges: pd.DataFrame,
    all_edges: pd.DataFrame,
    n_nodes: int,
    max_hop: int,
    max_neighbors: int,
    neg_ratio: int,
    hits_neg_per_pos: int,
    seed: int,
    device: torch.device,
    batch_size: int = 32,
    prebuilt_adj_out: "dict | None" = None,
    prebuilt_adj_in: "dict | None" = None,
) -> dict[str, float]:
    """在测试集上计算 AUC / AP / Hits@K。

    Hits@K 评估方式：每个正样本 vs. hits_neg_per_pos 个随机负样本。
    """
    model.eval()

    # 预构建邻接表（若未提供则用测试截止时刻）
    if prebuilt_adj_out is None or prebuilt_adj_in is None:
        test_cutoff = float(test_edges["timestamp"].max()) + 1e-9
        prebuilt_adj_out, prebuilt_adj_in = build_graph_adj(all_edges, cutoff_time=test_cutoff)

    # 预构建负采样邻接表（全训练期）
    neg_adj_out, _ = build_adj_out(all_edges)

    all_scores, all_labels = [], []
    pos_scores_list, neg_scores_mat = [], []

    samples = []
    for u_val, v_val, t_val in zip(
        test_edges["src"].to_numpy(), test_edges["dst"].to_numpy(), test_edges["timestamp"].to_numpy()
    ):
        u, v, t = int(u_val), int(v_val), float(t_val)
        samples.append((u, v, t, 1))
        neg_vs = sample_negatives(u, t, all_edges, n_nodes,
                                  strategy="random", k=neg_ratio, seed=seed,
                                  prebuilt_adj_out=neg_adj_out)
        for nv in neg_vs:
            samples.append((u, nv, t, 0))

    total = len(samples)
    pos_samples = [(u, v, t) for u, v, t, l in samples if l == 1]

    # ── AUC/AP：batch 推理 ────────────────────────────────────────────────────
    for i in range(0, total, batch_size):
        batch = samples[i: i + batch_size]
        graphs, labels = [], []
        for u, v, t, label in batch:
            g = extract_subgraph(
                u, v, t, all_edges, max_hop, max_neighbors_per_node=max_neighbors, seed=seed,
                prebuilt_adj_out=prebuilt_adj_out, prebuilt_adj_in=prebuilt_adj_in,
            )
            if g is not None:
                graphs.append(g)
                labels.append(label)
        if not graphs:
            continue
        bg = dgl.batch(graphs).to(device)
        with torch.no_grad():
            scores = model.forward_batch(bg).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(np.array(labels))

        done = min(i + batch_size, total)
        print(f"\r评估进度: {done}/{total}", end="", flush=True)

    print()
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # ── Hits@K：每个正样本 vs. hits_neg_per_pos 个负样本（批量推理） ────────
    if hits_neg_per_pos > 0:
        hits_batch_size = batch_size
        for pos_idx, (u, v, t) in enumerate(pos_samples):
            # 构建 [pos] + [neg × hits_neg_per_pos] 的批量
            neg_vs = sample_negatives(u, t, all_edges, n_nodes,
                                      strategy="random", k=hits_neg_per_pos, seed=seed + 1,
                                      prebuilt_adj_out=neg_adj_out)
            pairs = [(u, v, 1)] + [(u, nv, 0) for nv in neg_vs]

            row_scores: list[float] = []
            graphs_buf, idx_buf = [], []
            for pair_v, pair_label in [(pv, pl) for _, pv, pl in pairs]:
                _u, _v = u, pair_v
                g = extract_subgraph(
                    _u, _v, t, all_edges, max_hop, max_neighbors_per_node=max_neighbors, seed=seed,
                    prebuilt_adj_out=prebuilt_adj_out, prebuilt_adj_in=prebuilt_adj_in,
                )
                graphs_buf.append(g)

            # 批量推理
            valid_graphs = [g for g in graphs_buf if g is not None]
            if not valid_graphs:
                continue
            bg_hits = dgl.batch(valid_graphs).to(device)
            with torch.no_grad():
                batch_scores = model.forward_batch(bg_hits).cpu().numpy().tolist()

            # 填充 None 位置为 0.0
            score_iter = iter(batch_scores)
            all_hit_scores = [next(score_iter) if g is not None else 0.0 for g in graphs_buf]

            pos_score = all_hit_scores[0]
            neg_hit_scores = all_hit_scores[1:hits_neg_per_pos + 1]
            while len(neg_hit_scores) < hits_neg_per_pos:
                neg_hit_scores.append(0.0)

            pos_scores_list.append(pos_score)
            neg_scores_mat.append(neg_hit_scores[:hits_neg_per_pos])

            if (pos_idx + 1) % 50 == 0:
                print(f"\rHits@K 进度: {pos_idx+1}/{len(pos_samples)}", end="", flush=True)
        print()

    pos_arr = np.array(pos_scores_list)
    neg_arr = np.array(neg_scores_mat)

    metrics = compute_all_metrics(
        all_labels, all_scores,
        pos_scores=pos_arr if len(pos_arr) > 0 else None,
        neg_scores=neg_arr if len(neg_arr) > 0 else None,
        k_list=[10, 20, 50],
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint 路径（.pt 文件）")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--max_hop", type=int, default=2)
    parser.add_argument("--max_neighbors", type=int, default=30)
    parser.add_argument("--neg_ratio", type=int, default=1)
    parser.add_argument("--hits_neg", type=int, default=99,
                        help="Hits@K 每个正样本对应的负样本数")
    parser.add_argument("--max_test_samples", type=int, default=0,
                        help="测试集最大样本数（0=不限制，用于 smoke test）")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    data_dir = Path(args.data_dir)
    edges = pd.read_csv(data_dir / "edges.csv").sort_values("timestamp").reset_index(drop=True)
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)
    n_nodes = meta["n_nodes"]

    _, _, test_edges = temporal_split(edges)
    if args.max_test_samples > 0:
        test_edges = test_edges.head(args.max_test_samples).reset_index(drop=True)
    print(f"测试集大小: {len(test_edges)}")

    model = LinkPredModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_auc={ckpt['val_auc']:.4f}）")

    t0 = time.time()
    metrics = evaluate(
        model, test_edges, edges, n_nodes,
        args.max_hop, args.max_neighbors,
        args.neg_ratio, args.hits_neg,
        args.seed, device, args.batch_size,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*40}")
    print(f"测试集结果（耗时 {elapsed:.1f}s）：")
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("="*40)

    # 保存结果
    out_path = Path(args.ckpt).parent / "test_results.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"结果已保存到 {out_path}")


if __name__ == "__main__":
    main()
