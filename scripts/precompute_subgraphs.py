"""scripts/precompute_subgraphs.py — 离线预计算并缓存子图

用法：
    # 预计算单个数据集（默认参数）
    PYTHONPATH=. C:/conda/envs/gnn/python.exe scripts/precompute_subgraphs.py \
        --data_dir data/processed/college_msg

    # 批量预计算所有数据集
    PYTHONPATH=. C:/conda/envs/gnn/python.exe scripts/precompute_subgraphs.py \
        --data_dir data/processed/college_msg data/processed/bitcoin_otc data/processed/email_eu

    # 包含 TGAT 的边时间缓存（额外生成 _et 版本）
    ... --store_edge_time

缓存输出位置：
    <data_dir>/subgraphs/<split>_hop<H>_k<K>.bin       — DGLGraph 列表
    <data_dir>/subgraphs/<split>_hop<H>_k<K>_meta.json — [(u, v, t, label), ...]
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.negative_sampling import build_adj_out, sample_negatives
from src.graph.subgraph import extract_subgraph
from src.utils.seed import set_seed
from src.utils.split import temporal_split


def build_samples(
    edges: pd.DataFrame,
    all_edges: pd.DataFrame,
    n_nodes: int,
    neg_ratio: int,
    seed: int,
) -> list[tuple[int, int, float, int]]:
    """构造 (u, v, t, label) 样本列表（正样本 + 负样本）。"""
    all_time_adj_out, _ = build_adj_out(all_edges, cutoff_time=None)
    samples: list[tuple[int, int, float, int]] = []
    for u_val, v_val, t_val in zip(
        edges["src"].to_numpy(), edges["dst"].to_numpy(), edges["timestamp"].to_numpy()
    ):
        u, v, t = int(u_val), int(v_val), float(t_val)
        samples.append((u, v, t, 1))
        neg_vs = sample_negatives(
            u, t, all_edges, n_nodes,
            strategy="random", k=neg_ratio, seed=seed,
            all_time_adj_out=all_time_adj_out,
        )
        for nv in neg_vs:
            samples.append((u, nv, t, 0))
    return samples


def precompute_split(
    samples: list[tuple[int, int, float, int]],
    all_edges: pd.DataFrame,
    out_dir: Path,
    split_name: str,
    max_hop: int,
    max_neighbors: int,
    seed: int,
    store_edge_time: bool,
) -> None:
    """预计算一个 split 的所有子图并保存到磁盘。"""
    suffix = f"_hop{max_hop}_k{max_neighbors}"
    if store_edge_time:
        suffix += "_et"
    bin_path  = out_dir / f"{split_name}{suffix}.bin"
    meta_path = out_dir / f"{split_name}{suffix}_meta.json"

    if bin_path.exists() and meta_path.exists():
        print(f"  [跳过] {bin_path.name} 已存在")
        return

    print(f"  预计算 {split_name}（{len(samples)} 样本）...", flush=True)
    t0 = time.time()

    graphs: list[dgl.DGLGraph] = []
    meta: list[tuple[int, int, float, int]] = []  # 过滤掉返回 None 的样本

    for i, (u, v, t, label) in enumerate(samples):
        g = extract_subgraph(
            u, v, t, all_edges,
            max_hop=max_hop,
            max_neighbors_per_node=max_neighbors,
            seed=seed,
            store_edge_time=store_edge_time,
        )
        if g is not None:
            graphs.append(g)
            meta.append((u, v, t, label))

        if (i + 1) % 1000 == 0 or (i + 1) == len(samples):
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            remaining = (len(samples) - i - 1) / speed
            print(f"    {i+1}/{len(samples)}  {speed:.0f} 样本/s  "
                  f"预计剩余 {remaining:.0f}s", flush=True)

    dgl.save_graphs(str(bin_path), graphs)
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"  完成：{len(graphs)} 个子图 → {bin_path.name}  ({time.time()-t0:.1f}s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, nargs="+", required=True,
                        help="processed 数据目录列表，可传多个")
    parser.add_argument("--max_hop",       type=int, default=2)
    parser.add_argument("--max_neighbors", type=int, default=30)
    parser.add_argument("--neg_ratio",     type=int, default=1)
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--store_edge_time", action="store_true",
                        help="额外生成含 edata[dt] 的缓存（供 TGAT 使用）")
    args = parser.parse_args()

    set_seed(args.seed)

    for data_dir_str in args.data_dir:
        data_dir = Path(data_dir_str)
        print(f"\n{'='*60}")
        print(f"数据集: {data_dir.name}")

        edges = pd.read_csv(data_dir / "edges.csv")
        with open(data_dir / "meta.json") as f:
            meta = json.load(f)
        n_nodes = meta["n_nodes"]
        edges = edges.sort_values("timestamp").reset_index(drop=True)

        train_edges, val_edges, test_edges = temporal_split(edges)
        print(f"  节点={n_nodes}  train={len(train_edges)}  "
              f"val={len(val_edges)}  test={len(test_edges)}")

        out_dir = data_dir / "subgraphs"
        out_dir.mkdir(exist_ok=True)

        for split_name, split_edges in [
            ("train", train_edges),
            ("val",   val_edges),
            ("test",  test_edges),
        ]:
            samples = build_samples(
                split_edges, edges, n_nodes, args.neg_ratio, args.seed
            )
            # 默认缓存（不含边时间）
            precompute_split(
                samples, edges, out_dir, split_name,
                args.max_hop, args.max_neighbors, args.seed,
                store_edge_time=False,
            )
            # TGAT 缓存（含边时间）
            if args.store_edge_time:
                precompute_split(
                    samples, edges, out_dir, split_name,
                    args.max_hop, args.max_neighbors, args.seed,
                    store_edge_time=True,
                )

        print(f"数据集 {data_dir.name} 全部完成")


if __name__ == "__main__":
    main()
