"""src/dataset/base.py — 时序数据集基类"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.split import temporal_split


class TemporalDataset(ABC):
    """所有数据集的统一接口。

    子类必须实现：
        - raw_file_name (property): 原始文件名（相对于 raw_dir）
        - process(): 读取原始文件，输出标准格式到 processed_dir

    使用方式：
        ds = CollegeMsgDataset(raw_dir="data/raw", processed_dir="data/processed")
        ds.load()
        train, val, test = ds.get_splits()
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir) / self.name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # 加载后填充
        self.edges: pd.DataFrame | None = None      # columns: src, dst, timestamp[, timestamp_raw]
        self.node_feats: np.ndarray | None = None   # (n_nodes, feat_dim)
        self.meta: dict | None = None

    # ── 子类必须实现 ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """数据集名称（用于 processed_dir 子目录名）。"""
        ...

    @abstractmethod
    def process(self) -> None:
        """读取 raw_dir 中的原始文件，输出标准格式到 processed_dir。

        输出：
            processed_dir/edges.csv   — src, dst, timestamp, timestamp_raw
            processed_dir/nodes.csv   — node_id, feat_0, ..., feat_d
            processed_dir/meta.json   — n_nodes, n_edges, has_native_node_feature, ...
        """
        ...

    # ── 公共接口 ───────────────────────────────────────────────────────

    def load(self, force_reprocess: bool = False, first_time_only: bool = False) -> None:
        """加载数据集。若 processed_dir 不存在则先 process()。

        Args:
            force_reprocess:  强制重新预处理原始文件
            first_time_only:  True 时只保留每个 (src, dst) 对的最早一条边，
                              用于 New Link Prediction 任务（Step 1 正样本净化）
        """
        edges_path = self.processed_dir / "edges.csv"
        nodes_path = self.processed_dir / "nodes.csv"
        meta_path = self.processed_dir / "meta.json"

        if force_reprocess or not edges_path.exists():
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            self.process()

        self.edges = pd.read_csv(edges_path)
        if first_time_only:
            from src.graph.edge_split import filter_first_time_edges
            self.edges = filter_first_time_edges(self.edges)
        nodes_df = pd.read_csv(nodes_path)
        feat_cols = [c for c in nodes_df.columns if c != "node_id"]
        self.node_feats = nodes_df[feat_cols].values.astype(np.float32)

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

    def get_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """返回 (train_edges, val_edges, test_edges)。"""
        assert self.edges is not None, "请先调用 load()"
        return temporal_split(self.edges, self.train_ratio, self.val_ratio)

    # ── 工具方法（子类可调用） ─────────────────────────────────────────

    @staticmethod
    def _remove_self_loops(edges: pd.DataFrame) -> pd.DataFrame:
        return edges[edges["src"] != edges["dst"]].reset_index(drop=True)

    @staticmethod
    def _remap_node_ids(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
        """将节点 ID 重映射为连续整数 [0, n_nodes)，返回新边表和映射字典。"""
        all_nodes = pd.unique(pd.concat([edges["src"], edges["dst"]]))
        mapping = {old: new for new, old in enumerate(sorted(all_nodes))}
        edges = edges.copy()
        edges["src"] = edges["src"].map(mapping)
        edges["dst"] = edges["dst"].map(mapping)
        return edges, mapping

    @staticmethod
    def _normalize_timestamps(edges: pd.DataFrame) -> pd.DataFrame:
        """将 timestamp 归一化到 [0, 1]，原始值保留在 timestamp_raw。"""
        edges = edges.copy()
        if "timestamp_raw" not in edges.columns:
            edges["timestamp_raw"] = edges["timestamp"].copy()
        t_min = edges["timestamp"].min()
        t_max = edges["timestamp"].max()
        if t_max > t_min:
            edges["timestamp"] = (edges["timestamp"] - t_min) / (t_max - t_min)
        else:
            edges["timestamp"] = 0.0
        return edges

    @staticmethod
    def _compute_degree_features(edges: pd.DataFrame, n_nodes: int) -> np.ndarray:
        """计算每个节点的 [in_degree, out_degree, total_degree]，shape (n_nodes, 3)。"""
        src = edges["src"].to_numpy(dtype=np.int64)
        dst = edges["dst"].to_numpy(dtype=np.int64)
        out_deg = np.bincount(src, minlength=n_nodes).astype(np.float32)
        in_deg  = np.bincount(dst, minlength=n_nodes).astype(np.float32)
        total_deg = in_deg + out_deg
        return np.stack([in_deg, out_deg, total_deg], axis=1)

    def _save_standard_format(
        self,
        edges: pd.DataFrame,
        node_feats: np.ndarray,
        has_native_node_feature: bool,
    ) -> None:
        """将标准格式写入 processed_dir。"""
        n_nodes = node_feats.shape[0]
        feat_dim = node_feats.shape[1]

        # edges.csv
        edges.to_csv(self.processed_dir / "edges.csv", index=False)

        # nodes.csv
        feat_cols = [f"feat_{i}" for i in range(feat_dim)]
        nodes_df = pd.DataFrame(node_feats, columns=feat_cols)
        nodes_df.insert(0, "node_id", np.arange(n_nodes))
        nodes_df.to_csv(self.processed_dir / "nodes.csv", index=False)

        # meta.json
        meta = {
            "n_nodes": n_nodes,
            "n_edges": len(edges),
            "has_native_node_feature": has_native_node_feature,
            "feat_dim": feat_dim,
            "t_min": float(edges["timestamp"].min()),
            "t_max": float(edges["timestamp"].max()),
            "is_directed": True,
        }
        with open(self.processed_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.meta = meta
