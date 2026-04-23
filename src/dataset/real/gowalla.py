"""src/dataset/real/gowalla.py — Gowalla 社交网络预处理

数据来源：SNAP https://snap.stanford.edu/data/loc-gowalla.html
文件：loc-gowalla_edges.txt.gz（或解压后的 .txt）
格式：user_id  friend_id（制表符分隔，无时间戳）
特点：位置签到社交网络，约 196k 节点、190 万有向友谊边；
      无原生时间戳，以行序号作代理时间（同 Epinions 处理方式）
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class GowallaDataset(TemporalDataset):
    """Gowalla 友谊关系数据集（有向版本）。"""

    @property
    def name(self) -> str:
        return "gowalla"

    def process(self) -> None:
        raw_path = self.raw_dir / "gowalla" / "loc-gowalla_edges.txt"
        if not raw_path.exists():
            gz_path = raw_path.with_suffix(".txt.gz")
            assert gz_path.exists(), (
                f"原始文件不存在：{raw_path}（也可用 .gz 压缩版）\n"
                "请从 https://snap.stanford.edu/data/loc-gowalla.html 下载 loc-gowalla_edges.txt.gz"
            )
            raw_path = gz_path

        # 格式：user  friend（无时间戳列）
        edges = pd.read_csv(
            raw_path,
            sep="\t",
            header=None,
            names=["src", "dst"],
            dtype={"src": int, "dst": int},
        )

        # 无原生时间戳：以行序号作代理时间
        edges["timestamp"] = edges.index.astype(float)

        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(
            edges=edges,
            node_feats=node_feats,
            has_native_node_feature=False,
        )

        print(f"[Gowalla] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
