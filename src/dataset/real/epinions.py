"""src/dataset/real/epinions.py — Epinions 信任网络预处理

数据来源：SNAP https://snap.stanford.edu/data/soc-Epinions1.html
文件：soc-Epinions1.txt.gz（或解压后的 .txt）
格式：# 注释行 + "FromNodeId  ToNodeId"（制表符分隔，无时间戳）
特点：有向信任评分网络，约 75k 节点、508k 有向边；无原生时间戳，用边序号作代理时间
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class EpinionsDataset(TemporalDataset):
    """Epinions 信任关系数据集（有向版本）。"""

    @property
    def name(self) -> str:
        return "epinions"

    def process(self) -> None:
        raw_path = self.raw_dir / "epinions" / "soc-Epinions1.txt"
        if not raw_path.exists():
            gz_path = raw_path.with_suffix(".txt.gz")
            assert gz_path.exists(), (
                f"原始文件不存在：{raw_path}（也可用 .gz 压缩版）\n"
                "请从 https://snap.stanford.edu/data/soc-Epinions1.html 下载 soc-Epinions1.txt.gz"
            )
            raw_path = gz_path

        # 跳过以 # 开头的注释行
        edges = pd.read_csv(
            raw_path,
            sep="\t",
            comment="#",
            header=None,
            names=["src", "dst"],
            dtype={"src": int, "dst": int},
        )

        # 无原生时间戳：以行序号作为代理时间（用于 temporal_split）
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

        print(f"[Epinions] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
