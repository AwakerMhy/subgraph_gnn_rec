"""src/dataset/real/sx_askubuntu.py — StackExchange AskUbuntu 数据集预处理

数据来源：SNAP https://snap.stanford.edu/data/sx-askubuntu.html
格式：每行 "src dst timestamp"（空格分隔，Unix 时间戳）
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class SXAskUbuntuDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "sx_askubuntu"

    def process(self) -> None:
        raw_path = self.raw_dir / "sx-askubuntu" / "sx-askubuntu.txt"
        assert raw_path.exists(), f"原始文件不存在：{raw_path}"

        edges = pd.read_csv(
            raw_path,
            sep=r"\s+",
            header=None,
            names=["src", "dst", "timestamp"],
            dtype={"src": int, "dst": int, "timestamp": int},
        )

        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(edges=edges, node_feats=node_feats,
                                   has_native_node_feature=False)
        print(f"[SX-AskUbuntu] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
