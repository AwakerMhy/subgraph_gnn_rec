"""src/dataset/real/bitcoin_alpha.py — Bitcoin-Alpha 数据集预处理

数据来源：SNAP https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
格式：src,dst,rating,timestamp（CSV，rating 为信任评分 [-10,10]，丢弃）
与 Bitcoin-OTC 同源，规模略小
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class BitcoinAlphaDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "bitcoin_alpha"

    def process(self) -> None:
        raw_path = self.raw_dir / "soc-sign-bitcoin-alpha" / "soc-sign-bitcoin-alpha.csv"
        assert raw_path.exists(), f"原始文件不存在：{raw_path}"

        edges = pd.read_csv(
            raw_path,
            header=None,
            names=["src", "dst", "rating", "timestamp"],
            dtype={"src": int, "dst": int, "rating": int, "timestamp": float},
        )

        edges = edges[["src", "dst", "timestamp"]].copy()
        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(edges=edges, node_feats=node_feats,
                                   has_native_node_feature=False)
        print(f"[BitcoinAlpha] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
