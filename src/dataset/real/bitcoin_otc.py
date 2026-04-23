"""src/dataset/real/bitcoin_otc.py — Bitcoin-OTC 数据集预处理

数据来源：SNAP https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
格式：src,dest,rating,time（CSV，time 为 Unix 时间戳）
特点：无原生节点属性，边权重（rating）丢弃，度特征占位
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class BitcoinOTCDataset(TemporalDataset):
    """Bitcoin-OTC 信任评分数据集。"""

    @property
    def name(self) -> str:
        return "bitcoin_otc"

    def process(self) -> None:
        raw_path = self.raw_dir / "bitcoin_otc" / "soc-sign-bitcoinotc.csv"
        assert raw_path.exists(), (
            f"原始文件不存在：{raw_path}\n"
            "请从 https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html 下载"
        )

        edges = pd.read_csv(
            raw_path,
            header=None,
            names=["src", "dst", "rating", "timestamp"],
            dtype={"src": int, "dst": int, "rating": int, "timestamp": float},
        )

        # 丢弃 rating 列
        edges = edges[["src", "dst", "timestamp"]].copy()

        # 去除自环
        edges = self._remove_self_loops(edges)

        # 按时间排序
        edges = edges.sort_values("timestamp").reset_index(drop=True)

        # 节点 ID 重映射
        edges, _ = self._remap_node_ids(edges)

        # 归一化时间戳
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(
            edges=edges,
            node_feats=node_feats,
            has_native_node_feature=False,
        )

        print(
            f"[BitcoinOTC] 预处理完成："
            f"{n_nodes} 节点，{len(edges)} 边"
        )
