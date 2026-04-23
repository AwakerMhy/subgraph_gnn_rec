"""src/dataset/real/email_eu.py — Email-EU-core-temporal 数据集预处理

数据来源：SNAP https://snap.stanford.edu/data/email-EuAll.html
格式：src dst timestamp（空格分隔）
特点：高密度，重复边保留最早时间戳，无节点属性，度特征占位
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class EmailEUDataset(TemporalDataset):
    """Email-EU-core-temporal 数据集。"""

    @property
    def name(self) -> str:
        return "email_eu"

    def process(self) -> None:
        raw_path = self.raw_dir / "email_eu" / "email-Eu-core-temporal.txt"
        assert raw_path.exists(), (
            f"原始文件不存在：{raw_path}\n"
            "请从 https://snap.stanford.edu/data/email-EuAll.html 下载"
        )

        edges = pd.read_csv(
            raw_path,
            sep=r"\s+",
            comment="#",
            header=None,
            names=["src", "dst", "timestamp"],
            dtype={"src": int, "dst": int, "timestamp": int},
        )

        # 去除自环
        edges = self._remove_self_loops(edges)

        # 重复边：同 (src, dst) 只保留最早时间戳
        edges = (
            edges.sort_values("timestamp")
            .groupby(["src", "dst"], as_index=False)
            .first()
        )

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
            f"[EmailEU] 预处理完成："
            f"{n_nodes} 节点，{len(edges)} 边"
        )
