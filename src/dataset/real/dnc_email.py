"""src/dataset/real/dnc_email.py — DNC Email 数据集预处理

数据来源：Konnect http://konect.cc/networks/dnc-temporalGraph/
格式：Konnect 格式，以 % 开头的注释行，数据行为 "src dst weight timestamp"
特点：DNC 竞选委员会内部邮件往来，有向时序图
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class DNCEmailDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "dnc_email"

    def process(self) -> None:
        raw_path = (self.raw_dir / "dnc-email" / "dnc-temporalGraph"
                    / "out.dnc-temporalGraph")
        assert raw_path.exists(), f"原始文件不存在：{raw_path}"

        # 跳过以 % 开头的注释行
        edges = pd.read_csv(
            raw_path,
            sep=r"\s+",
            comment="%",
            header=None,
            names=["src", "dst", "weight", "timestamp"],
            dtype={"src": int, "dst": int, "weight": int, "timestamp": int},
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
        print(f"[DNC-Email] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
