"""src/dataset/real/higgs_reply.py — Higgs Twitter 回复网络预处理

数据来源：SNAP https://snap.stanford.edu/data/higgs-twitter.html
文件：higgs-reply_network.edgelist.gz
格式：% 注释 + "src dst timestamp"（Unix 时间戳）
特点：有向回复网络，约 256k 节点、361k 边；recip 极低（<0.05）；deg_mean≈1.4
"""
from __future__ import annotations

import gzip

import pandas as pd

from src.dataset.base import TemporalDataset


class HiggsReplyDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "higgs_reply"

    def process(self) -> None:
        gz_path = self.raw_dir / "higgs_reply" / "higgs-reply_network.edgelist.gz"
        assert gz_path.exists(), (
            f"原始文件不存在：{gz_path}\n"
            "请从 https://snap.stanford.edu/data/higgs-reply_network.edgelist.gz 下载"
        )

        rows: list[tuple[int, int, float]] = []
        with gzip.open(gz_path, "rt") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("%") or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                src, dst = int(parts[0]), int(parts[1])
                ts = float(parts[2]) if len(parts) >= 3 else float(len(rows))
                rows.append((src, dst, ts))

        edges = pd.DataFrame(rows, columns=["src", "dst", "timestamp"])
        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(edges=edges, node_feats=node_feats,
                                   has_native_node_feature=False)
        print(f"[HiggsReply] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
