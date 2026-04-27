"""src/dataset/real/wiki_vote.py — Wikipedia 投票网络预处理

数据来源：SNAP https://snap.stanford.edu/data/wiki-Vote.html
格式：# 注释行 + "FromNodeId\tToNodeId"（制表符分隔，无时间戳）
特点：有向投票信任网络，7115 节点、103689 边；recip≈0.056，deg_mean≈17；
     无原生时间戳，以行序号作代理时间（同 epinions 处理方式）
"""
from __future__ import annotations

import gzip

import pandas as pd

from src.dataset.base import TemporalDataset


class WikiVoteDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "wiki_vote"

    def process(self) -> None:
        raw_path = self.raw_dir / "wiki-Vote" / "wiki-Vote.txt.gz"
        assert raw_path.exists(), f"原始文件不存在：{raw_path}"

        rows: list[tuple[int, int]] = []
        with gzip.open(raw_path, "rt") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                a, b = line.split()
                rows.append((int(a), int(b)))

        edges = pd.DataFrame(rows, columns=["src", "dst"])
        edges["timestamp"] = edges.index.astype(float)

        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(edges=edges, node_feats=node_feats,
                                   has_native_node_feature=False)
        print(f"[WikiVote] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
