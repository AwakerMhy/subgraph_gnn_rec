"""src/dataset/real/lastfm_asia.py — LastFM Asia Social Network

数据来源：Benedek Rozemberczki https://github.com/benedekrozemberczki/datasets
文件：lastfm_asia_edges.csv
格式：无向边 "node_1,node_2"
处理：无向→双向化；行序号作代理时间
规模：~7624 节点，~27806 条原始边（双向化后 ~55612 条）
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class LastFMAsiaDatset(TemporalDataset):

    @property
    def name(self) -> str:
        return "lastfm_asia"

    def process(self) -> None:
        raw_path = self.raw_dir / "lastfm_asia" / "lastfm_asia_edges.csv"
        assert raw_path.exists(), (
            f"原始文件不存在：{raw_path}\n"
            "请从以下地址下载并放置到 data/raw/lastfm_asia/:\n"
            "https://raw.githubusercontent.com/benedekrozemberczki/datasets/master/lastfm_asia/lastfm_asia_edges.csv"
        )

        edges = pd.read_csv(raw_path, header=0)
        edges.columns = ["src", "dst"]
        edges = edges.dropna().astype(int)

        rev = edges.rename(columns={"src": "dst", "dst": "src"})
        edges = pd.concat([edges, rev], ignore_index=True).drop_duplicates()

        edges["timestamp"] = edges.index.astype(float)
        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)
        self._save_standard_format(edges, node_feats, has_native_node_feature=False)
        print(f"[LastFMAsia] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
