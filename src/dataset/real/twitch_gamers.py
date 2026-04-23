"""src/dataset/real/twitch_gamers.py — Twitch Gamers Social Network

数据来源：SNAP https://snap.stanford.edu/data/twitch_gamers.html
文件：large_twitch_edges.csv（~6.8M 条边）
格式：无向边 "numeric_id_1,numeric_id_2,..."（CSV，取前两列）
处理：无向→双向化；行序号作代理时间
规模：~168207 节点，~6.8M 条原始边
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class TwitchGamersDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "twitch_gamers"

    def process(self) -> None:
        raw_path = self.raw_dir / "twitch_gamers" / "large_twitch_edges.csv"
        assert raw_path.exists(), (
            f"原始文件不存在：{raw_path}\n"
            "请从 https://snap.stanford.edu/data/twitch_gamers.html 下载 large_twitch_edges.csv"
        )

        raw = pd.read_csv(raw_path, usecols=[0, 1], header=0)
        raw.columns = ["src", "dst"]
        raw = raw.dropna().astype(int)

        rev = raw.rename(columns={"src": "dst", "dst": "src"})
        edges = pd.concat([raw, rev], ignore_index=True).drop_duplicates()

        edges["timestamp"] = edges.index.astype(float)
        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)
        self._save_standard_format(edges, node_feats, has_native_node_feature=False)
        print(f"[TwitchGamers] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
