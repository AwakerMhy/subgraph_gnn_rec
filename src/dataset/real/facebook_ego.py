"""src/dataset/real/facebook_ego.py — Facebook Ego Network

数据来源：SNAP https://snap.stanford.edu/data/ego-Facebook.html
文件：facebook_combined.txt.gz（或解压后 .txt）
格式：无向边列表 "u v"，无时间戳
处理：无向→双向化；行序号作代理时间
规模：~4039 节点，~88234 条原始边（双向化后 ~176468 条有向边）
"""
from __future__ import annotations

import pandas as pd

from src.dataset.base import TemporalDataset


class FacebookEgoDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "facebook_ego"

    def process(self) -> None:
        raw_path = self.raw_dir / "facebook_ego" / "facebook_combined.txt"
        if not raw_path.exists():
            gz = raw_path.with_suffix(".txt.gz")
            assert gz.exists(), (
                f"原始文件不存在：{raw_path}\n"
                "请从 https://snap.stanford.edu/data/facebook_combined.txt.gz 下载"
            )
            raw_path = gz

        edges = pd.read_csv(
            raw_path, sep=" ", comment="#", header=None, names=["src", "dst"],
            dtype={"src": int, "dst": int},
        )
        # 无向 → 双向化
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
        print(f"[FacebookEgo] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
