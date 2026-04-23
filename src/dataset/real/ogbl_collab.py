"""src/dataset/real/ogbl_collab.py — OGB ogbl-collab 论文合作网络

数据来源：OGB https://ogb.stanford.edu/docs/linkprop/#ogbl-collab
用 year 字段作时间戳（有真实时序）；无向→双向化
规模：~235868 节点，~1285465 条原始边
依赖：pip install ogb
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset.base import TemporalDataset


class OgblCollabDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "ogbl_collab"

    def process(self) -> None:
        import gzip, urllib.request, zipfile
        from pathlib import Path

        raw_dir = self.raw_dir / "ogbl_collab"
        raw_dir.mkdir(parents=True, exist_ok=True)
        edge_file = raw_dir / "edge.csv.gz"
        year_file = raw_dir / "edge_year.csv.gz"

        base_url = "https://snap.stanford.edu/ogb/data/linkproppred/collab.zip"
        zip_path = raw_dir / "collab.zip"

        if not edge_file.exists():
            print("[ogbl-collab] 下载数据集（~25MB）...")
            urllib.request.urlretrieve(base_url, zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    if name.endswith(".csv.gz"):
                        fname = Path(name).name
                        with zf.open(name) as src, open(raw_dir / fname, "wb") as dst:
                            dst.write(src.read())
            zip_path.unlink()
            print("[ogbl-collab] 下载解压完成")

        print("[ogbl-collab] 读取边表...")
        with gzip.open(edge_file, "rt") as f:
            edge_data = pd.read_csv(f, header=None, names=["src", "dst"])
        with gzip.open(year_file, "rt") as f:
            year_data = pd.read_csv(f, header=None, names=["year"])

        edges = edge_data.copy()
        edges["timestamp"] = year_data["year"].astype(float)

        # 无向 → 双向化
        rev = edges[["dst", "src", "timestamp"]].rename(columns={"dst": "src", "src": "dst"})
        edges = pd.concat([edges, rev], ignore_index=True).drop_duplicates(subset=["src", "dst"])

        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feat = self._compute_degree_features(edges, n_nodes)
        self._save_standard_format(edges, node_feat, has_native_node_feature=False)
        print(f"[ogbl-collab] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
