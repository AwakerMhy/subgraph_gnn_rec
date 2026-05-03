"""src/dataset/real/advogato.py — Advogato 有向信任网络预处理

数据来源：KONECT http://konect.cc/networks/advogato/
格式：tar.bz2 内 out.advogato，% 注释行 + "src dst weight timestamp"
特点：有向信任网络，约 6.5k 节点、51k 边；有原生 Unix 时间戳；recip≈0.24，deg_mean≈7.8
"""
from __future__ import annotations

import tarfile
from io import TextIOWrapper

import pandas as pd

from src.dataset.base import TemporalDataset


class AdvogatoDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "advogato"

    def process(self) -> None:
        tar_path = self.raw_dir / "advogato" / "download.tsv.advogato.tar.bz2"
        assert tar_path.exists(), (
            f"原始文件不存在：{tar_path}\n"
            "请从 http://konect.cc/files/download.tsv.advogato.tar.bz2 下载"
        )

        rows: list[tuple[int, int, float]] = []
        with tarfile.open(tar_path, "r:bz2") as tf:
            for member in tf.getmembers():
                if member.name.endswith("out.advogato"):
                    f = tf.extractfile(member)
                    assert f is not None
                    for line in TextIOWrapper(f, encoding="utf-8"):
                        line = line.strip()
                        if not line or line.startswith("%"):
                            continue
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        src, dst = int(parts[0]), int(parts[1])
                        ts = float(parts[3]) if len(parts) >= 4 else float(len(rows))
                        rows.append((src, dst, ts))
                    break

        edges = pd.DataFrame(rows, columns=["src", "dst", "timestamp"])
        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(edges=edges, node_feats=node_feats,
                                   has_native_node_feature=False)
        print(f"[Advogato] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
