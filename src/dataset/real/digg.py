"""src/dataset/real/digg.py — Digg 有向回复网络预处理

数据来源：KONECT http://konect.cc/networks/munmun_digg_reply/
格式：tar.bz2 内 out.munmun_digg_reply，% 注释行 + "src dst [weight] [timestamp]"
特点：有向回复交互网络；无原生时间戳时以行序号作代理时间
"""
from __future__ import annotations

import tarfile
from io import TextIOWrapper

import pandas as pd

from src.dataset.base import TemporalDataset


class DiggDataset(TemporalDataset):

    @property
    def name(self) -> str:
        return "digg"

    def process(self) -> None:
        tar_path = self.raw_dir / "digg" / "download.tsv.munmun_digg_reply.tar.bz2"
        assert tar_path.exists(), (
            f"原始文件不存在：{tar_path}\n"
            "请从 http://konect.cc/files/download.tsv.munmun_digg_reply.tar.bz2 下载"
        )

        rows: list[tuple[int, int, float]] = []
        with tarfile.open(tar_path, "r:bz2") as tf:
            target = None
            for member in tf.getmembers():
                if "out." in member.name and not member.name.endswith("/"):
                    target = member
                    break
            assert target is not None, "在压缩包中找不到 out.* 文件"
            f = tf.extractfile(target)
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

        edges = pd.DataFrame(rows, columns=["src", "dst", "timestamp"])
        edges = self._remove_self_loops(edges)
        edges = edges.sort_values("timestamp").reset_index(drop=True)
        edges, _ = self._remap_node_ids(edges)
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(edges=edges, node_feats=node_feats,
                                   has_native_node_feature=False)
        print(f"[Digg] 预处理完成：{n_nodes} 节点，{len(edges)} 边")
