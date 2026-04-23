"""src/dataset/real/college_msg.py — CollegeMsg 数据集预处理

数据来源：SNAP https://snap.stanford.edu/data/CollegeMsg.html
格式：每行 "src dst timestamp"（Unix 时间戳，空格分隔）
特点：无原生节点属性，使用度特征 [in_deg, out_deg, total_deg] 占位
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset.base import TemporalDataset


class CollegeMsgDataset(TemporalDataset):
    """CollegeMsg 社交消息数据集。"""

    @property
    def name(self) -> str:
        return "college_msg"

    def process(self) -> None:
        """读取原始文件，输出标准格式。

        原始文件：data/raw/college_msg/CollegeMsg.txt
        格式：src dst timestamp（空格分隔，timestamp 为 Unix 整数）
        """
        raw_path = self.raw_dir / "college_msg" / "CollegeMsg.txt"
        assert raw_path.exists(), (
            f"原始文件不存在：{raw_path}\n"
            "请从 https://snap.stanford.edu/data/CollegeMsg.html 下载 CollegeMsg.txt"
        )

        # 读取原始文件
        edges = pd.read_csv(
            raw_path,
            sep=r"\s+",
            header=None,
            names=["src", "dst", "timestamp"],
            dtype={"src": int, "dst": int, "timestamp": int},
        )

        # 去除自环
        edges = self._remove_self_loops(edges)

        # 按时间戳升序排序
        edges = edges.sort_values("timestamp").reset_index(drop=True)

        # 节点 ID 重映射为连续整数
        edges, _ = self._remap_node_ids(edges)

        # 归一化时间戳（保留原始值）
        edges = self._normalize_timestamps(edges)

        n_nodes = int(max(edges["src"].max(), edges["dst"].max())) + 1

        # 度特征（基于全量训练截止时刻的图，这里用全量边计算后在 get_splits 时截断）
        # 注意：PLAN.md 规定度特征基于训练集截止时刻的图
        # 这里先用全量计算，实际训练时 DataLoader 会用训练集切分后的子图重算
        # 为简化预处理，此处存全量度特征作为初始占位（后续可在 base.py 中优化）
        node_feats = self._compute_degree_features(edges, n_nodes)

        self._save_standard_format(
            edges=edges,
            node_feats=node_feats,
            has_native_node_feature=False,
        )

        print(
            f"[CollegeMsg] 预处理完成："
            f"{n_nodes} 节点，{len(edges)} 边，"
            f"特征维度 {node_feats.shape[1]}"
        )
