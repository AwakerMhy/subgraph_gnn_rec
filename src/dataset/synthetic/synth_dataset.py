"""src/dataset/synthetic/synth_dataset.py — 合成数据集通用 Dataset 类"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset.base import TemporalDataset
from src.dataset.synthetic.generator_base import GeneratorBase


class SyntheticDataset(TemporalDataset):
    """将合成生成器输出包装为标准 TemporalDataset。

    使用示例：
        gen = SBMGenerator(n_nodes=500, ...)
        ds = SyntheticDataset(gen, dataset_name="synth_sbm", processed_dir="data/processed")
        ds.load()
    """

    def __init__(
        self,
        generator: GeneratorBase,
        dataset_name: str,
        processed_dir: str = "data/processed",
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> None:
        self._name = dataset_name
        self._generator = generator
        super().__init__(
            raw_dir="data/raw",       # 合成数据集不需要 raw_dir，占位即可
            processed_dir=processed_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

    @property
    def name(self) -> str:
        return self._name

    def process(self) -> None:
        edges = self._generator.generate()
        node_feats = self._generator.get_node_features().astype(np.float32)

        # 去除自环
        edges = self._remove_self_loops(edges)

        # 节点 ID 重映射（合成生成器通常已经是连续整数）
        edges, _ = self._remap_node_ids(edges)

        # 归一化时间戳
        edges = self._normalize_timestamps(edges)

        n_nodes = node_feats.shape[0]

        self._save_standard_format(
            edges=edges,
            node_feats=node_feats,
            has_native_node_feature=True,
        )

        print(
            f"[{self._name}] 生成完成："
            f"{n_nodes} 节点，{len(edges)} 边，"
            f"特征维度 {node_feats.shape[1]}"
        )
