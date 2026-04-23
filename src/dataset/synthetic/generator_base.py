"""src/dataset/synthetic/generator_base.py — 合成数据集生成器基类"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class GeneratorBase(ABC):
    """所有合成数据集生成器的统一接口。"""

    @abstractmethod
    def generate(self) -> pd.DataFrame:
        """生成有向边列表。

        Returns:
            DataFrame，columns: [src, dst, timestamp]
            timestamp 为浮点数，已按升序排序。
        """
        ...

    @abstractmethod
    def get_node_features(self) -> np.ndarray:
        """返回节点属性矩阵。

        Returns:
            np.ndarray，shape (n_nodes, feat_dim)，float32。
        """
        ...
