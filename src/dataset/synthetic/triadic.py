"""src/dataset/synthetic/triadic.py — 三角闭合合成数据集生成器

规则：三角闭合为主导规律。若 A→B、B→C 已存在，则 A→C 以高概率在短时间内出现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset.synthetic.generator_base import GeneratorBase


class TriadicGenerator(GeneratorBase):
    """三角闭合合成数据集生成器。"""

    def __init__(
        self,
        n_nodes: int = 400,
        base_p: float = 0.01,
        triadic_bonus: float = 0.4,
        T: int = 80,
        seed: int = 42,
    ) -> None:
        self.n_nodes = n_nodes
        self.base_p = base_p
        self.triadic_bonus = triadic_bonus
        self.T = T
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        edges: list[tuple[int, int, float]] = []
        adj: dict[int, set[int]] = {i: set() for i in range(self.n_nodes)}

        for step in range(self.T):
            t = float(step)
            # 枚举候选边（随机采样）
            n_candidates = max(10, self.n_nodes // 5)
            srcs = rng.integers(0, self.n_nodes, size=n_candidates)
            dsts = rng.integers(0, self.n_nodes, size=n_candidates)

            for u, v in zip(srcs, dsts):
                u, v = int(u), int(v)
                if u == v or v in adj[u]:
                    continue

                p = self.base_p
                # 三角闭合加成：检查是否存在 A→?→v（即 u 的出边邻居是否有 ?→v）
                for mid in adj[u]:
                    if v in adj[mid]:
                        p = min(1.0, p + self.triadic_bonus)
                        break  # 找到一条路径就加成

                if rng.random() < p:
                    edges.append((u, v, t))
                    adj[u].add(v)

        if not edges:
            raise RuntimeError("Triadic 生成器未产生任何边，请降低 base_p 或增大 triadic_bonus")

        df = pd.DataFrame(edges, columns=["src", "dst", "timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"[TriadicGenerator] 生成完成：{self.n_nodes} 节点，{len(df)} 边")
        return df

    def get_node_features(self) -> np.ndarray:
        """随机正态向量（16维）。"""
        rng = np.random.default_rng(self.seed + 1)
        return rng.normal(0, 1, size=(self.n_nodes, 16)).astype(np.float32)
