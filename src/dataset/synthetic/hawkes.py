"""src/dataset/synthetic/hawkes.py — 多元 Hawkes 过程生成器

规则：历史边对未来边产生时序激励，近期发生的边对相关节点的未来建边概率有正向影响。
使用简化的多元 Hawkes 过程：强度函数 λ_ij(t) = μ + Σ α·exp(-β·(t - t_k))
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset.synthetic.generator_base import GeneratorBase


class HawkesGenerator(GeneratorBase):
    """Hawkes 过程合成数据集生成器。"""

    def __init__(
        self,
        n_nodes: int = 300,
        mu: float = 0.1,
        alpha: float = 0.5,
        beta: float = 1.0,
        T: float = 50.0,
        seed: int = 42,
    ) -> None:
        self.n_nodes = n_nodes
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.seed = seed

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        edges: list[tuple[int, int, float]] = []

        # 记录每对节点的历史事件时刻（简化：只跟踪发出节点的历史）
        node_history: dict[int, list[float]] = {i: [] for i in range(self.n_nodes)}

        t = 0.0
        max_edges = self.n_nodes * 20  # 防止爆炸

        while t < self.T and len(edges) < max_edges:
            # 计算全局强度上界（简化为所有节点的基础强度之和）
            intensities = np.zeros(self.n_nodes, dtype=np.float64)
            for u in range(self.n_nodes):
                lam = self.mu
                for t_k in node_history[u]:
                    lam += self.alpha * np.exp(-self.beta * (t - t_k))
                intensities[u] = lam

            lambda_total = intensities.sum()
            if lambda_total <= 0:
                break

            # 下一事件时间间隔（指数分布）
            dt = rng.exponential(1.0 / lambda_total)
            t += dt
            if t >= self.T:
                break

            # 按强度比例选择源节点
            probs = intensities / lambda_total
            u = int(rng.choice(self.n_nodes, p=probs))

            # 随机选择目标节点（排除自身）
            candidates = [v for v in range(self.n_nodes) if v != u]
            v = int(rng.choice(candidates))

            edges.append((u, v, t))
            node_history[u].append(t)

        if not edges:
            raise RuntimeError("Hawkes 生成器未产生任何边，请调整参数")

        df = pd.DataFrame(edges, columns=["src", "dst", "timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_node_features(self) -> np.ndarray:
        """随机正态向量（16维）。"""
        rng = np.random.default_rng(self.seed + 1)
        return rng.normal(0, 1, size=(self.n_nodes, 16)).astype(np.float32)
