"""src/dataset/synthetic/sbm.py — 随机块模型（SBM）合成数据集生成器

规则：
- 同社区建边概率 p_in >> 跨社区 p_out
- 共同邻居加成（已有共同邻居的节点对建边概率提升）
- 活跃度加权（出度高的节点更容易发起新边）
- 节点属性：社区 one-hot + 随机噪声向量
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset.synthetic.generator_base import GeneratorBase


class SBMGenerator(GeneratorBase):
    """随机块模型生成器。"""

    def __init__(
        self,
        n_nodes: int = 500,
        n_communities: int = 5,
        p_in: float = 0.3,
        p_out: float = 0.02,
        T: int = 100,
        edges_per_step: int = 10,
        common_neighbor_bonus: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.p_in = p_in
        self.p_out = p_out
        self.T = T
        self.edges_per_step = edges_per_step
        self.common_neighbor_bonus = common_neighbor_bonus
        self.seed = seed

        assert n_nodes % n_communities == 0 or n_nodes >= n_communities, \
            "n_nodes 应可被 n_communities 整除（或 >= n_communities）"

        # 分配社区标签
        self._community = np.array_split(np.arange(n_nodes), n_communities)
        self._node_community = np.zeros(n_nodes, dtype=int)
        for c, members in enumerate(self._community):
            self._node_community[members] = c

        self._edges: list[tuple[int, int, float]] = []
        self._adj: dict[int, set[int]] = {i: set() for i in range(n_nodes)}

    def _base_prob(self, u: int, v: int) -> float:
        if self._node_community[u] == self._node_community[v]:
            return self.p_in
        return self.p_out

    def _edge_prob(self, u: int, v: int) -> float:
        p = self._base_prob(u, v)
        # 共同邻居加成
        common = len(self._adj[u] & self._adj[v])
        p = min(1.0, p + common * self.common_neighbor_bonus)
        return p

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        self._edges = []
        self._adj = {i: set() for i in range(self.n_nodes)}

        # 出度（活跃度）：每个节点初始随机出度权重
        activity = rng.exponential(1.0, size=self.n_nodes).astype(np.float32)
        activity /= activity.sum()

        for step in range(self.T):
            t = float(step)
            n_try = self.edges_per_step * 3  # 过采样后过滤

            # 按活跃度加权采样源节点
            srcs = rng.choice(self.n_nodes, size=n_try, p=activity)

            added = 0
            for u in srcs:
                if added >= self.edges_per_step:
                    break
                # 随机采样目标节点
                v = int(rng.integers(0, self.n_nodes))
                if v == u or v in self._adj[u]:
                    continue
                prob = self._edge_prob(u, v)
                if rng.random() < prob:
                    self._edges.append((u, v, t))
                    self._adj[u].add(v)
                    # 更新活跃度（出度增加）
                    activity[u] += 0.1
                    activity /= activity.sum()
                    added += 1

        if not self._edges:
            raise RuntimeError("SBM 生成器未产生任何边，请检查参数")

        df = pd.DataFrame(self._edges, columns=["src", "dst", "timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def get_node_features(self) -> np.ndarray:
        """社区 one-hot（n_communities 维）+ 随机噪声（8维）。"""
        rng = np.random.default_rng(self.seed + 1)
        one_hot = np.zeros((self.n_nodes, self.n_communities), dtype=np.float32)
        one_hot[np.arange(self.n_nodes), self._node_community] = 1.0
        noise = rng.normal(0, 0.1, size=(self.n_nodes, 8)).astype(np.float32)
        return np.concatenate([one_hot, noise], axis=1)
