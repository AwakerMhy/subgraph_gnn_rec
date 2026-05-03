"""src/dataset/synthetic/dcsbm.py — Degree-Corrected SBM 合成数据集生成器

标准 DC-SBM（有向版）：
    P(i→j) ∝ θ_out[i] · θ_in[j] · B[c_i, c_j]

其中 θ_out / θ_in 服从 Pareto 分布，产生幂律度分布，模拟真实社交网络的
"富者愈富"结构（少数枢纽节点拥有大量连接）。

与 SBMGenerator 的区别：
  - SBMGenerator：所有节点在社区内活跃度由指数分布初始化，且随时间更新
  - DCSBMGenerator：θ_out/θ_in 在生成前一次性确定，整个过程保持不变；
    边概率由 DC-SBM 公式严格计算，不依赖共同邻居加成
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.dataset.synthetic.generator_base import GeneratorBase


class DCSBMGenerator(GeneratorBase):
    """Degree-Corrected Stochastic Block Model 生成器（有向、时序）。

    Args:
        n_nodes:            节点总数（建议 >= 1000 以体现幂律效果）
        n_communities:      社区数
        B_in:               块矩阵对角线值（社区内连接亲和力）
        B_out:              块矩阵非对角线值（跨社区连接亲和力）
        T:                  时间步数（每步产生约 edges_per_step 条边）
        edges_per_step:     每时间步目标边数
        theta_alpha:        Pareto 分布形状参数 α（越小尾越重，社交网络典型值 2.0–3.0）
        community_sizes:    各社区大小列表；None 时均等划分
        allow_repeat:       是否允许同一节点对出现多次（True = 多重图）
        seed:               随机种子
    """

    def __init__(
        self,
        n_nodes: int = 1000,
        n_communities: int = 5,
        B_in: float = 1.0,
        B_out: float = 0.05,
        T: int = 200,
        edges_per_step: int = 10,
        theta_alpha: float = 2.5,
        community_sizes: list[int] | None = None,
        allow_repeat: bool = False,
        seed: int = 42,
    ) -> None:
        self.n_nodes = n_nodes
        self.n_communities = n_communities
        self.B_in = B_in
        self.B_out = B_out
        self.T = T
        self.edges_per_step = edges_per_step
        self.theta_alpha = theta_alpha
        self.allow_repeat = allow_repeat
        self.seed = seed

        # 社区划分
        if community_sizes is not None:
            assert sum(community_sizes) == n_nodes, "community_sizes 之和必须等于 n_nodes"
            assert len(community_sizes) == n_communities
            sizes = community_sizes
        else:
            base, rem = divmod(n_nodes, n_communities)
            sizes = [base + (1 if i < rem else 0) for i in range(n_communities)]

        self._community = np.empty(n_nodes, dtype=np.int32)
        offset = 0
        for c, sz in enumerate(sizes):
            self._community[offset:offset + sz] = c
            offset += sz

        # 度校正参数（生成时才初始化，保证 seed 可复现）
        self._theta_out: np.ndarray | None = None
        self._theta_in: np.ndarray | None = None

    # ── 私有工具 ───────────────────────────────────────────────────────

    def _init_theta(self, rng: np.random.Generator) -> None:
        """从 Pareto(alpha) 分布采样 θ_out / θ_in，归一化为概率向量。"""
        # numpy Pareto: rng.pareto(a) 返回 Pareto(a) - 1 的样本，即 x ≥ 0
        # 我们需要 θ ≥ 1，所以加 1
        theta_out = rng.pareto(self.theta_alpha, size=self.n_nodes) + 1.0
        theta_in  = rng.pareto(self.theta_alpha, size=self.n_nodes) + 1.0
        self._theta_out = (theta_out / theta_out.sum()).astype(np.float64)
        self._theta_in  = (theta_in  / theta_in.sum()).astype(np.float64)

    def _dst_probs(self, u: int) -> np.ndarray:
        """计算以 u 为源节点时，所有目标节点的采样概率。"""
        # B[c_u, c_v] 向量
        block_weights = np.where(
            self._community == self._community[u], self.B_in, self.B_out
        ).astype(np.float64)
        p = self._theta_in * block_weights  # type: ignore[operator]
        p[u] = 0.0  # 禁止自环
        total = p.sum()
        if total == 0:
            return p
        return p / total

    # ── 公共接口 ───────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """生成有向时序边列表。

        Returns:
            DataFrame，columns: [src, dst, timestamp]，按 timestamp 升序。
        """
        rng = np.random.default_rng(self.seed)
        self._init_theta(rng)

        edges: list[tuple[int, int, float]] = []
        existing: set[tuple[int, int]] = set()

        for step in range(self.T):
            t = float(step)
            # 按 θ_out 加权采样源节点（过采样以填满 edges_per_step）
            n_try = self.edges_per_step * 5
            srcs = rng.choice(self.n_nodes, size=n_try, p=self._theta_out)

            added = 0
            for u in srcs:
                if added >= self.edges_per_step:
                    break
                p_dst = self._dst_probs(int(u))
                if p_dst.sum() == 0:
                    continue
                v = int(rng.choice(self.n_nodes, p=p_dst))
                if not self.allow_repeat and (u, v) in existing:
                    continue
                edges.append((int(u), v, t))
                existing.add((int(u), v))
                added += 1

        if not edges:
            raise RuntimeError("DC-SBM 生成器未产生任何边，请检查参数")

        df = pd.DataFrame(edges, columns=["src", "dst", "timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"[DCSBMGenerator] 生成完成：{self.n_nodes} 节点，{len(df)} 边")
        return df

    def get_node_features(self) -> np.ndarray:
        """社区 one-hot + log(θ_out) + log(θ_in)，共 n_communities+2 维。"""
        assert self._theta_out is not None, "请先调用 generate()"
        one_hot = np.zeros((self.n_nodes, self.n_communities), dtype=np.float32)
        one_hot[np.arange(self.n_nodes), self._community] = 1.0
        log_out = np.log(self._theta_out).reshape(-1, 1).astype(np.float32)
        log_in  = np.log(self._theta_in).reshape(-1, 1).astype(np.float32)
        return np.concatenate([one_hot, log_out, log_in], axis=1)
