"""src/online/env.py — 在线社交推荐环境。

持有 G*（ground truth）和 G_t（当前观测图），管理 cooldown、用户采样、
接受/拒绝反馈，以及图演化。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.online.feedback import Feedback, FeedbackSimulator
from src.online.static_adj import StaticAdjacency
from src.online.user_selector import UserSelector


class OnlineEnv:
    """在线社交推荐仿真环境。

    属性：
        adj:      当前观测图 G_t（可变 StaticAdjacency）
        coverage: |E_t ∩ E*| / |E*|（增量维护）
    """

    def __init__(
        self,
        star_edges: pd.DataFrame,
        n_nodes: int,
        init_edge_ratio: float = 0.05,
        user_sample_ratio: float = 0.10,
        cooldown_rounds: int = 5,
        p_accept: float = 1.0,
        p_pos: float | None = None,
        p_neg: float = 0.0,
        seed: int = 42,
        init_stratified: bool = False,
        init_strategy: str | None = None,
        snowball_seeds: int = 5,
        user_selector_cfg: dict | None = None,
    ) -> None:
        self._n = n_nodes
        self._user_sample_ratio = user_sample_ratio
        self._cooldown_rounds = cooldown_rounds
        self._rng = np.random.default_rng(seed)

        # G* — 去重有向边集合
        pairs = list(zip(star_edges["src"].astype(int), star_edges["dst"].astype(int)))
        self._star_set: set[tuple[int, int]] = set(pairs)
        self._star_size = len(self._star_set)

        # 确定实际策略（init_strategy 优先于 init_stratified）
        if init_strategy is None:
            strategy = "stratified" if init_stratified else "random"
        else:
            strategy = init_strategy

        # G_0 — 初始化采样
        init_n = max(1, int(len(pairs) * init_edge_ratio))
        init_pairs = self._sample_init_edges(pairs, init_n, strategy, snowball_seeds)
        init_df = pd.DataFrame(init_pairs, columns=["src", "dst"])

        self.adj = StaticAdjacency(n_nodes, init_df)
        self._accepted_set: set[tuple[int, int]] = set(init_pairs)
        self._coverage_count = len(init_pairs)  # |E_t ∩ E*|

        # cooldown book: (u,v) -> unlock_round
        self._cooldown: dict[tuple[int, int], int] = {}

        _p_pos = p_pos if p_pos is not None else p_accept
        self._feedback_sim = FeedbackSimulator(
            self._star_set, p_pos=_p_pos, p_neg=p_neg, rng=self._rng
        )
        self._cooldown_mode: str = "hard"  # 由 loop.py 通过 set_cooldown_mode() 覆盖

        # 用户选择策略
        sel_cfg = user_selector_cfg or {}
        self._selector = UserSelector(
            n_nodes=n_nodes,
            strategy=sel_cfg.get("strategy", "uniform"),
            alpha=sel_cfg.get("alpha", 0.5),
            beta=sel_cfg.get("beta", 2.0),
            lam=sel_cfg.get("lam", 0.1),
            gamma=sel_cfg.get("gamma", 2.0),
            w=sel_cfg.get("w", 3),
            sample_ratio=sel_cfg.get("sample_ratio", user_sample_ratio),
            seed=seed + 1,
        )

    # ── 初始化策略 ────────────────────────────────────────────────────────────

    def _sample_init_edges(
        self,
        pairs: list[tuple[int, int]],
        init_n: int,
        strategy: str,
        snowball_seeds: int,
    ) -> list[tuple[int, int]]:
        if strategy == "random":
            idx = self._rng.choice(len(pairs), size=min(init_n, len(pairs)), replace=False)
            return [pairs[i] for i in idx]

        if strategy == "stratified":
            from collections import defaultdict  # noqa: PLC0415
            src_to_idx: dict[int, list[int]] = defaultdict(list)
            for i, (u, _) in enumerate(pairs):
                src_to_idx[u].append(i)
            guaranteed = {int(self._rng.choice(idxs)) for idxs in src_to_idx.values()}
            remaining = max(0, init_n - len(guaranteed))
            pool = [i for i in range(len(pairs)) if i not in guaranteed]
            if remaining > 0 and pool:
                extra = self._rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                guaranteed.update(extra.tolist())
            return [pairs[i] for i in guaranteed]

        if strategy == "snowball":
            # 多种子 BFS：从 snowball_seeds 个随机种子出发，BFS 扩展直到边数达到 init_n
            src_set = list({u for u, _ in pairs})
            seed_nodes = self._rng.choice(
                src_set, size=min(snowball_seeds, len(src_set)), replace=False
            ).tolist()
            adj_map: dict[int, list[int]] = {}
            for u, v in pairs:
                adj_map.setdefault(u, []).append(v)

            visited_edges: set[tuple[int, int]] = set()
            queue = list(seed_nodes)
            visited_nodes: set[int] = set(seed_nodes)
            while queue and len(visited_edges) < init_n:
                u = queue.pop(0)
                for v in adj_map.get(u, []):
                    if (u, v) not in visited_edges:
                        visited_edges.add((u, v))
                    if v not in visited_nodes:
                        visited_nodes.add(v)
                        queue.append(v)
                    if len(visited_edges) >= init_n:
                        break
            result = list(visited_edges)
            # 不足时随机补充
            if len(result) < init_n:
                remaining_pool = [p for p in pairs if p not in visited_edges]
                extra_n = min(init_n - len(result), len(remaining_pool))
                if extra_n > 0:
                    extra_idx = self._rng.choice(len(remaining_pool), size=extra_n, replace=False)
                    result += [remaining_pool[i] for i in extra_idx]
            return result

        if strategy == "forest_fire":
            # Forest Fire：每个种子以概率 p_fire=0.7 向出邻居扩散
            p_fire = 0.7
            src_set = list({u for u, _ in pairs})
            seed_nodes = self._rng.choice(
                src_set, size=min(snowball_seeds, len(src_set)), replace=False
            ).tolist()
            adj_map: dict[int, list[int]] = {}  # type: ignore[assignment]
            for u, v in pairs:
                adj_map.setdefault(u, []).append(v)

            visited_edges: set[tuple[int, int]] = set()  # type: ignore[assignment]
            burned: set[int] = set()

            def burn(node: int) -> None:
                if node in burned or len(visited_edges) >= init_n:
                    return
                burned.add(node)
                nbrs = list(adj_map.get(node, []))
                self._rng.shuffle(nbrs)
                k = max(1, int(np.ceil(self._rng.geometric(p=1 - p_fire + 1e-6) - 1)))
                for v in nbrs[:k]:
                    if len(visited_edges) >= init_n:
                        break
                    visited_edges.add((node, v))
                    burn(v)

            import sys  # noqa: PLC0415
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(max(old_limit, len(pairs) + 500))
            try:
                for seed in seed_nodes:
                    burn(seed)
                    if len(visited_edges) >= init_n:
                        break
            finally:
                sys.setrecursionlimit(old_limit)

            result = list(visited_edges)
            # 不足时用 stratified 兜底
            if len(result) < init_n:
                remaining_pool = [p for p in pairs if p not in visited_edges]
                extra_n = min(init_n - len(result), len(remaining_pool))
                if extra_n > 0:
                    extra_idx = self._rng.choice(len(remaining_pool), size=extra_n, replace=False)
                    result += [remaining_pool[i] for i in extra_idx]
            return result

        if strategy == "all_covered":
            from collections import defaultdict  # noqa: PLC0415
            src_to_idx: dict[int, list[int]] = defaultdict(list)
            dst_to_idx: dict[int, list[int]] = defaultdict(list)
            for i, (u, v) in enumerate(pairs):
                src_to_idx[u].append(i)
                dst_to_idx[v].append(i)

            guaranteed: set[int] = set()
            # 每个 src 至少一条出边
            for idxs in src_to_idx.values():
                guaranteed.add(int(self._rng.choice(idxs)))

            # 找仍孤立的节点（不在已选边的任一端点）
            covered: set[int] = set()
            for i in guaranteed:
                covered.add(pairs[i][0])
                covered.add(pairs[i][1])

            # 对仍孤立、但在 G* 中作为 dst 出现的节点，选一条入边
            for v, idxs in dst_to_idx.items():
                if v not in covered:
                    chosen = int(self._rng.choice(idxs))
                    guaranteed.add(chosen)
                    covered.add(pairs[chosen][0])
                    covered.add(v)

            # 补足到 init_n（允许低于 init_n 若 G* 本身边数不足）
            remaining = max(0, init_n - len(guaranteed))
            pool = [i for i in range(len(pairs)) if i not in guaranteed]
            if remaining > 0 and pool:
                extra = self._rng.choice(pool, size=min(remaining, len(pool)), replace=False)
                guaranteed.update(extra.tolist())
            return [pairs[i] for i in guaranteed]

        raise ValueError(f"未知 init_strategy: {strategy!r}，支持 random/stratified/snowball/forest_fire/all_covered")

    # ── 用户采样 ──────────────────────────────────────────────────────────────

    def sample_active_users(self, round_idx: int) -> list[int]:
        return self._selector.select(round_idx, self.adj)

    # ── 候选过滤 ──────────────────────────────────────────────────────────────

    def mask_existing_edges(self, u: int, cands: list[tuple[int, float]]) -> list[tuple[int, float]]:
        return [(v, s) for v, s in cands if not self.adj.has_edge(u, v) and v != u]

    def cooldown_excluded_nodes(self, u: int, round_idx: int) -> set[int]:
        """返回 u 在 round_idx 轮应排除的目标节点（用于冷启动随机填充）。

        hard 模式：unlock_round > round_idx 的节点仍在 cooldown 窗口内。
        decay 模式：round_idx - reject_round < cooldown_rounds 的节点仍在窗口内。
        """
        excluded: set[int] = set()
        if self._cooldown_mode == "hard":
            for (src, dst), unlock in self._cooldown.items():
                if src == u and unlock > round_idx:
                    excluded.add(dst)
        else:
            for (src, dst), reject_round in self._cooldown.items():
                if src == u and (round_idx - reject_round) < self._cooldown_rounds:
                    excluded.add(dst)
        return excluded

    def set_cooldown_mode(self, mode: str) -> None:
        """切换 cooldown 模式：'hard'（硬排除）或 'decay'（衰减权重）。"""
        if mode not in ("hard", "decay"):
            raise ValueError(f"cooldown_mode 须为 'hard' 或 'decay'，got {mode!r}")
        if mode == self._cooldown_mode:
            return
        N = self._cooldown_rounds
        if mode == "decay":
            # hard→decay：unlock_round = reject_round + N，精确反推 reject_round
            self._cooldown = {k: v - N for k, v in self._cooldown.items()}
        else:
            # decay→hard：reject_round → unlock_round = reject_round + N
            self._cooldown = {k: v + N for k, v in self._cooldown.items()}
        self._cooldown_mode = mode

    def mask_cooldown(
        self, u: int, cands: list[tuple[int, float]], round_idx: int
    ) -> list[tuple[int, float]]:
        if self._cooldown_mode == "hard":
            return [(v, s) for v, s in cands if self._cooldown.get((u, v), 0) <= round_idx]
        # decay 模式：对被拒绝的 pair 施加衰减权重，而不是硬排除
        result = []
        N = self._cooldown_rounds
        for v, s in cands:
            t_reject = self._cooldown.get((u, v))
            if t_reject is None:
                result.append((v, s))
            else:
                dt = round_idx - t_reject
                decay = 1.0 - float(__import__("math").exp(-dt / max(N, 1)))
                result.append((v, s * decay))
        return result

    # ── 环境步骤 ──────────────────────────────────────────────────────────────

    def step(self, recs: dict[int, list[int]], round_idx: int) -> Feedback:
        fb = self._feedback_sim.simulate(recs)

        for u, v in fb.rejected:
            if self._cooldown_mode == "decay":
                self._cooldown[(u, v)] = round_idx          # 记录拒绝时刻
            else:
                self._cooldown[(u, v)] = round_idx + self._cooldown_rounds  # 记录解锁时刻

        for u, v in fb.accepted:
            self._cooldown.pop((u, v), None)
            if (u, v) not in self._accepted_set:
                self.adj.add_edge(u, v)
                self._accepted_set.add((u, v))
                if (u, v) in self._star_set:
                    self._coverage_count += 1

        # 每 10 轮清理过期 cooldown 条目，防内存泄漏
        if round_idx % 10 == 0:
            if self._cooldown_mode == "hard":
                # v = unlock_round，保留还没解锁的
                self._cooldown = {k: v for k, v in self._cooldown.items() if v > round_idx}
            else:
                # v = reject_round，衰减到 e^{-10} 后可丢（约 10×cooldown_rounds 轮后）
                cutoff = round_idx - 10 * self._cooldown_rounds
                self._cooldown = {k: v for k, v in self._cooldown.items() if v > cutoff}

        self._selector.update_after_round(round_idx, fb.accepted)
        return fb

    # ── 指标辅助 ──────────────────────────────────────────────────────────────

    def coverage(self) -> float:
        return self._coverage_count / self._star_size if self._star_size > 0 else 0.0

    def get_adjacency(self) -> StaticAdjacency:
        return self.adj

    def get_observed_edges_df(self) -> pd.DataFrame:
        rows = [(u, v) for u, v in self.adj.iter_edges()]
        return pd.DataFrame(rows, columns=["src", "dst"])

    @property
    def star_set(self) -> set[tuple[int, int]]:
        return self._star_set

    @property
    def n_nodes(self) -> int:
        return self._n
