"""src/online/evaluator.py — 每轮指标计算。

指标：
- 推荐质量：Precision@K, Recall@K, MRR, NDCG@K（复用 src/utils/metrics.py）
- 覆盖率：|E_t ∩ E*| / |E*|（由 env 增量维护）
- 图结构相似度（每 graph_every_n 轮）：clustering coef 差值、degree KL 散度
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.online.feedback import Feedback
from src.online.static_adj import StaticAdjacency
from src.utils.metrics import compute_hits_at_k, compute_mrr, compute_ndcg_at_k


class RoundMetrics:
    def __init__(
        self,
        star_set: set[tuple[int, int]],
        n_nodes: int,
        k_list: list[int] | None = None,
        graph_every_n: int = 10,
        degree_bins: int = 50,
    ) -> None:
        self._star = star_set
        self._n = n_nodes
        self._ks = k_list or [5, 10, 20]
        self._graph_every = graph_every_n
        self._bins = degree_bins
        self._history: list[dict] = []

        # G* 的 out-degree 分布（固定 reference）
        deg_star = np.zeros(n_nodes, dtype=np.int32)
        for u, _ in star_set:
            deg_star[u] += 1
        self._deg_star = deg_star
        self._deg_hist_star = self._degree_hist(deg_star)

        # 预建 G_star 和 G_t 供 NX 计算复用（避免每轮重建）
        try:
            import networkx as nx
            self._G_star = nx.DiGraph()
            self._G_star.add_nodes_from(range(n_nodes))
            self._G_star.add_edges_from(star_set)
            self._cc_star = nx.average_clustering(self._G_star)
        except Exception:
            self._G_star = None
            self._cc_star = float("nan")

        self._G_t: "object | None" = None  # 延迟初始化
        self._G_t_undirected: "object | None" = None
        self._G_t_edge_count: int = -1  # 上次缓存时的边数

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def update(
        self,
        round_idx: int,
        recs: dict[int, list[int]],
        feedback: Feedback,
        adj: StaticAdjacency,
        coverage: float,
    ) -> dict[str, float]:
        row: dict[str, float] = {"round": round_idx, "coverage": coverage}

        # 推荐质量：对每个用户构造 (pos_scores, neg_scores) 再聚合
        pos_scores_list, neg_scores_list = [], []
        n_rec_total = n_accepted_total = 0
        accepted_set = set(feedback.accepted)

        for u, vs in recs.items():
            if not vs:
                continue
            # 当前轮正样本 = 被接受的 (u,v) 且 (u,v) 不在已有图中（排除历史边）
            pos_v = [v for v in vs if (u, v) in accepted_set]
            neg_v = [v for v in vs if (u, v) not in accepted_set]
            n_rec_total += len(vs)
            n_accepted_total += len(pos_v)

            if pos_v and neg_v:
                # 用均匀分（排序质量用），pos=1 neg=0
                pos_scores_list.append(np.ones(len(pos_v)))
                neg_scores_list.append(np.zeros((len(pos_v), len(neg_v))))

        row["precision_k"] = n_accepted_total / n_rec_total if n_rec_total > 0 else 0.0
        row["n_accepted"] = float(n_accepted_total)
        row["n_recs"] = float(n_rec_total)
        row["n_active_users"] = float(len(recs))
        row["n_skipped_users"] = float(sum(1 for vs in recs.values() if not vs))

        if pos_scores_list:
            pos_arr = np.concatenate(pos_scores_list)  # (N_pos,)
            neg_arr = np.vstack([
                np.pad(n, ((0, 0), (0, max(m.shape[1] for m in neg_scores_list) - n.shape[1])))
                if n.ndim == 2 else n
                for n in neg_scores_list
            ]) if False else None  # 简化：直接用聚合版

            # 聚合 MRR / Hits@K（每个正样本 vs 该用户所有负样本）
            mrr_vals, hits_vals = {k: [] for k in self._ks}, []
            for ps, ns in zip(pos_scores_list, neg_scores_list):
                if ps.size == 0 or ns.size == 0:
                    continue
                for k in self._ks:
                    mrr_vals[k].append(compute_mrr(ps, ns))
                hits_vals.append(compute_hits_at_k(ps, ns, k=self._ks[0]))

            for k in self._ks:
                row[f"mrr@{k}"] = float(np.mean(mrr_vals[k])) if mrr_vals[k] else 0.0
            row[f"hits@{self._ks[0]}"] = float(np.mean(hits_vals)) if hits_vals else 0.0

        # 新增指标：Hit Rate@K、Coverage@K、Novelty
        row.update(self._diversity_metrics(recs, accepted_set, adj, round_idx))

        # 图结构相似度（间隔计算）
        if round_idx % self._graph_every == 0:
            row.update(self._graph_similarity(adj))

        self._history.append(row)
        return row

    # ── 辅助：增量更新 G_t 缓存 ────────────────────────────────────────────────

    def _refresh_G_t(self, adj: StaticAdjacency) -> None:
        """当边数变化时增量更新缓存的有向图和无向图。"""
        cur_edges = adj.num_edges()
        if cur_edges == self._G_t_edge_count:
            return
        try:
            import networkx as nx  # noqa: PLC0415
            if self._G_t is None:
                self._G_t = nx.DiGraph()
                self._G_t.add_nodes_from(range(self._n))
            # add_edges_from 对已存在的边是幂等的，只新增才有效
            self._G_t.add_edges_from(adj.iter_edges())
            self._G_t_undirected = self._G_t.to_undirected()
            self._G_t_edge_count = cur_edges
        except Exception:
            pass

    # ── 多样性指标 ────────────────────────────────────────────────────────────

    def _diversity_metrics(
        self,
        recs: dict[int, list[int]],
        accepted_set: set[tuple[int, int]],
        adj: StaticAdjacency,
        round_idx: int = 0,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        k = self._ks[0]

        # Hit Rate@K：至少命中 1 个正样本的用户比例
        hit_users = 0
        total_users = 0
        for u, vs in recs.items():
            if not vs:
                continue
            total_users += 1
            top_vs = vs[:k]
            if any((u, v) in accepted_set for v in top_vs):
                hit_users += 1
        result[f"hit_rate@{k}"] = hit_users / total_users if total_users > 0 else 0.0

        # Coverage@K：所有推荐覆盖到的不同目标节点数 / n_nodes
        rec_targets: set[int] = set()
        for vs in recs.values():
            rec_targets.update(vs[:k])
        result[f"rec_coverage@{k}"] = len(rec_targets) / self._n if self._n > 0 else 0.0

        # Novelty：仅与 _graph_similarity 同频计算，避免每轮 BFS
        if round_idx % self._graph_every == 0:
            result["novelty"] = self._compute_novelty(recs, adj, k)

        return result

    def _compute_novelty(
        self,
        recs: dict[int, list[int]],
        adj: StaticAdjacency,
        k: int,
        max_pairs: int = 200,
    ) -> float:
        """推荐对在 G_t 中的平均最短路径长度（仅取有限样本以控制开销）。"""
        try:
            import networkx as nx  # noqa: PLC0415
            self._refresh_G_t(adj)
            if self._G_t_undirected is None:
                return float("nan")
            G_undirected = self._G_t_undirected

            pairs = [(u, v) for u, vs in recs.items() for v in vs[:k]]
            if len(pairs) > max_pairs:
                pairs = pairs[:max_pairs]

            lengths = []
            for u, v in pairs:
                try:
                    length = nx.shortest_path_length(G_undirected, u, v)
                    lengths.append(length)
                except nx.NetworkXNoPath:
                    pass
            return float(np.mean(lengths)) if lengths else float("nan")
        except Exception:
            return float("nan")

    # ── 图结构 ────────────────────────────────────────────────────────────────

    def _graph_similarity(self, adj: StaticAdjacency) -> dict[str, float]:
        try:
            import networkx as nx  # noqa: PLC0415
            self._refresh_G_t(adj)
            if self._G_t is None or self._G_star is None:
                return {"clustering_diff": float("nan"), "degree_kl": float("nan")}

            cc_t = nx.average_clustering(self._G_t)
            deg_t = np.array([d for _, d in self._G_t.out_degree()], dtype=np.float32)
            kl = self._kl_divergence(self._degree_hist(deg_t.astype(np.int32)), self._deg_hist_star)

            return {"clustering_diff": abs(cc_t - self._cc_star), "degree_kl": kl}
        except Exception:
            return {"clustering_diff": float("nan"), "degree_kl": float("nan")}

    def _degree_hist(self, degrees: np.ndarray) -> np.ndarray:
        hist, _ = np.histogram(degrees, bins=self._bins, range=(0, max(degrees.max(), 1)))
        hist = hist.astype(np.float64) + 1e-8  # smoothing
        return hist / hist.sum()

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        return float(np.sum(p * np.log(p / q)))

    # ── 结果输出 ──────────────────────────────────────────────────────────────

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._history)
