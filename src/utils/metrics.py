"""src/utils/metrics.py — AUC, AP, Hits@K, MRR, NDCG@K"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC。y_true: {0,1}，y_score: 预测概率。"""
    return float(roc_auc_score(y_true, y_score))


def compute_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision（area under PR curve）。"""
    return float(average_precision_score(y_true, y_score))


def _ranks(pos_scores: np.ndarray, neg_scores: np.ndarray) -> np.ndarray:
    """计算每个正样本在候选集中的排名（1-based）。

    Args:
        pos_scores: (N,)    正样本得分
        neg_scores: (N, M)  每个正样本对应的 M 个负样本得分

    Returns:
        ranks: (N,) int，每个正样本的排名（1 = 最高）
    """
    assert pos_scores.ndim == 1
    assert neg_scores.ndim == 2
    assert pos_scores.shape[0] == neg_scores.shape[0]
    # 排名 = 比正样本分数更高的负样本数 + 1
    return (neg_scores > pos_scores[:, None]).sum(axis=1) + 1


def compute_hits_at_k(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    k: int,
) -> float:
    """Hits@K：正样本排名 ≤ k 的比例。

    Args:
        pos_scores: (N,)    正样本得分
        neg_scores: (N, M)  每个正样本对应的 M 个负样本得分
        k:          命中阈值

    Returns:
        命中率（0 ~ 1）
    """
    ranks = _ranks(pos_scores, neg_scores)
    return float((ranks <= k).mean())


def compute_mrr(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
) -> float:
    """MRR（Mean Reciprocal Rank）。

    MRR = mean(1 / rank)，rank=1 时 MRR=1.0，rank 越大 MRR 越小。
    对精排任务比 AUC 更直接：衡量正样本能否排到候选集最前面。

    Args:
        pos_scores: (N,)    正样本得分
        neg_scores: (N, M)  每个正样本对应的 M 个负样本得分
    """
    ranks = _ranks(pos_scores, neg_scores).astype(np.float64)
    return float((1.0 / ranks).mean())


def compute_ndcg_at_k(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    k: int,
) -> float:
    """NDCG@K（Normalized Discounted Cumulative Gain）。

    每个 query 只有 1 个正样本，所以理想 DCG = 1/log2(2) = 1.0，
    NDCG@K = 1/log2(rank+1) if rank <= k else 0。

    Args:
        pos_scores: (N,)    正样本得分
        neg_scores: (N, M)  每个正样本对应的 M 个负样本得分
        k:          截断阈值
    """
    ranks = _ranks(pos_scores, neg_scores).astype(np.float64)
    dcg = np.where(ranks <= k, 1.0 / np.log2(ranks + 1), 0.0)
    return float(dcg.mean())


def compute_ranking_metrics(
    scores_by_query: dict[int, tuple[np.ndarray, np.ndarray]],
    k_list: list[int] | None = None,
) -> dict[str, float]:
    """按 query 分组计算排序指标（simulated_recall 协议专用）。

    Args:
        scores_by_query:  {query_id: (pos_scores_1d, neg_scores_2d)}
                          每个 query 对应一组正/负样本分数
        k_list:           需要计算的 K 值列表

    Returns:
        {'mrr': ..., 'ndcg@10': ..., 'ndcg@20': ..., 'hits@10': ..., ...}
    """
    if k_list is None:
        k_list = [10, 20, 50]

    all_pos = np.concatenate([v[0] for v in scores_by_query.values()])
    all_neg = np.concatenate([v[1] for v in scores_by_query.values()], axis=0)

    result: dict[str, float] = {"mrr": compute_mrr(all_pos, all_neg)}
    for k in k_list:
        result[f"ndcg@{k}"] = compute_ndcg_at_k(all_pos, all_neg, k)
        result[f"hits@{k}"] = compute_hits_at_k(all_pos, all_neg, k)

    return result


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    pos_scores: np.ndarray | None = None,
    neg_scores: np.ndarray | None = None,
    k_list: list[int] | None = None,
    include_ranking: bool = False,
) -> dict[str, float]:
    """计算全部评估指标，返回字典。

    Args:
        include_ranking:  True 时额外计算 MRR / NDCG@K（需传入 pos_scores / neg_scores）
    """
    if k_list is None:
        k_list = [10, 20, 50]

    result: dict[str, float] = {
        "auc": compute_auc(y_true, y_score),
        "ap": compute_ap(y_true, y_score),
    }

    if pos_scores is not None and neg_scores is not None:
        for k in k_list:
            result[f"hits@{k}"] = compute_hits_at_k(pos_scores, neg_scores, k)
        if include_ranking:
            result["mrr"] = compute_mrr(pos_scores, neg_scores)
            for k in k_list:
                result[f"ndcg@{k}"] = compute_ndcg_at_k(pos_scores, neg_scores, k)

    return result
