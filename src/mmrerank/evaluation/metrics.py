"""NDCG and DCG metrics with configurable ESCI gains."""

from __future__ import annotations

import numpy as np

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute DCG@k for a single ranked list.

    Args:
        relevances: Relevance scores in ranked order (position 0 = rank 1).
        k: Cutoff position.

    Returns:
        DCG@k score.
    """
    relevances = np.asarray(relevances, dtype=np.float64)[:k]
    if len(relevances) == 0:
        return 0.0
    positions = np.arange(1, len(relevances) + 1, dtype=np.float64)
    discounts = np.log2(positions + 1)
    return float(np.sum(relevances / discounts))


def ndcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Compute NDCG@k for a single ranked list.

    Args:
        relevances: Relevance scores in ranked order.
        k: Cutoff position.

    Returns:
        NDCG@k in [0, 1]. Returns 0.0 if the ideal DCG is zero.
    """
    actual_dcg = dcg_at_k(relevances, k)
    ideal_order = np.sort(relevances)[::-1]
    ideal_dcg = dcg_at_k(ideal_order, k)
    if ideal_dcg == 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def per_query_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_sizes: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Compute NDCG@k for each query group.

    Args:
        y_true: True relevance labels (gains).
        y_pred: Predicted scores.
        group_sizes: Number of items per query group.
        k: Cutoff.

    Returns:
        Array of per-query NDCG@k values.
    """
    ndcgs = []
    offset = 0
    for size in group_sizes:
        end = offset + size
        true_group = y_true[offset:end]
        pred_group = y_pred[offset:end]

        # Sort by predicted score descending
        ranked_indices = np.argsort(-pred_group)
        ranked_relevances = true_group[ranked_indices]

        ndcgs.append(ndcg_at_k(ranked_relevances, k))
        offset = end

    result = np.array(ndcgs, dtype=np.float64)
    logger.info("Per-query NDCG@%d: mean=%.4f, std=%.4f", k, result.mean(), result.std())
    return result


def mean_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_sizes: np.ndarray,
    k: int = 10,
) -> float:
    """Compute mean NDCG@k over all queries."""
    return float(per_query_ndcg(y_true, y_pred, group_sizes, k).mean())
