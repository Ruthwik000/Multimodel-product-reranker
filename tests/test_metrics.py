"""Tests for NDCG metrics with hand-computed examples."""

from __future__ import annotations

import numpy as np

from mmrerank.evaluation.metrics import dcg_at_k, mean_ndcg, ndcg_at_k, per_query_ndcg


def test_dcg_at_k_basic() -> None:
    """DCG@3 for a known ranking."""
    # Relevances in ranked order: [3, 2, 1]
    # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.2618 + 0.5 = 4.7618
    relevances = np.array([3.0, 2.0, 1.0])
    dcg = dcg_at_k(relevances, k=3)
    expected = 3.0 / np.log2(2) + 2.0 / np.log2(3) + 1.0 / np.log2(4)
    assert abs(dcg - expected) < 1e-6


def test_dcg_at_k_empty() -> None:
    """DCG of empty array is 0."""
    assert dcg_at_k(np.array([]), k=5) == 0.0


def test_ndcg_perfect_ranking() -> None:
    """NDCG is 1.0 for a perfect ranking."""
    relevances = np.array([3.0, 2.0, 1.0, 0.0])
    assert abs(ndcg_at_k(relevances, k=4) - 1.0) < 1e-6


def test_ndcg_worst_ranking() -> None:
    """NDCG for reversed ranking is less than 1."""
    relevances = np.array([0.0, 1.0, 2.0, 3.0])
    ndcg = ndcg_at_k(relevances, k=4)
    assert 0.0 < ndcg < 1.0


def test_ndcg_all_irrelevant() -> None:
    """NDCG is 0.0 when all items are irrelevant."""
    relevances = np.array([0.0, 0.0, 0.0])
    assert ndcg_at_k(relevances, k=3) == 0.0


def test_ndcg_single_relevant() -> None:
    """NDCG with single relevant item at position 1 is 1.0."""
    relevances = np.array([1.0, 0.0, 0.0])
    assert abs(ndcg_at_k(relevances, k=3) - 1.0) < 1e-6


def test_per_query_ndcg() -> None:
    """Per-query NDCG produces correct number of values."""
    y_true = np.array([1.0, 0.0, 0.5, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
    group_sizes = np.array([3, 3])

    result = per_query_ndcg(y_true, y_pred, group_sizes, k=3)
    assert len(result) == 2
    assert all(0.0 <= v <= 1.0 for v in result)


def test_mean_ndcg() -> None:
    """Mean NDCG is the average of per-query values."""
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_pred = np.array([0.9, 0.1, 0.1, 0.9])  # First query perfect, second reversed
    group_sizes = np.array([2, 2])

    mean = mean_ndcg(y_true, y_pred, group_sizes, k=2)
    assert 0.0 < mean <= 1.0


def test_ndcg_k_cutoff() -> None:
    """NDCG@1 only considers the top-ranked item."""
    relevances = np.array([0.0, 1.0, 1.0])
    ndcg_1 = ndcg_at_k(relevances, k=1)
    assert ndcg_1 == 0.0  # top item has relevance 0
