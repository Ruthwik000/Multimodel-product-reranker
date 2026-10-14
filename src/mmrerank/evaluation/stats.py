"""Paired bootstrap confidence interval for NDCG differences."""

from __future__ import annotations

import numpy as np

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def paired_bootstrap_ci(
    ndcg_a: np.ndarray,
    ndcg_b: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for mean(ndcg_a) - mean(ndcg_b).

    Args:
        ndcg_a: Per-query NDCG values for system A (e.g., text_plus_image).
        ndcg_b: Per-query NDCG values for system B (e.g., text_only).
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level (default 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with delta, ci_lower, ci_upper, p_value.
    """
    assert len(ndcg_a) == len(ndcg_b), "Arrays must have same length"
    n = len(ndcg_a)
    rng = np.random.RandomState(seed)

    observed_delta = float(np.mean(ndcg_a) - np.mean(ndcg_b))

    # Bootstrap
    deltas = np.zeros(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        deltas[i] = np.mean(ndcg_a[idx]) - np.mean(ndcg_b[idx])

    alpha = 1 - confidence
    ci_lower = float(np.percentile(deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(deltas, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of bootstraps with sign opposite to observed
    if observed_delta >= 0:
        p_value = float(np.mean(deltas <= 0))
    else:
        p_value = float(np.mean(deltas >= 0))

    result = {
        "delta": observed_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "n_queries": n,
        "n_bootstrap": n_bootstrap,
    }

    logger.info(
        "Bootstrap CI: delta=%.4f [%.4f, %.4f], p=%.4f",
        observed_delta,
        ci_lower,
        ci_upper,
        p_value,
    )
    return result
