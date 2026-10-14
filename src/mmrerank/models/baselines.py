"""Baseline rankers: random and BM25-only."""

from __future__ import annotations

import numpy as np

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


class RandomRanker:
    """Baseline: random ranking scores."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return random scores."""
        rng = np.random.RandomState(self.seed)
        scores = rng.uniform(size=X.shape[0])
        logger.info("RandomRanker: generated %d random scores", len(scores))
        return scores.astype(np.float32)


class Bm25OnlyRanker:
    """Baseline: rank by BM25 score only (assumes first feature is BM25)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return BM25 scores (first column of X)."""
        scores = X[:, 0].astype(np.float32)
        logger.info("BM25OnlyRanker: using %d scores", len(scores))
        return scores
