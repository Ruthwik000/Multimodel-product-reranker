"""Seeded query sampling that preserves all candidate products per query."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def sample_queries(
    df: pd.DataFrame,
    n_queries: int,
    seed: int,
    query_col: str = "query_id",
) -> pd.DataFrame:
    """Sample n_queries unique queries and return all their rows.

    Keeps every candidate product for each sampled query.
    """
    unique_queries = df[query_col].unique()
    n_queries = min(n_queries, len(unique_queries))
    rng = np.random.RandomState(seed)
    sampled = rng.choice(unique_queries, size=n_queries, replace=False)
    result = df[df[query_col].isin(sampled)].copy()
    logger.info(
        "Sampled %d queries (%d rows) from %d available",
        n_queries,
        len(result),
        len(unique_queries),
    )
    return result
