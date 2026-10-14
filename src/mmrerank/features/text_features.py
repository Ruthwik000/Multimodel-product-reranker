"""Text-based features: BM25 score, token overlap, title length."""

from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def compute_bm25_scores(df: pd.DataFrame, query_col: str = "query") -> np.ndarray:
    """Compute BM25 score of query vs product title within each query group.

    For each query group, builds a BM25 index over candidate product titles
    and scores the query against each title.
    """
    title_col = _find_title_col(df)
    scores = np.zeros(len(df), dtype=np.float32)

    for query_id, group in df.groupby("query_id"):
        idx = group.index
        titles = group[title_col].fillna("").astype(str).tolist()
        query_text = str(group[query_col].iloc[0])

        tokenized_titles = [t.lower().split() for t in titles]
        tokenized_query = query_text.lower().split()

        if not any(tokenized_titles):
            scores[idx] = 0.0
            continue

        bm25 = BM25Okapi(tokenized_titles)
        group_scores = bm25.get_scores(tokenized_query)
        scores[idx] = group_scores.astype(np.float32)

    return scores


def compute_token_overlap(df: pd.DataFrame, query_col: str = "query") -> np.ndarray:
    """Fraction of query tokens found in the product title."""
    title_col = _find_title_col(df)
    overlaps = np.zeros(len(df), dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        query_tokens = set(str(row[query_col]).lower().split())
        title_tokens = set(str(row.get(title_col, "")).lower().split())
        if not query_tokens:
            overlaps[i] = 0.0
        else:
            overlaps[i] = len(query_tokens & title_tokens) / len(query_tokens)

    return overlaps


def compute_title_length(df: pd.DataFrame) -> np.ndarray:
    """Length of the product title in tokens."""
    title_col = _find_title_col(df)
    return (
        df[title_col]
        .fillna("")
        .astype(str)
        .apply(lambda t: len(t.split()))
        .values.astype(np.float32)
    )


def _find_title_col(df: pd.DataFrame) -> str:
    """Find the product title column name."""
    for col in ["product_title", "title", "product_name"]:
        if col in df.columns:
            return col
    raise ValueError(f"No title column found in {list(df.columns)}")
