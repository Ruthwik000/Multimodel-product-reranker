"""Join ESCI data with SQID image embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)

# CLIP embedding dimension
CLIP_DIM = 512


def join_image_embeddings(
    examples: pd.DataFrame,
    product_features: pd.DataFrame,
    query_features: pd.DataFrame,
    product_id_col: str = "product_id",
    query_id_col: str = "query_id",
) -> pd.DataFrame:
    """Attach SQID CLIP embeddings to examples, flagging missing images.

    Adds columns: product_embedding, query_embedding, has_image.
    """
    # Determine the embedding column names in SQID data
    prod_emb_cols = _find_embedding_cols(product_features, "product")
    query_emb_cols = _find_embedding_cols(query_features, "query")

    # Build embedding arrays for products
    prod_lookup = _build_embedding_lookup(
        product_features, product_id_col, prod_emb_cols, "product"
    )
    query_lookup = _build_embedding_lookup(query_features, query_id_col, query_emb_cols, "query")

    # Attach to examples
    result = examples.copy()
    result["product_embedding"] = result[product_id_col].map(prod_lookup)
    result["query_embedding"] = result[query_id_col].map(query_lookup)

    # Flag missing — .map() returns NaN (not None) for missing keys
    result["has_image"] = result["product_embedding"].apply(
        lambda x: 0 if x is None or (isinstance(x, float) and np.isnan(x)) else 1
    )

    n_missing_prod = (result["has_image"] == 0).sum()
    n_missing_query = result["query_embedding"].isna().sum()
    logger.info(
        "Joined embeddings: %d rows, %d missing product images, %d missing query embeddings",
        len(result),
        n_missing_prod,
        n_missing_query,
    )
    return result


def compute_clip_similarity(row: pd.Series) -> float:
    """Compute cosine similarity between query and product CLIP embeddings."""
    q_emb = row.get("query_embedding")
    p_emb = row.get("product_embedding")
    if q_emb is None or p_emb is None:
        return np.nan
    q = np.array(q_emb, dtype=np.float32)
    p = np.array(p_emb, dtype=np.float32)
    norm_q = np.linalg.norm(q)
    norm_p = np.linalg.norm(p)
    if norm_q == 0 or norm_p == 0:
        return np.nan
    return float(np.dot(q, p) / (norm_q * norm_p))


def _find_embedding_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    """Find embedding columns — either a single array column or numbered columns."""
    # Single column with list/array values
    for col in df.columns:
        if "embedding" in col.lower() or "feature" in col.lower():
            if df[col].dtype == object:
                sample = df[col].iloc[0]
                if isinstance(sample, (list, np.ndarray)):
                    return [col]
    # Numbered columns like dim_0, dim_1, ...
    numbered = [c for c in df.columns if c.startswith("dim_") or c.startswith("f_")]
    if numbered:
        return sorted(numbered)
    # All numeric columns except ID columns
    numeric = [
        c for c in df.columns if df[c].dtype in [np.float32, np.float64] and "id" not in c.lower()
    ]
    if numeric:
        logger.info("Using %d numeric columns as %s embeddings", len(numeric), prefix)
        return numeric
    logger.warning("No embedding columns found for %s", prefix)
    return []


def _build_embedding_lookup(
    df: pd.DataFrame,
    id_col: str,
    emb_cols: list[str],
    label: str,
) -> dict:
    """Build a dict mapping ID -> embedding vector."""
    if not emb_cols:
        logger.warning("No embedding columns for %s, returning empty lookup", label)
        return {}

    lookup: dict = {}
    if len(emb_cols) == 1 and df[emb_cols[0]].dtype == object:
        # Single column with list values
        for _, row in df.iterrows():
            key = row[id_col]
            val = row[emb_cols[0]]
            if val is not None:
                lookup[key] = list(val) if isinstance(val, np.ndarray) else val
    else:
        # Multiple numeric columns
        for _, row in df.iterrows():
            key = row[id_col]
            vec = [float(row[c]) for c in emb_cols]
            if not all(np.isnan(v) for v in vec):
                lookup[key] = vec

    logger.info("Built %s embedding lookup: %d entries", label, len(lookup))
    return lookup
