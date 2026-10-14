"""Image-based features: CLIP cosine similarity and has_image flag."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def compute_clip_similarity(df: pd.DataFrame) -> np.ndarray:
    """Compute cosine similarity between query and product CLIP embeddings."""
    sims = np.full(len(df), np.nan, dtype=np.float32)

    for i, (_, row) in enumerate(df.iterrows()):
        q_emb = row.get("query_embedding")
        p_emb = row.get("product_embedding")
        if q_emb is None or p_emb is None:
            continue
        q = np.array(q_emb, dtype=np.float32)
        p = np.array(p_emb, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        norm_p = np.linalg.norm(p)
        if norm_q == 0 or norm_p == 0:
            continue
        sims[i] = float(np.dot(q, p) / (norm_q * norm_p))

    n_valid = np.isfinite(sims).sum()
    logger.info("CLIP similarity: %d/%d valid", n_valid, len(df))
    return sims


def compute_has_image(df: pd.DataFrame) -> np.ndarray:
    """Binary flag: 1 if product has an image embedding, 0 otherwise."""
    if "has_image" in df.columns:
        return df["has_image"].values.astype(np.float32)
    return (
        df["product_embedding"]
        .apply(lambda x: 0.0 if x is None else 1.0)
        .values.astype(np.float32)
    )
