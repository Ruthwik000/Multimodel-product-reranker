"""Feature builder: combine text and image features into X, y, groups."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmrerank.config import Config, GainsConfig
from mmrerank.features.image_features import compute_clip_similarity, compute_has_image
from mmrerank.features.text_features import (
    compute_bm25_scores,
    compute_title_length,
    compute_token_overlap,
)
from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)

TEXT_FEATURES = ["bm25_score", "token_overlap", "title_length"]
IMAGE_FEATURES = ["clip_similarity", "has_image"]
ALL_FEATURES = TEXT_FEATURES + IMAGE_FEATURES


def build_features(
    df: pd.DataFrame,
    cfg: Config,
    variant: str = "text_plus_image",
    cache_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build feature matrix X, labels y, group sizes, and feature names.

    Args:
        df: DataFrame with query-product pairs, sorted by query_id.
        cfg: Configuration object.
        variant: One of 'text_only', 'image_only', 'text_plus_image'.
        cache_dir: Optional directory to cache intermediate features.

    Returns:
        X, y, group_sizes, feature_names
    """
    # Sort by query_id to ensure correct grouping
    df = df.sort_values("query_id").reset_index(drop=True)

    # Compute all features
    feat_dict: dict[str, np.ndarray] = {}
    if variant in ("text_only", "text_plus_image"):
        feat_dict["bm25_score"] = compute_bm25_scores(df)
        feat_dict["token_overlap"] = compute_token_overlap(df)
        feat_dict["title_length"] = compute_title_length(df)

    if variant in ("image_only", "text_plus_image"):
        feat_dict["clip_similarity"] = compute_clip_similarity(df)
        feat_dict["has_image"] = compute_has_image(df)

    feature_names = list(feat_dict.keys())
    X = np.column_stack([feat_dict[f] for f in feature_names]).astype(np.float32)

    # Labels: map ESCI labels to gains
    y = _map_labels_to_gains(df, cfg.gains)

    # Group sizes
    group_sizes = df.groupby("query_id").size().values.astype(np.int32)

    logger.info(
        "Built features: variant=%s, X=%s, %d groups, %d features",
        variant,
        X.shape,
        len(group_sizes),
        len(feature_names),
    )

    # Cache if requested
    if cache_dir is not None:
        _cache_features(X, y, group_sizes, feature_names, variant, cache_dir)

    return X, y, group_sizes, feature_names


def _map_labels_to_gains(df: pd.DataFrame, gains: GainsConfig) -> np.ndarray:
    """Map ESCI labels to numeric gains."""
    label_col = _find_label_col(df)
    gain_map = {
        "E": gains.E,
        "S": gains.S,
        "C": gains.C,
        "I": gains.I,
        "Exact": gains.E,
        "Substitute": gains.S,
        "Complement": gains.C,
        "Irrelevant": gains.I,
    }
    labels = df[label_col].map(gain_map).fillna(gains.I).values.astype(np.float32)
    return labels


def _find_label_col(df: pd.DataFrame) -> str:
    """Find the ESCI label column name."""
    for col in ["esci_label", "label", "relevance_label", "relevance"]:
        if col in df.columns:
            return col
    raise ValueError(f"No label column found in {list(df.columns)}")


def _cache_features(
    X: np.ndarray,
    y: np.ndarray,
    group_sizes: np.ndarray,
    feature_names: list[str],
    variant: str,
    cache_dir: Path,
) -> None:
    """Cache computed features to disk."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"X_{variant}.npy", X)
    np.save(cache_dir / f"y_{variant}.npy", y)
    np.save(cache_dir / f"groups_{variant}.npy", group_sizes)
    logger.info("Cached features for variant '%s' to %s", variant, cache_dir)
