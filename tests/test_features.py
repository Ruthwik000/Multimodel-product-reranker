"""Tests for feature builders: text and image features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mmrerank.config import Config
from mmrerank.features.builder import build_features
from mmrerank.features.image_features import compute_clip_similarity, compute_has_image
from mmrerank.features.text_features import (
    compute_bm25_scores,
    compute_title_length,
    compute_token_overlap,
)


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic DataFrame for feature computation."""
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2],
            "product_id": ["p1", "p2", "p3", "p4", "p5"],
            "query": ["red shoes", "red shoes", "red shoes", "laptop bag", "laptop bag"],
            "product_title": [
                "Red Running Shoes",
                "Blue Dress Shoes",
                "Shoe Laces",
                "Laptop Sleeve Bag",
                "Phone Case",
            ],
            "esci_label": ["E", "S", "I", "E", "C"],
            "product_embedding": [[0.1, 0.2], [0.3, 0.4], None, [0.5, 0.6], [0.7, 0.8]],
            "query_embedding": [[0.1, 0.3], [0.1, 0.3], [0.1, 0.3], [0.2, 0.4], [0.2, 0.4]],
            "has_image": [1, 1, 0, 1, 1],
        }
    )


def test_bm25_scores(feature_df: pd.DataFrame) -> None:
    """BM25 scores are computed per query group."""
    scores = compute_bm25_scores(feature_df)
    assert len(scores) == len(feature_df)
    assert scores.dtype == np.float32
    # "Red Running Shoes" should score higher for "red shoes" than "Shoe Laces"
    assert scores[0] > scores[2]


def test_token_overlap(feature_df: pd.DataFrame) -> None:
    """Token overlap correctly computes fraction of query tokens in title."""
    overlaps = compute_token_overlap(feature_df)
    assert len(overlaps) == len(feature_df)
    # "Red Running Shoes" contains "red" and "shoes" -> 2/2 = 1.0
    assert overlaps[0] == 1.0
    # "Shoe Laces" contains neither "red" nor "shoes" exactly
    assert overlaps[2] < 1.0


def test_title_length(feature_df: pd.DataFrame) -> None:
    """Title length counts tokens correctly."""
    lengths = compute_title_length(feature_df)
    assert len(lengths) == len(feature_df)
    assert lengths[0] == 3  # "Red Running Shoes"
    assert lengths[4] == 2  # "Phone Case"


def test_clip_similarity(feature_df: pd.DataFrame) -> None:
    """CLIP similarity is NaN for missing embeddings."""
    sims = compute_clip_similarity(feature_df)
    assert len(sims) == len(feature_df)
    assert np.isnan(sims[2])  # missing product_embedding
    assert np.isfinite(sims[0])  # both embeddings present


def test_has_image(feature_df: pd.DataFrame) -> None:
    """Has image flag is correct."""
    flags = compute_has_image(feature_df)
    assert flags[0] == 1.0
    assert flags[2] == 0.0


def test_build_features_text_only(feature_df: pd.DataFrame) -> None:
    """Text-only variant produces 3 features."""
    cfg = Config()
    X, y, groups, names = build_features(feature_df, cfg, "text_only")
    assert X.shape[1] == 3
    assert set(names) == {"bm25_score", "token_overlap", "title_length"}
    assert len(y) == len(feature_df)
    assert groups.sum() == len(feature_df)


def test_build_features_image_only(feature_df: pd.DataFrame) -> None:
    """Image-only variant produces 2 features."""
    cfg = Config()
    X, y, groups, names = build_features(feature_df, cfg, "image_only")
    assert X.shape[1] == 2
    assert set(names) == {"clip_similarity", "has_image"}


def test_build_features_text_plus_image(feature_df: pd.DataFrame) -> None:
    """Text+image variant produces 5 features."""
    cfg = Config()
    X, y, groups, names = build_features(feature_df, cfg, "text_plus_image")
    assert X.shape[1] == 5
