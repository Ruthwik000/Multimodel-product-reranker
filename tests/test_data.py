"""Tests for data loaders, sampling, and join modules."""

from __future__ import annotations

import pandas as pd
import pytest

from mmrerank.data.join import join_image_embeddings
from mmrerank.data.loaders import filter_locale, filter_reduced
from mmrerank.data.sampling import sample_queries


@pytest.fixture
def synthetic_examples() -> pd.DataFrame:
    """Synthetic ESCI-like examples DataFrame."""
    return pd.DataFrame(
        {
            "query_id": [1, 1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6],
            "product_id": [f"p{i}" for i in range(16)],
            "query": [
                "red shoes",
                "red shoes",
                "red shoes",
                "laptop",
                "laptop",
                "blue dress",
                "blue dress",
                "blue dress",
                "blue dress",
                "phone case",
                "phone case",
                "leather bag",
                "leather bag",
                "leather bag",
                "headphones",
                "headphones",
            ],
            "product_title": [f"Product {i}" for i in range(16)],
            "esci_label": [
                "E",
                "S",
                "I",
                "E",
                "C",
                "S",
                "E",
                "I",
                "C",
                "E",
                "S",
                "I",
                "E",
                "S",
                "C",
                "E",
            ],
            "query_locale": ["us"] * 12 + ["es"] * 4,
            "small_version": [1] * 10 + [0] * 6,
            "split": ["train"] * 10 + ["test"] * 6,
        }
    )


def test_filter_locale(synthetic_examples: pd.DataFrame) -> None:
    """Filter to US locale keeps only US rows."""
    result = filter_locale(synthetic_examples, "us")
    assert len(result) == 12
    assert (result["query_locale"] == "us").all()


def test_filter_locale_other(synthetic_examples: pd.DataFrame) -> None:
    """Filter to ES locale keeps only ES rows."""
    result = filter_locale(synthetic_examples, "es")
    assert len(result) == 4


def test_filter_reduced(synthetic_examples: pd.DataFrame) -> None:
    """Filter to reduced version keeps only small_version=1."""
    result = filter_reduced(synthetic_examples)
    assert len(result) == 10


def test_sample_queries(synthetic_examples: pd.DataFrame) -> None:
    """Sampling N queries keeps all candidates for each sampled query."""
    result = sample_queries(synthetic_examples, n_queries=3, seed=42)
    sampled_qids = result["query_id"].unique()
    assert len(sampled_qids) == 3
    # Every product for each sampled query should be present
    for qid in sampled_qids:
        orig_count = len(synthetic_examples[synthetic_examples["query_id"] == qid])
        sampled_count = len(result[result["query_id"] == qid])
        assert orig_count == sampled_count


def test_sample_queries_reproducible(synthetic_examples: pd.DataFrame) -> None:
    """Same seed produces same sample."""
    r1 = sample_queries(synthetic_examples, 3, seed=42)
    r2 = sample_queries(synthetic_examples, 3, seed=42)
    pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))


def test_join_image_embeddings(synthetic_examples: pd.DataFrame) -> None:
    """Join adds embedding columns and has_image flag."""
    prod_features = pd.DataFrame(
        {
            "product_id": ["p0", "p1", "p3"],
            "embedding": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        }
    )
    query_features = pd.DataFrame(
        {
            "query_id": [1, 2],
            "embedding": [[0.7, 0.8], [0.9, 1.0]],
        }
    )
    result = join_image_embeddings(synthetic_examples, prod_features, query_features)
    assert "has_image" in result.columns
    assert "product_embedding" in result.columns
    assert "query_embedding" in result.columns
    # p0, p1, p3 have images in prod_features; p2 doesn't
    assert result.loc[result["product_id"] == "p0", "has_image"].iloc[0] == 1
    assert result.loc[result["product_id"] == "p2", "has_image"].iloc[0] == 0
