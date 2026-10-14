"""Tests for data splits with leakage assertions."""

from __future__ import annotations

import pandas as pd
import pytest

from mmrerank.data.splits import split_by_query


@pytest.fixture
def synthetic_split_df() -> pd.DataFrame:
    """Synthetic DataFrame with official split column."""
    rows = []
    # 10 train queries, each with 3 products
    for q in range(10):
        for p in range(3):
            rows.append(
                {
                    "query_id": q,
                    "product_id": f"p{q}_{p}",
                    "query": f"query {q}",
                    "product_title": f"Product {q}_{p}",
                    "esci_label": "E",
                    "split": "train",
                }
            )
    # 5 test queries
    for q in range(10, 15):
        for p in range(3):
            rows.append(
                {
                    "query_id": q,
                    "product_id": f"p{q}_{p}",
                    "query": f"query {q}",
                    "product_title": f"Product {q}_{p}",
                    "esci_label": "S",
                    "split": "test",
                }
            )
    return pd.DataFrame(rows)


def test_split_no_overlap(synthetic_split_df: pd.DataFrame) -> None:
    """Train, val, and test have zero query overlap."""
    train, val, test = split_by_query(synthetic_split_df, val_fraction=0.2, seed=42)
    train_q = set(train["query_id"].unique())
    val_q = set(val["query_id"].unique())
    test_q = set(test["query_id"].unique())

    assert train_q.isdisjoint(val_q)
    assert train_q.isdisjoint(test_q)
    assert val_q.isdisjoint(test_q)


def test_split_preserves_all_rows(synthetic_split_df: pd.DataFrame) -> None:
    """All rows are present across splits."""
    train, val, test = split_by_query(synthetic_split_df, val_fraction=0.2, seed=42)
    total = len(train) + len(val) + len(test)
    assert total == len(synthetic_split_df)


def test_split_test_from_official(synthetic_split_df: pd.DataFrame) -> None:
    """Test set comes from the official split column."""
    _, _, test = split_by_query(synthetic_split_df, val_fraction=0.2, seed=42)
    test_qids = set(test["query_id"].unique())
    # Original test queries are 10-14
    assert test_qids == {10, 11, 12, 13, 14}


def test_split_val_fraction(synthetic_split_df: pd.DataFrame) -> None:
    """Validation fraction is approximately correct."""
    train, val, _ = split_by_query(synthetic_split_df, val_fraction=0.2, seed=42)
    train_queries = len(train["query_id"].unique())
    val_queries = len(val["query_id"].unique())
    total_train_queries = train_queries + val_queries
    actual_fraction = val_queries / total_train_queries
    assert 0.1 <= actual_fraction <= 0.4  # loose bounds for small data
