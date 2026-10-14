"""Query-grouped train/val/test splits with leakage assertions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def split_by_query(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
    query_col: str = "query_id",
    split_col: str = "split",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test by query, using the official split column.

    The test set comes from the official 'split' column (value 'test').
    The train set is further split into train/val using GroupShuffleSplit.
    """
    # Use official split if available
    if split_col in df.columns:
        test_df = df[df[split_col] == "test"].copy()
        train_full = df[df[split_col] == "train"].copy()
        logger.info("Official split: %d train, %d test rows", len(train_full), len(test_df))
    else:
        logger.warning("No '%s' column found, splitting manually", split_col)
        test_df, train_full = _manual_split(df, seed, query_col)

    # Carve validation from train
    train_df, val_df = _split_validation(train_full, val_fraction, seed, query_col)

    # Assert zero query overlap
    _assert_no_overlap(train_df, val_df, test_df, query_col)

    logger.info(
        "Final split: train=%d, val=%d, test=%d rows",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def _split_validation(
    train_full: pd.DataFrame,
    val_fraction: float,
    seed: int,
    query_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation set from train using GroupShuffleSplit."""
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    groups = train_full[query_col].values
    train_idx, val_idx = next(gss.split(train_full, groups=groups))
    train_df = train_full.iloc[train_idx].copy()
    val_df = train_full.iloc[val_idx].copy()
    return train_df, val_df


def _manual_split(
    df: pd.DataFrame, seed: int, query_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fallback manual split when no official split column exists."""
    unique_queries = df[query_col].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_queries)
    n_test = max(1, len(unique_queries) // 5)
    test_queries = set(unique_queries[:n_test])
    test_df = df[df[query_col].isin(test_queries)].copy()
    train_df = df[~df[query_col].isin(test_queries)].copy()
    return test_df, train_df


def _assert_no_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    query_col: str,
) -> None:
    """Assert that no query appears in more than one split."""
    train_q = set(train_df[query_col].unique())
    val_q = set(val_df[query_col].unique())
    test_q = set(test_df[query_col].unique())

    assert train_q.isdisjoint(val_q), f"Train-Val overlap: {train_q & val_q}"
    assert train_q.isdisjoint(test_q), f"Train-Test overlap: {train_q & test_q}"
    assert val_q.isdisjoint(test_q), f"Val-Test overlap: {val_q & test_q}"
    logger.info("✓ No query overlap between splits")
