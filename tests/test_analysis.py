"""Tests for error analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from mmrerank.evaluation.analysis import classify_visual_queries


@pytest.fixture
def query_df() -> pd.DataFrame:
    """Synthetic queries for visual classification."""
    return pd.DataFrame(
        {
            "query_id": [1, 2, 3, 4, 5],
            "query": [
                "red leather bag",
                "laptop charger",
                "blue silk dress",
                "usb cable",
                "vintage wooden table",
            ],
        }
    )


def test_classify_visual(query_df: pd.DataFrame) -> None:
    """Visual queries are correctly identified."""
    words = ["red", "blue", "leather", "silk", "vintage"]
    labels = classify_visual_queries(query_df, words)
    assert labels.iloc[0] == "visual"  # "red leather bag"
    assert labels.iloc[1] == "non-visual"  # "laptop charger"
    assert labels.iloc[2] == "visual"  # "blue silk dress"
    assert labels.iloc[3] == "non-visual"  # "usb cable"
    assert labels.iloc[4] == "visual"  # "vintage wooden table"


def test_classify_empty_lexicon(query_df: pd.DataFrame) -> None:
    """Empty lexicon classifies everything as non-visual."""
    labels = classify_visual_queries(query_df, [])
    assert (labels == "non-visual").all()


def test_classify_case_insensitive(query_df: pd.DataFrame) -> None:
    """Classification is case-insensitive."""
    words = ["RED", "Blue"]
    labels = classify_visual_queries(query_df, words)
    assert labels.iloc[0] == "visual"  # "red" matches "RED"
    assert labels.iloc[2] == "visual"  # "blue" matches "Blue"
