"""Tests for utility modules: seed, logging, io."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmrerank.utils.io import read_parquet, write_parquet, write_text
from mmrerank.utils.seed import set_seed


def test_set_seed_reproducible() -> None:
    """Setting the same seed produces same random numbers."""
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_set_seed_different() -> None:
    """Different seeds produce different random numbers."""
    set_seed(42)
    a = np.random.rand(5)
    set_seed(99)
    b = np.random.rand(5)
    assert not np.array_equal(a, b)


def test_write_read_parquet(tmp_path: Path) -> None:
    """Round-trip parquet write and read."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "test.parquet"
    write_parquet(df, path)
    result = read_parquet(path)
    pd.testing.assert_frame_equal(df, result)


def test_write_text(tmp_path: Path) -> None:
    """Write text creates file with correct content."""
    path = tmp_path / "sub" / "test.txt"
    write_text("hello world", path)
    assert path.read_text() == "hello world"
