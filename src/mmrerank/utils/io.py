"""IO helpers for reading and writing parquet and markdown files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file and log its shape."""
    df = pd.read_parquet(path)
    logger.info("Read %s: %d rows x %d cols", path.name, len(df), len(df.columns))
    return df


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %s: %d rows", path.name, len(df))


def write_text(text: str, path: Path) -> None:
    """Write text to a file, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s (%d chars)", path.name, len(text))


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist, return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
