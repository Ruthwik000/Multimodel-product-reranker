"""Load raw ESCI and SQID data files."""

from __future__ import annotations

import pandas as pd

from mmrerank.config import Config
from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def load_esci_examples(cfg: Config) -> pd.DataFrame:
    """Load the ESCI examples (query-product pairs with labels)."""
    raw = cfg.paths.raw
    # Try the reduced version first if configured
    if cfg.use_reduced_version:
        path = raw / "shopping_queries_dataset_examples.parquet"
    else:
        path = raw / "shopping_queries_dataset_examples.parquet"

    df = pd.read_parquet(path)
    logger.info("Loaded ESCI examples: %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_esci_products(cfg: Config) -> pd.DataFrame:
    """Load the ESCI product catalogue."""
    path = cfg.paths.raw / "shopping_queries_dataset_products.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded ESCI products: %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_sqid_product_features(cfg: Config) -> pd.DataFrame:
    """Load SQID product image features (CLIP embeddings)."""
    path = cfg.paths.raw / "product_image_features.parquet"
    if not path.exists():
        # Try alternative name
        path = cfg.paths.raw / "sqid_product_features.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded SQID product features: %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_sqid_query_features(cfg: Config) -> pd.DataFrame:
    """Load SQID query features (CLIP query embeddings)."""
    path = cfg.paths.raw / "query_image_features.parquet"
    if not path.exists():
        path = cfg.paths.raw / "sqid_query_features.parquet"
    df = pd.read_parquet(path)
    logger.info("Loaded SQID query features: %d rows, columns: %s", len(df), list(df.columns))
    return df


def filter_locale(df: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Filter a DataFrame to a specific locale."""
    col = "query_locale" if "query_locale" in df.columns else "locale"
    if col not in df.columns:
        logger.warning("No locale column found, returning all rows")
        return df
    filtered = df[df[col] == locale].copy()
    logger.info("Filtered to locale '%s': %d -> %d rows", locale, len(df), len(filtered))
    return filtered


def filter_reduced(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the reduced (small_version) subset if the column exists."""
    if "small_version" in df.columns:
        filtered = df[df["small_version"] == 1].copy()
        logger.info("Filtered to reduced version: %d -> %d rows", len(df), len(filtered))
        return filtered
    logger.warning("No 'small_version' column found, returning all rows")
    return df
