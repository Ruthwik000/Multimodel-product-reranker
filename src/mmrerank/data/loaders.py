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
    candidates = [
        cfg.paths.raw / "product_features.parquet",
        cfg.paths.raw / "product_image_features.parquet",
        cfg.paths.raw / "sqid_product_features.parquet",
        cfg.paths.raw / "data" / "product_features.parquet",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    df = pd.read_parquet(path)
    logger.info("Loaded SQID product features: %d rows, columns: %s", len(df), list(df.columns))
    return df


def load_sqid_query_features(cfg: Config) -> pd.DataFrame:
    """Load SQID query features (CLIP query embeddings)."""
    candidates = [
        cfg.paths.raw / "query_features.parquet",
        cfg.paths.raw / "query_image_features.parquet",
        cfg.paths.raw / "sqid_query_features.parquet",
        cfg.paths.raw / "data" / "query_features.parquet",
    ]
    path = next((p for p in candidates if p.exists()), candidates[0])
    df = pd.read_parquet(path)
    logger.info("Loaded SQID query features: %d rows, columns: %s", len(df), list(df.columns))
    return df


def filter_locale(df: pd.DataFrame, locale: str) -> pd.DataFrame:
    """Filter a DataFrame to a specific locale."""
    for col in ["product_locale", "query_locale", "locale"]:
        if col in df.columns:
            filtered = df[df[col] == locale].copy()
            logger.info(
                "Filtered on '%s' to '%s': %d -> %d rows", col, locale, len(df), len(filtered)
            )
            return filtered
    logger.warning("No locale column found, returning all rows")
    return df


def filter_reduced(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to the reduced (small_version) subset if the column exists."""
    if "small_version" in df.columns:
        filtered = df[df["small_version"] == 1].copy()
        logger.info("Filtered to reduced version: %d -> %d rows", len(df), len(filtered))
        return filtered
    logger.warning("No 'small_version' column found, returning all rows")
    return df
