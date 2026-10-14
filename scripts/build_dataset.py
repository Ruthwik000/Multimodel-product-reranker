"""Build the processed dataset: load, filter, sample, join, and save."""

from __future__ import annotations

from mmrerank.config import load_config
from mmrerank.data.join import join_image_embeddings
from mmrerank.data.loaders import (
    filter_locale,
    filter_reduced,
    load_esci_examples,
    load_esci_products,
    load_sqid_product_features,
    load_sqid_query_features,
)
from mmrerank.data.sampling import sample_queries
from mmrerank.utils.io import ensure_dir, write_parquet
from mmrerank.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Build the processed dataset from raw data."""
    setup_logging()
    cfg = load_config()

    logger.info("Loading raw data...")
    examples = load_esci_examples(cfg)
    products = load_esci_products(cfg)
    prod_features = load_sqid_product_features(cfg)
    query_features = load_sqid_query_features(cfg)

    # Filter locale and reduced version
    examples = filter_locale(examples, cfg.locale)
    if cfg.use_reduced_version:
        examples = filter_reduced(examples)

    # Sample queries for train and test first
    total_queries = cfg.n_train_queries + cfg.n_test_queries
    sampled = sample_queries(examples, total_queries, cfg.seed)

    # Merge product titles into sampled examples
    product_cols = ["product_id"]
    if "product_title" in products.columns:
        product_cols.append("product_title")
    if "product_title" not in sampled.columns:
        sampled = sampled.merge(
            products[product_cols].drop_duplicates(subset=["product_id"]),
            on="product_id",
            how="left",
        )

    # Join image embeddings
    joined = join_image_embeddings(
        sampled,
        prod_features,
        query_features,
        product_id_col="product_id",
        query_id_col="query_id",
    )

    # Save
    out_dir = ensure_dir(cfg.paths.processed)
    write_parquet(joined, out_dir / "dataset.parquet")
    logger.info("Build complete. Saved to %s", out_dir / "dataset.parquet")


if __name__ == "__main__":
    main()
