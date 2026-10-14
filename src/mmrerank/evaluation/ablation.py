"""Ablation framework: run multiple feature variants and compare."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from mmrerank.config import Config
from mmrerank.evaluation.metrics import per_query_ndcg
from mmrerank.features.builder import build_features
from mmrerank.models.baselines import Bm25OnlyRanker, RandomRanker
from mmrerank.models.ranker import LambdaRankRanker
from mmrerank.utils.logging_setup import get_logger
from mmrerank.utils.seed import set_seed

logger = get_logger(__name__)

VARIANTS = ["text_only", "image_only", "text_plus_image"]
BASELINES = ["random", "bm25_only"]


def run_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    """Run all ablation variants and baselines across multiple seeds.

    Returns a DataFrame with columns: variant, seed, ndcg_mean, ndcg_std.
    """
    if seeds is None:
        seeds = [cfg.seed + i for i in range(cfg.n_seeds)]

    results: list[dict[str, Any]] = []

    # Run baselines (seed-independent except random)
    for seed in seeds:
        results.extend(_run_baselines(test_df, cfg, seed))

    # Run LambdaRank variants
    for variant in VARIANTS:
        for seed in seeds:
            result = _run_variant(train_df, val_df, test_df, cfg, variant, seed)
            results.append(result)

    results_df = pd.DataFrame(results)
    logger.info("Ablation complete: %d results", len(results_df))
    return results_df


def _run_variant(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    """Train and evaluate a single LambdaRank variant."""
    set_seed(seed)
    logger.info("Running variant=%s, seed=%d", variant, seed)

    X_train, y_train, g_train, feat_names = build_features(train_df, cfg, variant)
    X_val, y_val, g_val, _ = build_features(val_df, cfg, variant)
    X_test, y_test, g_test, _ = build_features(test_df, cfg, variant)

    # Replace NaN with 0 for LightGBM
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    ranker = LambdaRankRanker(cfg.lightgbm, seed=seed)
    ranker.fit(X_train, y_train, g_train, X_val, y_val, g_val)

    preds = ranker.predict(X_test)
    pq_ndcg = per_query_ndcg(y_test, preds, g_test, cfg.ndcg_k)

    return {
        "variant": variant,
        "seed": seed,
        "ndcg_mean": float(pq_ndcg.mean()),
        "ndcg_std": float(pq_ndcg.std()),
        "n_queries": len(pq_ndcg),
        "feature_names": feat_names,
    }


def _run_baselines(test_df: pd.DataFrame, cfg: Config, seed: int) -> list[dict[str, Any]]:
    """Run baseline rankers on the test set."""
    results = []

    # Build full features for baselines
    X_test, y_test, g_test, _ = build_features(test_df, cfg, "text_plus_image")
    X_test = np.nan_to_num(X_test, nan=0.0)

    # Random baseline
    random_ranker = RandomRanker(seed=seed)
    random_preds = random_ranker.predict(X_test)
    pq = per_query_ndcg(y_test, random_preds, g_test, cfg.ndcg_k)
    results.append(
        {
            "variant": "random",
            "seed": seed,
            "ndcg_mean": float(pq.mean()),
            "ndcg_std": float(pq.std()),
            "n_queries": len(pq),
            "feature_names": [],
        }
    )

    # BM25-only baseline
    # Build text features to get BM25 as first column
    X_text, y_text, g_text, _ = build_features(test_df, cfg, "text_only")
    X_text = np.nan_to_num(X_text, nan=0.0)
    bm25_ranker = Bm25OnlyRanker()
    bm25_preds = bm25_ranker.predict(X_text)
    pq = per_query_ndcg(y_text, bm25_preds, g_text, cfg.ndcg_k)
    results.append(
        {
            "variant": "bm25_only",
            "seed": seed,
            "ndcg_mean": float(pq.mean()),
            "ndcg_std": float(pq.std()),
            "n_queries": len(pq),
            "feature_names": [],
        }
    )

    return results


def summarize_ablation(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ablation results across seeds."""
    summary = (
        results_df.groupby("variant")
        .agg(
            mean_ndcg=("ndcg_mean", "mean"),
            std_ndcg=("ndcg_mean", "std"),
            n_runs=("ndcg_mean", "count"),
        )
        .reset_index()
        .sort_values("mean_ndcg", ascending=False)
    )
    logger.info("Ablation summary:\n%s", summary.to_string(index=False))
    return summary
