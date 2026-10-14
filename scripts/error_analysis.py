"""Error analysis: visual vs non-visual, top improved/degraded, feature importance."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from mmrerank.config import Config, load_config
from mmrerank.data.splits import split_by_query
from mmrerank.evaluation.analysis import (
    plot_feature_importance,
    top_improved_degraded,
    visual_nonvisual_ndcg,
)
from mmrerank.features.builder import build_features
from mmrerank.models.ranker import LambdaRankRanker
from mmrerank.utils.io import ensure_dir, read_parquet, write_text
from mmrerank.utils.logging_setup import get_logger, setup_logging
from mmrerank.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    """Run error analysis pipeline."""
    setup_logging()
    cfg = load_config()
    set_seed(cfg.seed)

    # Load data and split
    df = read_parquet(cfg.paths.processed / "dataset.parquet")
    train_df, val_df, test_df = split_by_query(df, val_fraction=cfg.val_fraction, seed=cfg.seed)

    figures_dir = ensure_dir(cfg.paths.figures)
    reports_dir = ensure_dir(cfg.paths.reports)

    # Train text_only and text_plus_image models
    text_only_preds, text_only_fnames = _train_and_predict(
        train_df, val_df, test_df, cfg, "text_only"
    )
    text_plus_preds, text_plus_fnames = _train_and_predict(
        train_df, val_df, test_df, cfg, "text_plus_image"
    )

    # Visual vs non-visual analysis
    test_sorted = test_df.sort_values("query_id").reset_index(drop=True)
    vn_results = visual_nonvisual_ndcg(test_sorted, text_plus_preds, cfg)
    logger.info("Visual vs non-visual:\n%s", vn_results.to_string(index=False))

    # Top improved / degraded
    improved, degraded = top_improved_degraded(
        test_sorted, text_only_preds, text_plus_preds, cfg, n=15
    )

    # Feature importance
    ranker = _train_full_model(train_df, val_df, cfg)
    plot_feature_importance(ranker, text_plus_fnames, figures_dir / "feature_importance.png")

    # Write analysis report
    _write_analysis_report(vn_results, improved, degraded, reports_dir)

    logger.info("Error analysis complete.")


def _train_and_predict(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    variant: str,
) -> tuple[np.ndarray, list[str]]:
    """Train a model and return test predictions."""
    set_seed(cfg.seed)
    X_train, y_train, g_train, fnames = build_features(train_df, cfg, variant)
    X_val, y_val, g_val, _ = build_features(val_df, cfg, variant)
    X_test, y_test, g_test, _ = build_features(test_df, cfg, variant)

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)

    ranker = LambdaRankRanker(cfg.lightgbm, seed=cfg.seed)
    ranker.fit(X_train, y_train, g_train, X_val, y_val, g_val)
    return ranker.predict(X_test), fnames


def _train_full_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cfg: Config,
) -> LambdaRankRanker:
    """Train a text_plus_image model and return it."""
    set_seed(cfg.seed)
    X_train, y_train, g_train, fnames = build_features(train_df, cfg, "text_plus_image")
    X_val, y_val, g_val, _ = build_features(val_df, cfg, "text_plus_image")

    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)

    ranker = LambdaRankRanker(cfg.lightgbm, seed=cfg.seed)
    ranker.fit(X_train, y_train, g_train, X_val, y_val, g_val)
    return ranker


def _write_analysis_report(
    vn_results: pd.DataFrame,
    improved: pd.DataFrame,
    degraded: pd.DataFrame,
    reports_dir: Path,
) -> None:
    """Write error analysis results to reports/results.md."""
    results_path = reports_dir / "results.md"
    existing = results_path.read_text(encoding="utf-8") if results_path.exists() else ""

    lines = [
        "",
        "## Visual vs Non-Visual Query Analysis",
        "",
        "| Group | N Queries | NDCG@10 Mean | NDCG@10 Std |",
        "|-------|-----------|-------------|-------------|",
    ]
    for _, row in vn_results.iterrows():
        lines.append(
            f"| {row['group']} | {row['n_queries']} | "
            f"{row['ndcg_mean']:.4f} | {row['ndcg_std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top 15 Improved Queries (text_plus_image vs text_only)",
            "",
            "| Query | NDCG Before | NDCG After | Delta |",
            "|-------|------------|------------|-------|",
        ]
    )
    for _, row in improved.head(15).iterrows():
        q = str(row["query"])[:50]
        lines.append(
            f"| {q} | {row['ndcg_before']:.4f} | {row['ndcg_after']:.4f} | {row['delta']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Top 15 Degraded Queries (text_plus_image vs text_only)",
            "",
            "| Query | NDCG Before | NDCG After | Delta |",
            "|-------|------------|------------|-------|",
        ]
    )
    for _, row in degraded.head(15).iterrows():
        q = str(row["query"])[:50]
        lines.append(
            f"| {q} | {row['ndcg_before']:.4f} | {row['ndcg_after']:.4f} | {row['delta']:.4f} |"
        )

    lines.append("")
    updated = existing.rstrip() + "\n" + "\n".join(lines) + "\n"
    write_text(updated, results_path)


if __name__ == "__main__":
    main()
