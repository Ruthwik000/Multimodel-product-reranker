"""Train LambdaRank models and run ablation across variants and seeds."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mmrerank.config import Config, load_config
from mmrerank.data.splits import split_by_query
from mmrerank.evaluation.ablation import run_ablation, summarize_ablation
from mmrerank.evaluation.metrics import per_query_ndcg
from mmrerank.features.builder import build_features
from mmrerank.models.ranker import LambdaRankRanker
from mmrerank.utils.io import ensure_dir, read_parquet, write_text
from mmrerank.utils.logging_setup import get_logger, setup_logging
from mmrerank.utils.seed import set_seed

logger = get_logger(__name__)


def main() -> None:
    """Run training and ablation pipeline."""
    setup_logging()
    cfg = load_config()
    set_seed(cfg.seed)

    # Load processed dataset
    dataset_path = cfg.paths.processed / "dataset.parquet"
    df = read_parquet(dataset_path)

    # Split
    train_df, val_df, test_df = split_by_query(df, val_fraction=cfg.val_fraction, seed=cfg.seed)

    # Run ablation
    seeds = [cfg.seed + i for i in range(cfg.n_seeds)]
    results_df = run_ablation(train_df, val_df, test_df, cfg, seeds)
    summary_df = summarize_ablation(results_df)

    # Save results
    reports_dir = ensure_dir(cfg.paths.reports)
    results_df.to_csv(reports_dir / "ablation_raw.csv", index=False)
    summary_df.to_csv(reports_dir / "ablation_summary.csv", index=False)

    # Save per-query NDCG for the best text_plus_image run
    _save_per_query_ndcg(train_df, val_df, test_df, cfg, reports_dir)

    # Write results table to markdown
    _write_results_md(summary_df, reports_dir)

    # Log to experiment_log.md
    _log_experiment(cfg, summary_df)

    logger.info("Training and ablation complete.")


def _save_per_query_ndcg(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    reports_dir: Path,
) -> None:
    """Train text_only and text_plus_image models, save per-query NDCG."""
    for variant in ["text_only", "text_plus_image"]:
        set_seed(cfg.seed)
        X_train, y_train, g_train, fnames = build_features(train_df, cfg, variant)
        X_val, y_val, g_val, _ = build_features(val_df, cfg, variant)
        X_test, y_test, g_test, _ = build_features(test_df, cfg, variant)

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        ranker = LambdaRankRanker(cfg.lightgbm, seed=cfg.seed)
        ranker.fit(X_train, y_train, g_train, X_val, y_val, g_val)

        preds = ranker.predict(X_test)
        pq = per_query_ndcg(y_test, preds, g_test, cfg.ndcg_k)

        np.save(reports_dir / f"pq_ndcg_{variant}.npy", pq)
        logger.info("Saved per-query NDCG for %s: mean=%.4f", variant, pq.mean())


def _write_results_md(summary_df: pd.DataFrame, reports_dir: Path) -> None:
    """Write ablation summary to reports/results.md."""
    lines = [
        "# Results",
        "",
        "## Ablation Summary",
        "",
        "| Variant | NDCG@10 Mean | NDCG@10 Std | Runs |",
        "|---------|-------------|-------------|------|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {row['variant']} | {row['mean_ndcg']:.4f} | {row['std_ndcg']:.4f} "
            f"| {int(row['n_runs'])} |"
        )
    lines.append("")

    write_text("\n".join(lines), reports_dir / "results.md")


def _log_experiment(cfg: Config, summary_df: pd.DataFrame) -> None:
    """Append experiment results to docs/experiment_log.md."""
    log_path = Path("docs/experiment_log.md")
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    config_hash = cfg.config_hash()

    lines = []
    for _, row in summary_df.iterrows():
        lines.append(
            f"| {now} | {config_hash} | {row['variant']} | "
            f"{row['mean_ndcg']:.4f} | {row['std_ndcg']:.4f} | auto |"
        )

    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    updated = existing.rstrip() + "\n" + "\n".join(lines) + "\n"
    write_text(updated, log_path)


if __name__ == "__main__":
    main()
