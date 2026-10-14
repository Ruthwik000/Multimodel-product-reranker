"""Evaluate trained models: compute NDCG and statistical tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mmrerank.config import load_config
from mmrerank.evaluation.stats import paired_bootstrap_ci
from mmrerank.utils.io import write_text
from mmrerank.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)


def main() -> None:
    """Run evaluation: load per-query NDCG and compute bootstrap CI."""
    setup_logging()
    cfg = load_config()

    reports_dir = cfg.paths.reports

    # Load per-query NDCG arrays
    pq_text_plus = _load_pq(reports_dir / "pq_ndcg_text_plus_image.npy")
    pq_text_only = _load_pq(reports_dir / "pq_ndcg_text_only.npy")

    if pq_text_plus is None or pq_text_only is None:
        logger.error("Per-query NDCG files not found. Run train.py first.")
        return

    # Ensure same length (use minimum)
    n = min(len(pq_text_plus), len(pq_text_only))
    pq_text_plus = pq_text_plus[:n]
    pq_text_only = pq_text_only[:n]

    # Paired bootstrap CI
    ci_result = paired_bootstrap_ci(pq_text_plus, pq_text_only, seed=cfg.seed)

    # Append to results.md
    _append_ci_results(ci_result, reports_dir / "results.md")
    logger.info("Evaluation complete.")


def _load_pq(path: Path) -> np.ndarray | None:
    """Load per-query NDCG array if it exists."""
    if path.exists():
        return np.load(path)
    logger.warning("File not found: %s", path)
    return None


def _append_ci_results(ci: dict, results_path: Path) -> None:
    """Append bootstrap CI results to results.md."""
    existing = results_path.read_text(encoding="utf-8") if results_path.exists() else ""

    lines = [
        "",
        "## Statistical Test: text_plus_image vs text_only",
        "",
        f"- **Delta (mean NDCG@10)**: {ci['delta']:.4f}",
        f"- **95% CI**: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]",
        f"- **p-value**: {ci['p_value']:.4f}",
        f"- **N queries**: {ci['n_queries']}",
        f"- **N bootstrap samples**: {ci['n_bootstrap']}",
        "",
    ]

    updated = existing.rstrip() + "\n" + "\n".join(lines) + "\n"
    write_text(updated, results_path)


if __name__ == "__main__":
    main()
