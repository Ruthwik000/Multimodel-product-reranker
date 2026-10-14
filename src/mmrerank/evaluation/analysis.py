"""Error analysis: visual vs non-visual queries, improvement/degradation tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmrerank.config import Config
from mmrerank.evaluation.metrics import per_query_ndcg
from mmrerank.features.builder import build_features
from mmrerank.models.ranker import LambdaRankRanker
from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


def classify_visual_queries(
    df: pd.DataFrame,
    visual_words: list[str],
    query_col: str = "query",
) -> pd.Series:
    """Classify each query as 'visual' or 'non-visual' based on lexicon match."""
    visual_set = {w.lower() for w in visual_words}

    def _is_visual(query: str) -> str:
        tokens = set(str(query).lower().split())
        return "visual" if tokens & visual_set else "non-visual"

    return df[query_col].apply(_is_visual)


def visual_nonvisual_ndcg(
    df: pd.DataFrame,
    preds: np.ndarray,
    cfg: Config,
    query_col: str = "query",
) -> pd.DataFrame:
    """Compute NDCG separately for visual vs non-visual queries."""
    df = df.sort_values("query_id").reset_index(drop=True)

    _, y, group_sizes, _ = build_features(df, cfg, "text_plus_image")
    pq_ndcg = per_query_ndcg(y, preds, group_sizes, cfg.ndcg_k)

    # Classify each unique query as visual or non-visual
    query_visual = (
        df.drop_duplicates("query_id")[query_col]
        .apply(
            lambda q: (
                "visual"
                if set(str(q).lower().split()) & {w.lower() for w in cfg.visual_lexicon.words}
                else "non-visual"
            )
        )
        .values
    )

    results = []
    for label in ["visual", "non-visual"]:
        mask = query_visual == label
        if mask.sum() > 0:
            results.append(
                {
                    "group": label,
                    "n_queries": int(mask.sum()),
                    "ndcg_mean": float(pq_ndcg[mask].mean()),
                    "ndcg_std": float(pq_ndcg[mask].std()),
                }
            )

    return pd.DataFrame(results)


def top_improved_degraded(
    df: pd.DataFrame,
    preds_before: np.ndarray,
    preds_after: np.ndarray,
    cfg: Config,
    n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find queries that improved/degraded the most between two rankers."""
    df = df.sort_values("query_id").reset_index(drop=True)
    _, y, group_sizes, _ = build_features(df, cfg, "text_plus_image")

    ndcg_before = per_query_ndcg(y, preds_before, group_sizes, cfg.ndcg_k)
    ndcg_after = per_query_ndcg(y, preds_after, group_sizes, cfg.ndcg_k)
    deltas = ndcg_after - ndcg_before

    unique_qids = df.drop_duplicates("query_id")["query_id"].values
    unique_queries = df.drop_duplicates("query_id")["query"].values

    delta_df = pd.DataFrame(
        {
            "query_id": unique_qids,
            "query": unique_queries,
            "ndcg_before": ndcg_before,
            "ndcg_after": ndcg_after,
            "delta": deltas,
        }
    )

    improved = delta_df.nlargest(n, "delta")
    degraded = delta_df.nsmallest(n, "delta")

    return improved, degraded


def plot_feature_importance(
    ranker: LambdaRankRanker,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Plot and save feature importance bar chart."""
    importances = ranker.feature_importance()

    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_idx = np.argsort(importances)
    ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color="#4C72B0",
    )
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("LambdaRank Feature Importance")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance plot to %s", output_path)
