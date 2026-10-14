"""LightGBM LambdaRank ranker wrapper."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from mmrerank.config import LightGBMParams
from mmrerank.utils.logging_setup import get_logger

logger = get_logger(__name__)


class LambdaRankRanker:
    """Wrapper around LGBMRanker for LambdaRank learning-to-rank."""

    def __init__(self, params: LightGBMParams, seed: int = 42) -> None:
        self.params = params
        self.seed = seed
        self.model: lgb.LGBMRanker | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        group_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        group_val: np.ndarray | None = None,
    ) -> LambdaRankRanker:
        """Train the LambdaRank model with optional early stopping on validation.

        LightGBM requires integer labels for ranking. We map float gains to
        integer levels and pass label_gain to configure the gain per level.
        """
        # Map float gains to integer labels for LightGBM
        y_train_int, label_gain = _to_int_labels(y_train)
        y_val_int = _to_int_labels(y_val)[0] if y_val is not None else None

        self.model = lgb.LGBMRanker(
            objective=self.params.objective,
            metric=self.params.metric,
            n_estimators=self.params.n_estimators,
            num_leaves=self.params.num_leaves,
            learning_rate=self.params.learning_rate,
            min_child_samples=self.params.min_child_samples,
            random_state=self.seed,
            verbose=self.params.verbose,
            label_gain=",".join(str(g) for g in label_gain),
        )

        fit_kwargs: dict[str, Any] = {
            "X": X_train,
            "y": y_train_int,
            "group": group_train,
        }

        if X_val is not None and y_val_int is not None and group_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val_int)]
            fit_kwargs["eval_group"] = [group_val]
            fit_kwargs["eval_at"] = self.params.eval_at
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(
                    stopping_rounds=self.params.early_stopping_rounds,
                    verbose=self.params.verbose > 0,
                ),
                lgb.log_evaluation(period=0),
            ]

        self.model.fit(**fit_kwargs)
        best_iter = getattr(self.model, "best_iteration_", self.params.n_estimators)
        logger.info("Trained LambdaRank: best_iteration=%d", best_iter)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict relevance scores."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)

    def feature_importance(self, importance_type: str = "gain") -> np.ndarray:
        """Return feature importances."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.feature_importances_

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Saved model to %s", path)

    def load(self, path: Path) -> LambdaRankRanker:
        """Load model from disk."""
        with open(path, "rb") as f:
            self.model = pickle.load(f)  # noqa: S301
        logger.info("Loaded model from %s", path)
        return self


def _to_int_labels(y: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Map float gain values to integer labels and return the gain list.

    LightGBM ranking requires integer labels. We sort unique gain values
    and assign integer labels 0, 1, 2, ... with label_gain specifying
    the actual gain for each integer level.
    """
    unique_gains = sorted(set(float(v) for v in y))
    gain_to_int = {g: i for i, g in enumerate(unique_gains)}
    y_int = np.array([gain_to_int[float(v)] for v in y], dtype=np.int32)
    return y_int, unique_gains
