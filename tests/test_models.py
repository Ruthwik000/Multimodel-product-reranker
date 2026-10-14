"""Tests for models: LambdaRank ranker and baselines."""

from __future__ import annotations

import numpy as np
import pytest

from mmrerank.config import LightGBMParams
from mmrerank.models.baselines import Bm25OnlyRanker, RandomRanker
from mmrerank.models.ranker import LambdaRankRanker


@pytest.fixture
def tiny_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Tiny synthetic data for ranker testing."""
    np.random.seed(42)
    n = 30
    X = np.random.rand(n, 3).astype(np.float32)
    y = np.random.choice([0.0, 0.01, 0.1, 1.0], size=n).astype(np.float32)
    # 6 groups of 5
    groups = np.array([5] * 6, dtype=np.int32)
    return X, y, groups, X, y, groups


def test_random_ranker() -> None:
    """Random ranker produces scores of correct length."""
    X = np.random.rand(10, 3).astype(np.float32)
    ranker = RandomRanker(seed=42)
    scores = ranker.predict(X)
    assert len(scores) == 10
    assert scores.dtype == np.float32


def test_random_ranker_reproducible() -> None:
    """Same seed produces same random scores."""
    X = np.random.rand(10, 3).astype(np.float32)
    r1 = RandomRanker(seed=42).predict(X)
    r2 = RandomRanker(seed=42).predict(X)
    np.testing.assert_array_equal(r1, r2)


def test_bm25_only_ranker() -> None:
    """BM25-only ranker returns first column."""
    X = np.array([[0.5, 0.1, 0.2], [0.9, 0.3, 0.4]], dtype=np.float32)
    ranker = Bm25OnlyRanker()
    scores = ranker.predict(X)
    np.testing.assert_array_almost_equal(scores, [0.5, 0.9])


def test_lambdarank_fit_predict(tiny_data: tuple) -> None:
    """LambdaRank can fit and predict on tiny data."""
    X_tr, y_tr, g_tr, X_te, y_te, g_te = tiny_data
    params = LightGBMParams(n_estimators=10, verbose=-1)
    ranker = LambdaRankRanker(params, seed=42)
    ranker.fit(X_tr, y_tr, g_tr, X_te, y_te, g_te)
    preds = ranker.predict(X_te)
    assert len(preds) == len(X_te)


def test_lambdarank_not_fitted() -> None:
    """Predict before fit raises RuntimeError."""
    params = LightGBMParams()
    ranker = LambdaRankRanker(params)
    with pytest.raises(RuntimeError):
        ranker.predict(np.array([[1.0, 2.0, 3.0]]))


def test_lambdarank_save_load(tiny_data: tuple, tmp_path) -> None:
    """Model can be saved and loaded."""
    X_tr, y_tr, g_tr, X_te, y_te, g_te = tiny_data
    params = LightGBMParams(n_estimators=10, verbose=-1)
    ranker = LambdaRankRanker(params, seed=42)
    ranker.fit(X_tr, y_tr, g_tr)

    path = tmp_path / "model.pkl"
    ranker.save(path)

    loaded = LambdaRankRanker(params).load(path)
    preds_orig = ranker.predict(X_te)
    preds_loaded = loaded.predict(X_te)
    np.testing.assert_array_almost_equal(preds_orig, preds_loaded)
