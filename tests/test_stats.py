"""Tests for paired bootstrap confidence interval."""

from __future__ import annotations

import numpy as np

from mmrerank.evaluation.stats import paired_bootstrap_ci


def test_bootstrap_identical_arrays() -> None:
    """Identical arrays produce delta of 0 and CI containing 0."""
    a = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    ci = paired_bootstrap_ci(a, a, n_bootstrap=1000, seed=42)
    assert abs(ci["delta"]) < 1e-10
    assert ci["ci_lower"] <= 0.0 <= ci["ci_upper"]


def test_bootstrap_clearly_better() -> None:
    """When A is clearly better than B, delta is positive."""
    a = np.array([0.9, 0.95, 0.85, 0.92, 0.88, 0.91, 0.87, 0.93])
    b = np.array([0.3, 0.35, 0.25, 0.32, 0.28, 0.31, 0.27, 0.33])
    ci = paired_bootstrap_ci(a, b, n_bootstrap=5000, seed=42)
    assert ci["delta"] > 0
    assert ci["ci_lower"] > 0  # entire CI above 0


def test_bootstrap_n_queries() -> None:
    """Reports correct number of queries."""
    a = np.random.rand(50)
    b = np.random.rand(50)
    ci = paired_bootstrap_ci(a, b, seed=42)
    assert ci["n_queries"] == 50


def test_bootstrap_reproducible() -> None:
    """Same seed produces same results."""
    a = np.random.rand(20)
    b = np.random.rand(20)
    ci1 = paired_bootstrap_ci(a, b, seed=42)
    ci2 = paired_bootstrap_ci(a, b, seed=42)
    assert ci1["delta"] == ci2["delta"]
    assert ci1["ci_lower"] == ci2["ci_lower"]
