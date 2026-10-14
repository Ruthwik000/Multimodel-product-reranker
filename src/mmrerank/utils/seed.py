"""Reproducibility: seed all random generators."""

from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set seeds for Python stdlib, numpy."""
    random.seed(seed)
    np.random.seed(seed)
