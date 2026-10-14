"""Typed configuration loaded from YAML."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    """File-system paths for all pipeline stages."""

    raw: Path = Path("data/raw")
    interim: Path = Path("data/interim")
    processed: Path = Path("data/processed")
    reports: Path = Path("reports")
    figures: Path = Path("reports/figures")
    configs: Path = Path("configs")


@dataclass
class LightGBMParams:
    """LightGBM LambdaRank hyper-parameters."""

    objective: str = "lambdarank"
    metric: str = "ndcg"
    n_estimators: int = 200
    num_leaves: int = 31
    learning_rate: float = 0.1
    min_child_samples: int = 20
    eval_at: list[int] = field(default_factory=lambda: [10])
    early_stopping_rounds: int = 30
    verbose: int = -1


@dataclass
class GainsConfig:
    """ESCI relevance label gains for NDCG computation."""

    E: float = 1.0  # noqa: E741
    S: float = 0.1  # noqa: E741
    C: float = 0.01
    I: float = 0.0  # noqa: E741


@dataclass
class VisualLexicon:
    """Configurable lexicon for visual-query detection."""

    words: list[str] = field(
        default_factory=lambda: [
            "red",
            "blue",
            "green",
            "black",
            "white",
            "pink",
            "purple",
            "yellow",
            "orange",
            "brown",
            "gold",
            "silver",
            "grey",
            "gray",
            "striped",
            "floral",
            "plaid",
            "polka",
            "checkered",
            "printed",
            "leather",
            "cotton",
            "silk",
            "wool",
            "denim",
            "velvet",
            "linen",
            "satin",
            "suede",
            "lace",
            "mesh",
            "knit",
            "vintage",
            "modern",
            "retro",
            "minimalist",
            "boho",
            "elegant",
        ]
    )


@dataclass
class Config:
    """Top-level experiment configuration."""

    seed: int = 42
    locale: str = "us"
    use_reduced_version: bool = True
    n_train_queries: int = 3000
    n_test_queries: int = 1000
    val_fraction: float = 0.15
    ndcg_k: int = 10
    n_seeds: int = 3
    gains: GainsConfig = field(default_factory=GainsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    lightgbm: LightGBMParams = field(default_factory=LightGBMParams)
    visual_lexicon: VisualLexicon = field(default_factory=VisualLexicon)

    def config_hash(self) -> str:
        """Return a short hash of the config for experiment tracking."""
        d = _to_dict(self)
        raw = json.dumps(d, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]


def load_config(path: Path | str | None = None) -> Config:
    """Load a Config from a YAML file, falling back to defaults."""
    if path is None:
        path = Path("configs/default.yaml")
    path = Path(path)
    if not path.exists():
        return Config()
    with open(path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}
    return _from_dict(raw)


def _from_dict(raw: dict[str, Any]) -> Config:
    """Recursively build a Config from a dictionary."""
    gains = GainsConfig(**raw.pop("gains", {}))
    paths_raw = raw.pop("paths", {})
    paths = PathsConfig(**{k: Path(v) for k, v in paths_raw.items()})
    lgbm = LightGBMParams(**raw.pop("lightgbm", {}))
    lexicon_raw = raw.pop("visual_lexicon", {})
    lexicon = VisualLexicon(**lexicon_raw)
    return Config(gains=gains, paths=paths, lightgbm=lgbm, visual_lexicon=lexicon, **raw)


def _to_dict(obj: Any) -> Any:
    """Recursively convert dataclasses / Paths to plain dicts / strings."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj
