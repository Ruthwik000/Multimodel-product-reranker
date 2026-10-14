"""Tests for config module."""

from __future__ import annotations

from pathlib import Path

import yaml

from mmrerank.config import Config, GainsConfig, PathsConfig, load_config


def test_config_defaults() -> None:
    """Config defaults are sane."""
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.locale == "us"
    assert cfg.ndcg_k == 10
    assert cfg.n_train_queries == 3000
    assert cfg.n_test_queries == 1000


def test_gains_defaults() -> None:
    """ESCI gains have expected defaults."""
    g = GainsConfig()
    assert g.E == 1.0
    assert g.S == 0.1
    assert g.C == 0.01
    assert g.I == 0.0


def test_paths_are_pathlib() -> None:
    """Paths config uses pathlib.Path."""
    p = PathsConfig()
    assert isinstance(p.raw, Path)
    assert isinstance(p.processed, Path)


def test_load_config_missing_file() -> None:
    """Loading from a nonexistent file returns defaults."""
    cfg = load_config(Path("nonexistent_config.yaml"))
    assert cfg.seed == 42


def test_load_config_from_yaml(tmp_path: Path) -> None:
    """Loading from a valid YAML overrides defaults."""
    yaml_content = {"seed": 123, "locale": "jp", "ndcg_k": 5}
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(yaml.dump(yaml_content))

    cfg = load_config(yaml_path)
    assert cfg.seed == 123
    assert cfg.locale == "jp"
    assert cfg.ndcg_k == 5


def test_config_hash_deterministic() -> None:
    """Config hash is deterministic."""
    cfg1 = Config()
    cfg2 = Config()
    assert cfg1.config_hash() == cfg2.config_hash()


def test_config_hash_changes() -> None:
    """Different configs produce different hashes."""
    cfg1 = Config(seed=42)
    cfg2 = Config(seed=99)
    assert cfg1.config_hash() != cfg2.config_hash()
