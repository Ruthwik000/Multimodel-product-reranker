"""Download ESCI and SQID datasets to data/raw/."""

from __future__ import annotations

import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files

from mmrerank.config import load_config
from mmrerank.utils.io import ensure_dir
from mmrerank.utils.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

ESCI_REPO_URL = "https://github.com/amazon-science/esci-data.git"
SQID_HF_REPO = "crossingminds/shopping-queries-image-dataset"


def download_esci(raw_dir: Path) -> None:
    """Download ESCI dataset via shallow clone or direct file download."""
    esci_dir = raw_dir / "esci-data"
    examples_file = raw_dir / "shopping_queries_dataset_examples.parquet"

    if examples_file.exists():
        logger.info("ESCI examples already present at %s", examples_file)
        return

    if esci_dir.exists():
        logger.info("ESCI repo already cloned at %s", esci_dir)
    else:
        logger.info("Cloning ESCI repo (shallow)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", ESCI_REPO_URL, str(esci_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Git clone failed: %s", e.stderr)
            raise

    # Find and copy/link parquet files to raw_dir
    _copy_esci_files(esci_dir, raw_dir)


def _copy_esci_files(esci_dir: Path, raw_dir: Path) -> None:
    """Find ESCI parquet files and copy them to raw_dir."""
    import shutil

    patterns = ["**/*examples*.parquet", "**/*products*.parquet"]
    for pattern in patterns:
        for src in esci_dir.glob(pattern):
            dest = raw_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
                size_mb = dest.stat().st_size / (1024 * 1024)
                logger.info("Copied %s (%.1f MB)", dest.name, size_mb)


def download_sqid(raw_dir: Path) -> None:
    """Download SQID image features from Hugging Face."""
    logger.info("Listing files in SQID repo: %s", SQID_HF_REPO)
    try:
        repo_files = list_repo_files(SQID_HF_REPO)
        logger.info("SQID repo contains %d files: %s", len(repo_files), repo_files)
    except Exception as e:
        logger.warning("Could not list SQID repo files: %s", e)
        repo_files = []

    # Download parquet files
    parquet_files = [f for f in repo_files if f.endswith(".parquet")]
    if not parquet_files:
        logger.warning("No parquet files found in SQID repo, trying known names")
        parquet_files = [
            "product_image_features.parquet",
            "query_image_features.parquet",
        ]

    for filename in parquet_files:
        dest = raw_dir / Path(filename).name
        if dest.exists():
            logger.info("SQID file already present: %s", dest.name)
            continue
        logger.info("Downloading SQID file: %s", filename)
        try:
            downloaded = hf_hub_download(
                repo_id=SQID_HF_REPO,
                filename=filename,
                local_dir=str(raw_dir),
                repo_type="dataset",
            )
            size_mb = Path(downloaded).stat().st_size / (1024 * 1024)
            logger.info("Downloaded %s (%.1f MB)", filename, size_mb)
        except Exception as e:
            logger.error("Failed to download %s: %s", filename, e)


def main() -> None:
    """Entry point for data download."""
    setup_logging()
    cfg = load_config()
    raw_dir = ensure_dir(cfg.paths.raw)

    logger.info("Downloading data to %s", raw_dir)
    download_esci(raw_dir)
    download_sqid(raw_dir)
    logger.info("Download complete.")


if __name__ == "__main__":
    main()
