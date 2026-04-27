"""Pre-download NLP model weights into a local HuggingFace cache.

Run this once on the cluster (ideally via the matching SLURM script) before
running compute_nlp_features so that the GPU job doesn't need outbound
internet to fetch models.

Usage:
    python -m data_pipeline.scripts.download_models \
        --cache-dir /data/EECS-Theory/Zahir_DAYNGRAPH500/data/hf_cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "BAAI/bge-m3",            # embeddings
    "ProsusAI/finbert",       # sentiment
    # Optional, for Phase E (LLM extraction). Uncomment if/when needed.
    # "Qwen/Qwen2.5-14B-Instruct",
]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--cache-dir", required=True, type=Path,
        help="Local HuggingFace cache directory.",
    )
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="HuggingFace model IDs to download.",
    )
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Late import — keeps `--help` snappy and lets this script function as a
    # smoke test even if torch isn't installed yet.
    from huggingface_hub import snapshot_download

    for model_id in args.models:
        logger.info("Downloading %s ...", model_id)
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=args.cache_dir,
            local_dir_use_symlinks=False,
        )
        logger.info("  → %s", path)

    logger.info("All models downloaded into %s", args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
