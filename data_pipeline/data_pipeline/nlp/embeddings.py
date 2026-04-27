"""bge-m3 embeddings wrapper.

Produces dense vectors for filings sections (Risk Factors, MD&A, Market
Risk). The vectors feed two downstream signals:

  1. Q-over-Q cosine drift — see drift.py — for the "Lazy Prices" signal.
  2. Cross-stock similarity — for richer graph edges if we want them.

Why bge-m3
----------
- Top of the MTEB leaderboard for general-English embeddings as of writing.
- 8K context — comfortably handles a full Risk Factors section (median
  ~5K tokens).
- ~600M params, fp16 fits in <2 GB of GPU memory; CPU inference is workable
  for small batches.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default model — chosen for quality / speed balance.
DEFAULT_MODEL_ID = "BAAI/bge-m3"


class EmbeddingsModel:
    """Lazy-loaded sentence-transformers wrapper around bge-m3.

    The model is only loaded on first .encode() call so that CLI scripts
    can import this module without paying the load cost during dry-runs.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[Path] = None,
        device: str = "auto",
        max_length: int = 8192,
    ):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.max_length = max_length
        self._model = None
        self._dim: Optional[int] = None

    def _load(self):
        if self._model is not None:
            return
        # Late import — keeps dataclass tests fast.
        import torch
        from sentence_transformers import SentenceTransformer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading embeddings model %s on %s (cache_dir=%s)",
            self.model_id, self.device, self.cache_dir,
        )
        kwargs: dict = {"device": self.device}
        if self.cache_dir:
            kwargs["cache_folder"] = str(self.cache_dir)
        self._model = SentenceTransformer(self.model_id, **kwargs)
        self._model.max_seq_length = self.max_length
        # bge-m3 dim = 1024
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info("Embeddings model ready (dim=%d)", self._dim)

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def encode(
        self,
        texts: Iterable[str],
        batch_size: int = 8,
        normalize: bool = True,
    ) -> np.ndarray:
        """Encode a list of texts to dense vectors.

        Returns: np.ndarray of shape (n, dim). When normalize=True, vectors
        are L2-normalised (so cosine similarity becomes a dot product).
        """
        self._load()
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        # sentence-transformers handles batching, GPU transfer, mixed-precision.
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)
