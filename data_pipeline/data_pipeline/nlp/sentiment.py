"""FinBERT sentiment scoring.

FinBERT (ProsusAI/finbert) is the de-facto baseline in finance NLP — a
BERT-base fine-tuned on Reuters financial news for three-class sentiment
(positive / negative / neutral). ~110M params, ~440 MB, runs comfortably
on a V100 or A100; CPU inference is workable too for small batches.

We score at the *sentence* level and aggregate up. Long sections (full
Risk Factors / MD&A) are tokenised and chunked because BERT has 512-token
context. Per-section we report:

  - mean(score) — net sentiment, in [-1, +1]
  - frac_negative — fraction of sentences classified negative
  - frac_positive — fraction classified positive
  - n_sentences — chunk count, useful as a confidence weight
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "ProsusAI/finbert"
LABELS = ("positive", "negative", "neutral")


@dataclass
class SentimentSummary:
    n_sentences: int
    mean_score: float           # +1 = pure positive, -1 = pure negative
    frac_positive: float
    frac_negative: float
    frac_neutral: float


# very simple sentence splitter. Filings text is messy enough that
# spaCy/nltk don't help much.
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z])")


def _split_into_sentences(text: str, min_chars: int = 30) -> list[str]:
    if not text:
        return []
    raw = _SENT_SPLIT.split(text)
    return [s.strip() for s in raw if len(s.strip()) >= min_chars]


class SentimentModel:
    """Lazy-loaded FinBERT wrapper."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: Optional[Path] = None,
        device: str = "auto",
        max_length: int = 256,        # most sentences fit; longer get truncated
    ):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.device = device
        self.max_length = max_length
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "Loading sentiment model %s on %s",
            self.model_id, self.device,
        )
        kwargs: dict = {}
        if self.cache_dir:
            kwargs["cache_dir"] = str(self.cache_dir)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, **kwargs)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **kwargs,
        )
        self._model.eval()
        self._model.to(self.device)
        # Sanity-check label order. ProsusAI/finbert uses
        # {0: positive, 1: negative, 2: neutral} historically — pin via
        # id2label so we can't get it wrong.
        self._id2label = self._model.config.id2label
        logger.info("Sentiment model ready (id2label=%s)", self._id2label)

    def _label_index(self, name: str) -> int:
        for idx, lbl in self._id2label.items():
            if lbl.lower() == name:
                return int(idx)
        raise ValueError(f"FinBERT does not expose label '{name}': {self._id2label}")

    def score_sentences(
        self,
        sentences: Iterable[str],
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return shape (n, 3) softmax probabilities in column order
        (positive, negative, neutral)."""
        import torch

        self._load()
        sentences = [s for s in sentences if s]
        if not sentences:
            return np.zeros((0, 3), dtype=np.float32)

        out_chunks: list[np.ndarray] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # Reorder columns to (positive, negative, neutral).
            ordered = np.column_stack([
                probs[:, self._label_index("positive")],
                probs[:, self._label_index("negative")],
                probs[:, self._label_index("neutral")],
            ])
            out_chunks.append(ordered.astype(np.float32))
        return np.concatenate(out_chunks, axis=0)

    def summarise_section(self, section_text: str, batch_size: int = 32) -> SentimentSummary:
        sentences = _split_into_sentences(section_text)
        if not sentences:
            return SentimentSummary(0, 0.0, 0.0, 0.0, 0.0)
        probs = self.score_sentences(sentences, batch_size=batch_size)
        argmax = probs.argmax(axis=1)   # 0=pos, 1=neg, 2=neu
        n = len(sentences)
        frac_pos = float((argmax == 0).sum() / n)
        frac_neg = float((argmax == 1).sum() / n)
        frac_neu = float((argmax == 2).sum() / n)
        # Mean score: +1 for pos, -1 for neg, 0 for neutral, weighted by p.
        mean_score = float((probs[:, 0] - probs[:, 1]).mean())
        return SentimentSummary(
            n_sentences=n,
            mean_score=mean_score,
            frac_positive=frac_pos,
            frac_negative=frac_neg,
            frac_neutral=frac_neu,
        )
