"""Compute NLP features (embeddings + sentiment + drift) over parsed filings.

Reads the combined parsed-filings parquet, runs bge-m3 on each section,
runs FinBERT on each section's sentences, then computes Q-over-Q drift.

Usage:
    python -m data_pipeline.scripts.compute_nlp_features \
        --filings-parquet /data/.../data/processed/edgar/_combined.parquet \
        --out-dir /data/.../data/processed/nlp_features \
        --hf-cache /data/.../data/hf_cache
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SECTIONS = ["risk_factors", "mda", "market_risk"]


def _truncate_for_embedding(text: str, max_chars: int = 40_000) -> str:
    """bge-m3 has 8K-token context. ~40k chars is a safe cap for English."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    # Take the start (where Risk Factors / MD&A intros live) plus the tail.
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + " ... " + text[-tail:]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--filings-parquet", required=True, type=Path,
                        help="Combined parsed-filings parquet (one row per filing).")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Output directory for embeddings.parquet + sentiment.parquet + drift.parquet.")
    parser.add_argument("--hf-cache", required=True, type=Path,
                        help="HuggingFace cache directory (pre-downloaded models).")
    parser.add_argument("--device", default="auto", help="cuda | cpu | auto")
    parser.add_argument("--embedding-batch", type=int, default=8)
    parser.add_argument("--sentiment-batch", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N filings (smoke-test).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.filings_parquet)
    if args.limit is not None:
        df = df.head(args.limit).copy()
    logger.info("Loaded %d filings from %s", len(df), args.filings_parquet)

    # ---------------- embeddings
    from data_pipeline.nlp.embeddings import EmbeddingsModel
    from data_pipeline.nlp.sentiment import SentimentModel

    embedder = EmbeddingsModel(cache_dir=args.hf_cache, device=args.device)
    sentimenter = SentimentModel(cache_dir=args.hf_cache, device=args.device)

    embed_rows: list[dict] = []
    sent_rows: list[dict] = []

    start = time.time()

    for section in SECTIONS:
        texts = df[section].fillna("").tolist()
        clean_texts = [_truncate_for_embedding(t) for t in texts]
        # only encode non-empty
        idx_non_empty = [i for i, t in enumerate(clean_texts) if t]
        non_empty_texts = [clean_texts[i] for i in idx_non_empty]

        logger.info(
            "Encoding section=%s — %d non-empty / %d total",
            section, len(non_empty_texts), len(clean_texts),
        )
        if non_empty_texts:
            vecs = embedder.encode(non_empty_texts, batch_size=args.embedding_batch)
        else:
            vecs = np.zeros((0, embedder.dim), dtype=np.float32)

        # one row per filing × section
        v_iter = iter(vecs)
        for i, row in df.iterrows():
            embedding: list[float]
            if i in idx_non_empty:
                embedding = next(v_iter).tolist()
            else:
                embedding = [0.0] * embedder.dim
            embed_rows.append({
                "cik": row["cik"],
                "accession": row["accession"],
                "form": row["form"],
                "filing_date": row["filing_date"],
                "period": row["period"],
                "section_name": section,
                "embedding": embedding,
                "len_chars": len(clean_texts[i] if isinstance(i, int) else ""),
            })

    emb_df = pd.DataFrame(embed_rows)
    emb_path = args.out_dir / "embeddings.parquet"
    emb_df.to_parquet(emb_path, index=False, compression="snappy")
    logger.info("Wrote %d rows → %s", len(emb_df), emb_path)

    # ---------------- sentiment
    for section in SECTIONS:
        logger.info("Sentiment scoring section=%s", section)
        for _, row in df.iterrows():
            text = row[section] or ""
            summary = sentimenter.summarise_section(
                text, batch_size=args.sentiment_batch,
            )
            sent_rows.append({
                "cik": row["cik"],
                "accession": row["accession"],
                "form": row["form"],
                "filing_date": row["filing_date"],
                "period": row["period"],
                "section": section,
                "n_sentences": summary.n_sentences,
                "mean_score": summary.mean_score,
                "frac_positive": summary.frac_positive,
                "frac_negative": summary.frac_negative,
                "frac_neutral": summary.frac_neutral,
            })

    sent_df = pd.DataFrame(sent_rows)
    sent_path = args.out_dir / "sentiment.parquet"
    sent_df.to_parquet(sent_path, index=False, compression="snappy")
    logger.info("Wrote %d rows → %s", len(sent_df), sent_path)

    # ---------------- drift features
    from data_pipeline.nlp.drift import compute_drift_features

    drift_df = compute_drift_features(emb_df, section="mda")
    drift_path = args.out_dir / "drift.parquet"
    drift_df.to_parquet(drift_path, index=False, compression="snappy")
    logger.info("Wrote %d rows → %s", len(drift_df), drift_path)

    elapsed = time.time() - start
    logger.info("Done — total elapsed %.1f s", elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
