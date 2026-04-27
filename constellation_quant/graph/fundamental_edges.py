"""Fundamental-similarity edges via cosine similarity over fundamental vectors.

For each prediction date, take each stock's current fundamental feature
vector (P/E, P/B, D/E, growth rate, market cap, ...), pairwise cosine-
similarity them, and connect pairs with similarity > threshold.

This captures relationships that sector labels miss: two stocks in different
sectors but with very similar financial profiles often respond to macro
events in related ways (e.g. a high-growth software name and a high-growth
consumer-discretionary name both re-rate with the same duration factor).

Because fundamentals update at most quarterly, this graph is effectively
stable within a quarter — the builder exposes that directly via a
`refresh_frequency` parameter for cache-friendly reuse.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from constellation_quant.graph.correlation_edges import EdgeSpec


class FundamentalEdgeBuilder:
    """Cosine-similarity edges over a per-ticker fundamental feature vector.

    Args:
        threshold: Similarity cutoff in [-1, 1]; edges kept when > threshold.
        feature_columns: Optional list of columns to use; defaults to all
            numeric columns in the supplied DataFrame.
        top_k: If set, keep top-K most similar neighbours regardless of
            threshold.
        min_valid_features: Minimum non-NaN feature count for a ticker to
            be connectable. Below that, the ticker is excluded (no edges).
    """

    def __init__(
        self,
        threshold: float = 0.7,
        feature_columns: Optional[Sequence[str]] = None,
        top_k: Optional[int] = None,
        min_valid_features: int = 2,
    ):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        self.threshold = threshold
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.top_k = top_k
        self.min_valid = min_valid_features

    def build(
        self,
        vectors: pd.DataFrame,
        universe_tickers: Sequence[str],
    ) -> EdgeSpec:
        """Build cosine-similarity edges at a single date.

        Args:
            vectors: DataFrame indexed by ticker, columns = fundamental
                features. Rows with too many NaNs are dropped.
            universe_tickers: Canonical node ordering.
        """
        n = len(universe_tickers)
        if n < 2 or vectors.empty:
            return _empty_spec(n)

        cols = self.feature_columns or [
            c for c in vectors.columns
            if pd.api.types.is_numeric_dtype(vectors[c])
        ]
        if not cols:
            return _empty_spec(n)

        # Align and filter.
        valid = vectors.loc[vectors.index.isin(universe_tickers), cols]
        valid = valid.dropna(thresh=self.min_valid)
        if len(valid) < 2:
            return _empty_spec(n)

        arr = valid.fillna(0.0).to_numpy(dtype=np.float32)
        sim = _cosine_similarity(arr)
        np.fill_diagonal(sim, 0.0)

        if self.top_k is not None:
            k = min(self.top_k, len(valid) - 1)
            topk = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
            src = np.repeat(np.arange(len(valid)), k).astype(np.int64)
            dst = topk.reshape(-1).astype(np.int64)
            mask = src != dst
            src_local, dst_local = src[mask], dst[mask]
        else:
            src_local, dst_local = np.where(sim > self.threshold)

        if src_local.size == 0:
            return _empty_spec(n)

        weight_values = sim[src_local, dst_local].astype(np.float32)

        # Remap to universe indexing.
        members = list(valid.index)
        lookup = {t: i for i, t in enumerate(universe_tickers)}
        src_global = np.array([lookup[members[i]] for i in src_local], dtype=np.int64)
        dst_global = np.array([lookup[members[i]] for i in dst_local], dtype=np.int64)

        edge_index = np.stack([src_global, dst_global], axis=0)
        return EdgeSpec(
            edge_index=edge_index,
            edge_weight=weight_values,
            num_nodes=n,
        )


def _cosine_similarity(a: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of row vectors in `a`. Zero-vec rows → 0."""
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    a_unit = a / norms
    return (a_unit @ a_unit.T).astype(np.float32)


def _empty_spec(n: int) -> EdgeSpec:
    return EdgeSpec(
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=np.zeros((0,),  dtype=np.float32),
        num_nodes=n,
    )
