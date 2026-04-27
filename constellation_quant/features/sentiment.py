"""Sentiment aggregation — combine multi-source signals into daily per-ticker features.

Input (from `SentimentDownloader`): long-format parquet per ticker
    columns = [date, source, score ∈ [-1, +1], volume]

Output: wide DataFrame indexed by date with columns:
    sent_composite      — weighted mean across sources
    sent_momentum_5d    — 5-day change in composite
    sent_divergence     — |news - social| (news_sources vs social_sources)
    sent_mention_spike  — total volume / 20d rolling mean

Missing data is filled with neutral (0.0) rather than propagating NaN — per
the project spec, a stock with no sentiment coverage should still contribute
to training. All values are bounded: score in [-1, +1], everything else
clipped to reasonable ranges during normalisation.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Set

import numpy as np
import pandas as pd

from constellation_quant.utils import get_logger

log = get_logger(__name__)


DEFAULT_SOURCE_WEIGHTS = {
    "finviz":     0.4,
    "stocktwits": 0.3,
    "reddit":     0.3,
}

# Source taxonomy for divergence computation.
NEWS_SOURCES:   Set[str] = {"finviz"}
SOCIAL_SOURCES: Set[str] = {"stocktwits", "reddit"}


class SentimentFeatures:
    """Aggregate per-source daily scores into composite features."""

    def __init__(self, config: Optional[Mapping] = None):
        cfg = dict(config or {})
        sources_cfg = cfg.get("sources", {}) or {}
        if sources_cfg:
            self.weights = {
                name: float((sources_cfg.get(name) or {}).get("weight", 0.0))
                for name in DEFAULT_SOURCE_WEIGHTS
            }
        else:
            self.weights = dict(DEFAULT_SOURCE_WEIGHTS)
        self.missing_fill = float(cfg.get("missing_fill", 0.0))
        self.momentum_window = 5
        self.spike_window = 20

    # ── Public API ─────────────────────────────────────────────────────

    def compute(
        self,
        sentiment_frames: Mapping[str, pd.DataFrame],
        daily_index: Optional[pd.DatetimeIndex] = None,
    ) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        for ticker, df in sentiment_frames.items():
            if df is None or df.empty:
                out[ticker] = self._empty_output(daily_index)
                continue
            out[ticker] = self.compute_one(df, daily_index=daily_index)
        return out

    def compute_one(
        self,
        long_df: pd.DataFrame,
        daily_index: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """Aggregate one ticker's long-format sentiment to daily features."""
        df = long_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        composite = self._weighted_composite(df)
        divergence = self._divergence(df)
        volume = df.groupby("date")["volume"].sum()

        if daily_index is not None:
            composite  = composite.reindex(daily_index).fillna(self.missing_fill)
            divergence = divergence.reindex(daily_index).fillna(self.missing_fill)
            volume     = volume.reindex(daily_index).fillna(0.0)

        momentum = composite - composite.shift(self.momentum_window)
        baseline = volume.rolling(self.spike_window, min_periods=1).mean().replace(0.0, np.nan)
        spike = (volume / baseline).fillna(1.0)

        return pd.DataFrame({
            "sent_composite":     composite,
            "sent_momentum_5d":   momentum.fillna(0.0),
            "sent_divergence":    divergence,
            "sent_mention_spike": spike,
        })

    # ── Internals ──────────────────────────────────────────────────────

    def _weighted_composite(self, df: pd.DataFrame) -> pd.Series:
        """Apply per-source weights, then average per date."""
        df = df.copy()
        df["weight"] = df["source"].map(self.weights).fillna(0.0)
        df["weighted_score"] = df["score"] * df["weight"]
        grouped = df.groupby("date")
        return (grouped["weighted_score"].sum() /
                grouped["weight"].sum().replace(0.0, np.nan))

    def _divergence(self, df: pd.DataFrame) -> pd.Series:
        """|mean(news_scores) - mean(social_scores)| per date."""
        news = df[df["source"].isin(NEWS_SOURCES)].groupby("date")["score"].mean()
        social = df[df["source"].isin(SOCIAL_SOURCES)].groupby("date")["score"].mean()
        combined = pd.concat([news.rename("news"), social.rename("social")], axis=1)
        return (combined["news"] - combined["social"]).abs()

    def _empty_output(self, daily_index: Optional[pd.DatetimeIndex]) -> pd.DataFrame:
        idx = daily_index if daily_index is not None else pd.DatetimeIndex([])
        return pd.DataFrame(
            {
                "sent_composite":     self.missing_fill,
                "sent_momentum_5d":   0.0,
                "sent_divergence":    0.0,
                "sent_mention_spike": 1.0,
            },
            index=idx,
        )
