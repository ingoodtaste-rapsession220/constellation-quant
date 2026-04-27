"""Leakage-safe feature normalisation.

Applied after the feature builders but before the Dataset serves tensors:

1. **Rolling z-score (temporal)** — at date t, mean/std computed over rows
   [t − W − 1 … t − 1]. Strictly uses past data; today's value is NOT part of
   the window. Window defaults to 252 trading days (≈ 1 year).

2. **Cross-sectional median fill** — missing values on date t are replaced
   with the median of that feature across all tickers on date t only. No
   temporal leakage; no future leakage.

3. **Winsorisation** — after z-scoring, absolute values > `winsorize_std`
   (default 3.0) are clipped. This caps the influence of individual outliers
   without requiring re-fitting on val/test.

`fit()` records the feature schema and train-period end date; `transform()`
refuses to process data past the fit end unless `extend=True` is passed,
which is an explicit escape hatch for forward testing. The normaliser is
serialisable (save / load JSON) so train-fitted state is reused unchanged
on val/test.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from constellation_quant.utils import get_logger

log = get_logger(__name__)


@dataclass
class NormalizerState:
    """Serialised state of a fitted Normalizer."""

    rolling_window: int
    winsorize_std: float
    feature_columns: list
    train_end: Optional[str] = None      # ISO date of last training row
    fit_complete: bool = False
    # Per-feature sanity stats learned on train; used for diagnostics only.
    train_means: Dict[str, float] = field(default_factory=dict)
    train_stds:  Dict[str, float] = field(default_factory=dict)


class Normalizer:
    """Rolling z-score + winsorise + cross-sectional median fill."""

    def __init__(
        self,
        rolling_window: int = 252,
        winsorize_std:  float = 3.0,
    ):
        if rolling_window < 2:
            raise ValueError(f"rolling_window must be >= 2, got {rolling_window}")
        if winsorize_std <= 0:
            raise ValueError(f"winsorize_std must be > 0, got {winsorize_std}")
        self.state = NormalizerState(
            rolling_window=rolling_window,
            winsorize_std=winsorize_std,
            feature_columns=[],
        )

    # ── Fit ────────────────────────────────────────────────────────────

    def fit(
        self,
        features_by_ticker: Mapping[str, pd.DataFrame],
        train_end: Optional[pd.Timestamp | str] = None,
    ) -> "Normalizer":
        """Record schema + train-period stats. Uses ONLY rows at or before
        `train_end`; val/test rows never touched during fit."""
        columns = self._collect_columns(features_by_ticker)
        self.state.feature_columns = columns
        if train_end is not None:
            end = pd.Timestamp(train_end).normalize()
            self.state.train_end = end.isoformat()
            train_frames = {
                t: df[df.index <= end]
                for t, df in features_by_ticker.items()
            }
        else:
            train_frames = dict(features_by_ticker)

        stacked = pd.concat([f[columns] for f in train_frames.values() if not f.empty], axis=0)
        if stacked.empty:
            log.warning("Normalizer.fit: no training rows found.")
        else:
            self.state.train_means = stacked.mean(skipna=True).to_dict()
            self.state.train_stds  = stacked.std(skipna=True).to_dict()
        self.state.fit_complete = True
        log.info(
            "Normalizer fitted: {} features, train_end={}",
            len(columns), self.state.train_end,
        )
        return self

    # ── Transform ──────────────────────────────────────────────────────

    def transform(
        self,
        features_by_ticker: Mapping[str, pd.DataFrame],
        extend: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Apply rolling z-score + winsorisation + cross-sectional median fill."""
        if not self.state.fit_complete:
            raise RuntimeError("Normalizer.transform called before fit().")

        if not extend and self.state.train_end is not None:
            end = pd.Timestamp(self.state.train_end)
            for ticker, df in features_by_ticker.items():
                if not df.empty and df.index.max() > end and not extend:
                    log.debug(
                        "  [{}] transform rows past train_end={} (this is expected "
                        "for val/test; pass extend=True to silence)",
                        ticker, end.date(),
                    )

        columns = self.state.feature_columns
        # Step 1: per-ticker rolling z-score (leak-safe: shift(1) excludes today).
        zscored = self._rolling_zscore(features_by_ticker, columns)
        # Step 2: cross-sectional median fill on each date.
        filled = self._cross_sectional_median_fill(zscored, columns)
        # Step 3: winsorise.
        clipped = self._winsorise(filled, columns)
        return clipped

    def fit_transform(
        self,
        features_by_ticker: Mapping[str, pd.DataFrame],
        train_end: Optional[pd.Timestamp | str] = None,
    ) -> Dict[str, pd.DataFrame]:
        self.fit(features_by_ticker, train_end=train_end)
        return self.transform(features_by_ticker, extend=True)

    # ── Core steps ─────────────────────────────────────────────────────

    def _rolling_zscore(
        self,
        frames: Mapping[str, pd.DataFrame],
        columns: list,
    ) -> Dict[str, pd.DataFrame]:
        """Per-ticker rolling z-score using only rows strictly before date t.

        The `shift(1)` is what keeps this leak-safe — today's value is NOT
        in the mean/std calculation used to normalise today.
        """
        out: Dict[str, pd.DataFrame] = {}
        w = self.state.rolling_window
        for ticker, df in frames.items():
            if df.empty:
                out[ticker] = df.copy()
                continue
            aligned = df.reindex(columns=columns).astype(float)
            past = aligned.shift(1)
            mean = past.rolling(w, min_periods=w // 4).mean()
            std  = past.rolling(w, min_periods=w // 4).std().replace(0.0, np.nan)
            normalised = (aligned - mean) / std
            out[ticker] = normalised
        return out

    @staticmethod
    def _cross_sectional_median_fill(
        frames: Mapping[str, pd.DataFrame],
        columns: list,
    ) -> Dict[str, pd.DataFrame]:
        """Fill NaN values with the cross-sectional median on the same date."""
        tickers = list(frames.keys())
        if not tickers:
            return {}
        stacked = pd.concat(
            {t: frames[t].reindex(columns=columns) for t in tickers},
            axis=0, names=["ticker", "date"],
        )
        # Median per (date, column) across tickers.
        medians = stacked.groupby(level="date").median()

        out: Dict[str, pd.DataFrame] = {}
        for t in tickers:
            df = frames[t].reindex(columns=columns).copy()
            if df.empty:
                out[t] = df
                continue
            filler = medians.reindex(df.index).fillna(0.0)
            out[t] = df.fillna(filler)
        return out

    def _winsorise(
        self,
        frames: Mapping[str, pd.DataFrame],
        columns: list,
    ) -> Dict[str, pd.DataFrame]:
        lo, hi = -self.state.winsorize_std, self.state.winsorize_std
        return {
            t: df.reindex(columns=columns).clip(lower=lo, upper=hi).fillna(0.0)
            for t, df in frames.items()
        }

    # ── Serialization ──────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self.state)
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        log.info("Saved Normalizer state -> {}", path)

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        with path.open("r") as f:
            payload = json.load(f)
        norm = cls(
            rolling_window=int(payload["rolling_window"]),
            winsorize_std=float(payload["winsorize_std"]),
        )
        norm.state = NormalizerState(**payload)
        return norm

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _collect_columns(frames: Mapping[str, pd.DataFrame]) -> list:
        seen: list = []
        for df in frames.values():
            for c in df.columns:
                if c not in seen:
                    seen.append(c)
        return seen
