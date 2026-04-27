"""Master orchestrator: combines all feature builders + normaliser.

Reads `feature_config.yaml`, instantiates the builders for each enabled
feature group, runs them in sequence against the cleaned price /
fundamentals / sentiment / graph data, and produces per-ticker daily
feature DataFrames ready for the Dataset + GNN.

Pipeline:

    cleaned price     ──┐
    fundamentals       ─┤
    sentiment          ─┼─►  per-group builders  ──►  per-ticker feature
    graph / returns    ─┘    (technical, fund.,        DataFrames (raw)
                             sent., graph-derived)         │
                                                            ▼
                                                      Normalizer
                                                      (rolling-z + fill +
                                                       winsorise)
                                                            │
                                                            ▼
                                                   per-ticker normalised
                                                   DataFrames, daily index

The engine's `.compute(...)` returns the same Dict[ticker, DataFrame] shape
the Dataset consumes via its `feature_engine` callable (Phase 1). A single
engine instance can be fit once on the train split and transform val/test
without refitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from constellation_quant.features.fundamental import FundamentalFeatures
from constellation_quant.features.graph_features import GraphFeatures
from constellation_quant.features.normalizer import Normalizer
from constellation_quant.features.sentiment import SentimentFeatures
from constellation_quant.features.technical import TechnicalFeatures
from constellation_quant.utils import get_logger

log = get_logger(__name__)


@dataclass
class FeatureComputeRequest:
    """Bundle of input frames for a feature computation."""
    price_frames: Dict[str, pd.DataFrame]
    fundamentals_frames: Dict[str, pd.DataFrame] = field(default_factory=dict)
    sentiment_frames:    Dict[str, pd.DataFrame] = field(default_factory=dict)
    sector_map:          Dict[str, str]          = field(default_factory=dict)
    returns_wide:        Optional[pd.DataFrame]  = None
    universe_fn:         Optional[Callable[[pd.Timestamp], List[str]]] = None
    graph_dates:         Optional[List[pd.Timestamp]] = None


class FeatureEngine:
    """Orchestrates all feature groups + normalisation into one call.

    Args:
        feature_cfg: Parsed `feature_config.yaml`.
        normalizer:  Optional pre-fitted normaliser. If None, a fresh one is
                     constructed from `feature_cfg.normalization` and will
                     be fit on the first `compute(..., fit=True)` call.
    """

    def __init__(
        self,
        feature_cfg: Mapping[str, Any],
        normalizer: Optional[Normalizer] = None,
    ):
        self.cfg = dict(feature_cfg)
        self._setup_builders()

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            norm_cfg = self.cfg.get("normalization", {}) or {}
            self.normalizer = Normalizer(
                rolling_window=int(norm_cfg.get("rolling_zscore_window", 252)),
                winsorize_std=float(norm_cfg.get("winsorize_std", 3.0)),
            )

    # ── Public API ─────────────────────────────────────────────────────

    def compute(
        self,
        request: FeatureComputeRequest,
        fit: bool = False,
        train_end: Optional[pd.Timestamp | str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute + (optionally fit) + transform the full feature matrix.

        Args:
            request: Input data bundle.
            fit: If True, calls `Normalizer.fit()` on the combined raw
                features (using `train_end` to truncate) before transforming.
            train_end: End of training period — required when `fit=True`.
        """
        groups: Dict[str, Dict[str, pd.DataFrame]] = {}

        if self._technical:
            groups["technical"] = self._technical.compute(request.price_frames)

        if self._fundamental and request.fundamentals_frames:
            daily_idx = self._infer_daily_index(request.price_frames)
            groups["fundamental"] = self._fundamental.compute(
                request.fundamentals_frames,
                request.price_frames,
                daily_index=daily_idx,
                sector_map=request.sector_map,
            )

        if self._sentiment and request.sentiment_frames:
            daily_idx = self._infer_daily_index(request.price_frames)
            groups["sentiment"] = self._sentiment.compute(
                request.sentiment_frames,
                daily_index=daily_idx,
            )

        if self._graph_features and request.returns_wide is not None:
            if request.universe_fn is None or request.graph_dates is None:
                log.warning(
                    "Graph features enabled but universe_fn / graph_dates missing — skipping."
                )
            else:
                groups["graph"] = self._graph_features.compute(
                    request.returns_wide,
                    request.graph_dates,
                    request.universe_fn,
                    sector_map=request.sector_map,
                )

        raw_per_ticker = self._merge_groups(groups)

        if fit:
            self.normalizer.fit(raw_per_ticker, train_end=train_end)
        return self.normalizer.transform(raw_per_ticker, extend=True)

    def save_normalizer(self, path: Path) -> None:
        self.normalizer.save(path)

    def load_normalizer(self, path: Path) -> None:
        self.normalizer = Normalizer.load(path)

    def feature_names(self) -> List[str]:
        """Stable ordered list of all feature columns the engine produces."""
        names: List[str] = []
        if self._technical:
            names += self._technical.feature_names()
        if self._fundamental:
            names += ["pe", "pb", "de", "roe", "fcf_yield", "div_yield",
                      "log_market_cap", "rev_growth_yoy"]
        if self._sentiment:
            names += ["sent_composite", "sent_momentum_5d",
                      "sent_divergence", "sent_mention_spike"]
        if self._graph_features:
            names += ["degree", "avg_neighbor_return",
                      "sector_momentum", "betweenness_centrality"]
        return names

    # ── Setup ──────────────────────────────────────────────────────────

    def _setup_builders(self) -> None:
        tech_cfg = self.cfg.get("technical", {}) or {}
        self._technical = TechnicalFeatures(tech_cfg) if tech_cfg.get("enabled", True) else None

        fund_cfg = self.cfg.get("fundamental", {}) or {}
        self._fundamental = FundamentalFeatures(fund_cfg) if fund_cfg.get("enabled", True) else None

        sent_cfg = self.cfg.get("sentiment", {}) or {}
        self._sentiment = SentimentFeatures(sent_cfg) if sent_cfg.get("enabled", False) else None

        graph_cfg = self.cfg.get("graph_derived", {}) or {}
        self._graph_features = GraphFeatures() if graph_cfg.get("enabled", False) else None

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _infer_daily_index(price_frames: Mapping[str, pd.DataFrame]) -> Optional[pd.DatetimeIndex]:
        all_dates: List = []
        for df in price_frames.values():
            if df is None or df.empty:
                continue
            dates = pd.to_datetime(
                df["date"] if "date" in df.columns else df.index
            ).normalize()
            all_dates.extend(dates.tolist())
        if not all_dates:
            return None
        return pd.DatetimeIndex(sorted(set(all_dates)))

    @staticmethod
    def _merge_groups(
        groups: Mapping[str, Mapping[str, pd.DataFrame]],
    ) -> Dict[str, pd.DataFrame]:
        """Outer-join each group's per-ticker frames on the date index."""
        tickers: set = set()
        for per_ticker in groups.values():
            tickers.update(per_ticker.keys())

        out: Dict[str, pd.DataFrame] = {}
        for ticker in sorted(tickers):
            pieces: List[pd.DataFrame] = []
            for group_name, per_ticker in groups.items():
                df = per_ticker.get(ticker)
                if df is None or df.empty:
                    continue
                pieces.append(df)
            if pieces:
                merged = pd.concat(pieces, axis=1)
                # De-duplicate columns in case of a naming clash across groups.
                merged = merged.loc[:, ~merged.columns.duplicated()]
                out[ticker] = merged.sort_index()
        return out


def build_feature_engine_from_config(feature_cfg: Mapping[str, Any]) -> FeatureEngine:
    """Convenience factory: builds a FeatureEngine from the parsed config dict."""
    return FeatureEngine(feature_cfg)
