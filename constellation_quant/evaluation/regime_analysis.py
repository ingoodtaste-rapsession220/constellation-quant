"""Regime-aware evaluation.

Detects four regimes on the S&P 500 benchmark return series:
    - bull       : rolling 60-day return > 0
    - bear       : rolling 60-day return < 0
    - high_vol   : realised 20-day vol > `high_vol_threshold` (annualised)
    - low_vol    : realised 20-day vol < `low_vol_threshold`

Then slices a `BacktestResult` by regime and reports per-regime summary
stats. Flags big divergences so we can see when the model's edge collapses
(e.g. only works in calm bull markets).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from constellation_quant.evaluation.backtester import BacktestResult, TRADING_DAYS_PER_YEAR
from constellation_quant.utils import get_logger

log = get_logger(__name__)


@dataclass
class RegimeStats:
    """Summary per regime."""
    name:           str
    n_days:         int
    annual_return:  float
    annual_vol:     float
    sharpe:         float
    max_drawdown:   float
    hit_rate:       float                    # fraction of days with positive return


@dataclass
class RegimeConfig:
    trend_window:        int   = 60
    vol_window:          int   = 20
    high_vol_threshold:  float = 0.20        # annualised vol > 20% → high vol
    low_vol_threshold:   float = 0.10        # annualised vol < 10% → low vol


# ── Classifier ─────────────────────────────────────────────────────────────


class RegimeClassifier:
    """Assigns a regime label to each trading day in a benchmark return series."""

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.cfg = config or RegimeConfig()

    def classify(self, benchmark_returns: pd.Series) -> pd.DataFrame:
        """Return a DataFrame with columns {trend, vol_regime}.

        Args:
            benchmark_returns: Daily total returns of a broad-market proxy
                (SPY or the portfolio's own equity curve — either works
                for sub-sampling purposes).
        """
        r = benchmark_returns.astype(float).sort_index()
        trend = r.rolling(self.cfg.trend_window, min_periods=self.cfg.trend_window).sum()
        vol = r.rolling(self.cfg.vol_window, min_periods=self.cfg.vol_window).std() \
                * np.sqrt(TRADING_DAYS_PER_YEAR)

        trend_label = pd.Series(index=r.index, dtype=object)
        trend_label[:] = "unknown"
        trend_label[trend > 0] = "bull"
        trend_label[trend < 0] = "bear"

        vol_label = pd.Series(index=r.index, dtype=object)
        vol_label[:] = "unknown"
        vol_label[vol > self.cfg.high_vol_threshold] = "high_vol"
        vol_label[vol < self.cfg.low_vol_threshold]  = "low_vol"

        return pd.DataFrame({"trend": trend_label, "vol_regime": vol_label})


# ── Analyzer ───────────────────────────────────────────────────────────────


class RegimeAnalyzer:
    """Slice a backtest by regime and report per-regime stats."""

    def __init__(self, classifier: Optional[RegimeClassifier] = None):
        self.classifier = classifier or RegimeClassifier()

    def analyze(
        self,
        result: BacktestResult,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, RegimeStats]:
        """Compute per-regime summary stats.

        Args:
            result: A completed `BacktestResult`.
            benchmark_returns: Optional benchmark series. If None, uses the
                strategy's own daily returns as a proxy — fine for
                distinguishing calm vs volatile periods, less so for
                bull/bear (which ideally use SPY).
        """
        series = benchmark_returns if benchmark_returns is not None else result.daily_returns
        labels = self.classifier.classify(series)

        returns = result.daily_returns
        out: Dict[str, RegimeStats] = {}

        for column, label in (
            ("trend",      "bull"),
            ("trend",      "bear"),
            ("vol_regime", "high_vol"),
            ("vol_regime", "low_vol"),
        ):
            aligned = labels[column].reindex(returns.index, fill_value="unknown")
            sel = (aligned == label)
            regime_returns = returns[sel]
            out[label] = self._stats(label, regime_returns)
        out["all"] = self._stats("all", returns)
        return out

    @staticmethod
    def _stats(name: str, returns: pd.Series) -> RegimeStats:
        if len(returns) == 0:
            return RegimeStats(
                name=name, n_days=0,
                annual_return=float("nan"), annual_vol=float("nan"),
                sharpe=float("nan"), max_drawdown=float("nan"),
                hit_rate=float("nan"),
            )
        equity = (1.0 + returns).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        mean_daily = returns.mean()
        vol_daily = returns.std(ddof=0)
        sharpe = (
            float(mean_daily / vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR))
            if vol_daily > 1e-12 else float("nan")
        )
        return RegimeStats(
            name=name,
            n_days=len(returns),
            annual_return=float((1.0 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1.0),
            annual_vol=float(vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR)),
            sharpe=sharpe,
            max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
            hit_rate=float((returns > 0).mean()),
        )


def regime_stats_to_dataframe(stats: Dict[str, RegimeStats]) -> pd.DataFrame:
    """Compact table for reporting."""
    rows = []
    for label, s in stats.items():
        rows.append({
            "regime":        s.name,
            "n_days":        s.n_days,
            "annual_return": s.annual_return,
            "annual_vol":    s.annual_vol,
            "sharpe":        s.sharpe,
            "max_drawdown":  s.max_drawdown,
            "hit_rate":      s.hit_rate,
        })
    return pd.DataFrame(rows)
