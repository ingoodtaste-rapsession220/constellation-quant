"""Evaluation: metrics, backtester, portfolio construction, regime analysis, significance tests."""

from constellation_quant.evaluation.backtester import (
    Backtester,
    BacktestResult,
    DailyPrediction,
    TRADING_DAYS_PER_YEAR,
)
from constellation_quant.evaluation.metrics import (
    AggregateMetrics,
    DailyMetrics,
    aggregate_metrics,
    daily_metrics,
    hit_rate,
    long_short_spread,
    spearman_corr,
)
from constellation_quant.evaluation.portfolio import (
    EqualWeightLongShort,
    PortfolioConstructor,
    PortfolioWeights,
    RiskParityLongShort,
    SectorNeutralLongShort,
    build_portfolio_constructor,
)
from constellation_quant.evaluation.regime_analysis import (
    RegimeAnalyzer,
    RegimeClassifier,
    RegimeConfig,
    RegimeStats,
    regime_stats_to_dataframe,
)
from constellation_quant.evaluation.significance import (
    BootstrapResult,
    SignificanceResult,
    bootstrap_sharpe_diff,
    diebold_mariano,
    paired_t_test_ic,
)

__all__ = [
    # Metrics
    "DailyMetrics", "AggregateMetrics",
    "daily_metrics", "aggregate_metrics",
    "spearman_corr", "hit_rate", "long_short_spread",
    # Backtester
    "DailyPrediction", "BacktestResult", "Backtester",
    "TRADING_DAYS_PER_YEAR",
    # Portfolio
    "PortfolioWeights", "PortfolioConstructor",
    "EqualWeightLongShort", "RiskParityLongShort", "SectorNeutralLongShort",
    "build_portfolio_constructor",
    # Regime
    "RegimeConfig", "RegimeClassifier", "RegimeAnalyzer", "RegimeStats",
    "regime_stats_to_dataframe",
    # Significance
    "SignificanceResult", "BootstrapResult",
    "paired_t_test_ic", "diebold_mariano", "bootstrap_sharpe_diff",
]
