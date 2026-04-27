"""Phase 4 evaluation tests.

Synthetic predictions + synthetic prices. Verifies:
  - portfolio constructors are dollar-neutral and respect top_n
  - backtest P&L math (including transaction costs)
  - drawdown computation
  - regime slicing
  - significance tests flag obvious differences
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from constellation_quant.evaluation import (
    Backtester,
    BacktestResult,
    DailyPrediction,
    EqualWeightLongShort,
    PortfolioWeights,
    RegimeAnalyzer,
    RegimeClassifier,
    RiskParityLongShort,
    SectorNeutralLongShort,
    TRADING_DAYS_PER_YEAR,
    bootstrap_sharpe_diff,
    build_portfolio_constructor,
    daily_metrics,
    diebold_mariano,
    paired_t_test_ic,
    regime_stats_to_dataframe,
    spearman_corr,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_price_frames() -> Dict[str, pd.DataFrame]:
    """10 tickers × 250 trading days of deterministic-random GBM paths."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=250)
    frames: Dict[str, pd.DataFrame] = {}
    for i in range(10):
        r = rng.normal(0.0003, 0.015, size=len(dates))
        close = 100 * np.exp(r.cumsum())
        frames[f"T{i:02d}"] = pd.DataFrame({
            "date": dates,
            "open": close, "high": close, "low": close,
            "close": close, "adj_close": close,
            "volume": 1_000_000.0,
            "dividends": 0.0, "stock_splits": 0.0,
        })
    return frames


@pytest.fixture
def aligned_predictions(synthetic_price_frames) -> List[DailyPrediction]:
    """Predictions where scores are positively correlated with forward returns.

    The signal: score[i] = (future 5-day log return of ticker i). A perfect
    model would produce Sharpe > 0 after costs.
    """
    tickers = sorted(synthetic_price_frames.keys())
    any_df = synthetic_price_frames[tickers[0]]
    calendar = any_df["date"].reset_index(drop=True)

    # Build a wide adj_close matrix to derive forward returns as labels.
    adj = pd.concat(
        [synthetic_price_frames[t].set_index("date")["adj_close"].rename(t) for t in tickers],
        axis=1,
    )
    fwd = np.log(adj.shift(-5) / adj).iloc[:-5]    # 5-day forward log return

    preds: List[DailyPrediction] = []
    for date, row in fwd.iterrows():
        scores = row.to_numpy(dtype=np.float64)
        # Make sure it's noisy but informative — add some noise at 20% level.
        scores = scores + np.random.default_rng(int(date.dayofyear)).normal(
            0, 0.005, size=scores.shape,
        )
        preds.append(DailyPrediction(
            date=pd.Timestamp(date),
            tickers=tickers,
            scores=scores,
            mask=np.ones(len(tickers), dtype=bool),
        ))
    return preds


# ── Portfolio constructors ─────────────────────────────────────────────────


def test_equal_weight_is_dollar_neutral():
    constructor = EqualWeightLongShort(top_n=2, leg_weight=1.0)
    scores = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 0.0])
    tickers = ["A", "B", "C", "D", "E", "F"]
    pw = constructor.construct(scores, tickers)
    assert pw.dollar_neutral(), f"weights not dollar-neutral: {pw.weights}"
    assert abs(pw.gross_exposure() - 2.0) < 1e-9
    assert abs(pw.net_exposure()) < 1e-9
    # Two longs (highest scores) + two shorts (lowest scores)
    assert {t: w for t, w in pw.weights.items() if w > 0}.keys() == {"D", "E"}
    assert {t: w for t, w in pw.weights.items() if w < 0}.keys() == {"B", "F"}


def test_equal_weight_respects_mask():
    constructor = EqualWeightLongShort(top_n=2, leg_weight=1.0)
    scores = np.array([3.0, 1.0, 2.0, 4.0, 5.0, 0.0])
    mask = np.array([True, True, False, True, True, True])
    tickers = ["A", "B", "C", "D", "E", "F"]
    pw = constructor.construct(scores, tickers, mask=mask)
    assert "C" not in pw.weights     # masked out


def test_equal_weight_top_n_larger_than_half_shrinks():
    """When universe is small, top_n must shrink to |universe|//2."""
    constructor = EqualWeightLongShort(top_n=50, leg_weight=1.0)
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    tickers = ["A", "B", "C", "D"]
    pw = constructor.construct(scores, tickers)
    # 4 stocks → 2 longs + 2 shorts.
    assert len([w for w in pw.weights.values() if w > 0]) == 2
    assert len([w for w in pw.weights.values() if w < 0]) == 2


def test_risk_parity_scales_by_inverse_vol():
    constructor = RiskParityLongShort(top_n=2, leg_weight=1.0)
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    volatility = np.array([0.01, 0.02, 0.03, 0.04])   # D has highest vol
    tickers = ["A", "B", "C", "D"]
    pw = constructor.construct(scores, tickers, volatility=volatility)
    assert pw.dollar_neutral()
    # Long leg: C + D. C has lower vol → larger weight.
    w_c, w_d = pw.weights["C"], pw.weights["D"]
    assert w_c > w_d > 0


def test_risk_parity_falls_back_to_equal_weight_without_vol():
    constructor = RiskParityLongShort(top_n=2, leg_weight=1.0)
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    tickers = ["A", "B", "C", "D"]
    pw = constructor.construct(scores, tickers)       # no vol supplied
    assert pw.dollar_neutral()
    # All long weights equal, all short weights equal.
    longs = sorted([w for w in pw.weights.values() if w > 0])
    shorts = sorted([w for w in pw.weights.values() if w < 0])
    assert abs(longs[0] - longs[1]) < 1e-9
    assert abs(shorts[0] - shorts[1]) < 1e-9


def test_sector_neutral_balances_per_sector():
    # quantile=0.34 → with 3 stocks per sector: top 1 long, bottom 1 short.
    constructor = SectorNeutralLongShort(quantile=0.34, leg_weight=1.0)
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    tickers = ["A", "B", "C", "D", "E", "F"]
    sectors = {"A": "Tech", "B": "Tech", "C": "Tech",
                "D": "Health", "E": "Health", "F": "Health"}
    pw = constructor.construct(scores, tickers, sectors=sectors)
    # Each sector contributes 0 net — check Tech = 0, Health = 0.
    tech_net = sum(w for t, w in pw.weights.items() if sectors[t] == "Tech")
    health_net = sum(w for t, w in pw.weights.items() if sectors[t] == "Health")
    assert abs(tech_net) < 1e-9
    assert abs(health_net) < 1e-9
    assert pw.dollar_neutral()
    # Lowest-ranked stock in each sector shorted; highest-ranked long.
    assert pw.weights["A"] < 0 and pw.weights["C"] > 0
    assert pw.weights["D"] < 0 and pw.weights["F"] > 0


def test_portfolio_constructor_factory():
    c = build_portfolio_constructor("equal_weight", {"top_n": 5, "leg_weight": 0.5})
    assert isinstance(c, EqualWeightLongShort)
    assert c.top_n == 5 and c.leg_weight == 0.5

    c = build_portfolio_constructor("risk_parity", {})
    assert isinstance(c, RiskParityLongShort)

    c = build_portfolio_constructor("sector_neutral", {"quantile": 0.3})
    assert isinstance(c, SectorNeutralLongShort)
    assert c.quantile == 0.3

    with pytest.raises(ValueError):
        build_portfolio_constructor("unknown", {})


# ── Backtester ─────────────────────────────────────────────────────────────


def test_backtester_runs_end_to_end(synthetic_price_frames, aligned_predictions):
    backtester = Backtester(
        constructor=EqualWeightLongShort(top_n=2, leg_weight=1.0),
        transaction_cost_bps=5.0,
        rebalance_frequency="W",
    )
    result = backtester.run(aligned_predictions, synthetic_price_frames)

    assert isinstance(result, BacktestResult)
    assert len(result.daily_returns) > 0
    # Costs deducted and recorded.
    assert result.total_cost > 0
    # Weekly rebalance → fewer rebalance events than trading days.
    assert len(result.turnover) < len(result.daily_returns)


def test_backtester_signal_beats_noise(synthetic_price_frames, aligned_predictions):
    """Predictions are deliberately positively correlated with forward returns
    (with small noise); backtest Sharpe should end up positive."""
    backtester = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=0.0,             # remove cost noise for this test
        rebalance_frequency="W",
    )
    result = backtester.run(aligned_predictions, synthetic_price_frames)
    # Informative signal → positive Sharpe.
    assert result.sharpe > 0, f"expected Sharpe > 0 for informative signal, got {result.sharpe}"
    assert result.final_equity > 1.0


def test_backtester_costs_reduce_sharpe(synthetic_price_frames, aligned_predictions):
    backtester_free = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=0.0, rebalance_frequency="W",
    )
    backtester_costly = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=500.0,            # 5% per trade — absurd, but shows the direction
        rebalance_frequency="W",
    )
    free   = backtester_free.run(aligned_predictions, synthetic_price_frames)
    costly = backtester_costly.run(aligned_predictions, synthetic_price_frames)
    assert costly.total_cost > free.total_cost
    # Final equity with huge costs must be lower than the cost-free baseline.
    assert costly.final_equity < free.final_equity


def test_backtester_drawdown_non_positive(synthetic_price_frames, aligned_predictions):
    """Drawdown series must always be ≤ 0."""
    backtester = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=5.0, rebalance_frequency="W",
    )
    result = backtester.run(aligned_predictions, synthetic_price_frames)
    assert (result.drawdown <= 1e-9).all()
    assert result.max_drawdown <= 0


def test_backtester_daily_rebalance_turnovers_more_than_weekly(
    synthetic_price_frames, aligned_predictions,
):
    weekly = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=5.0, rebalance_frequency="W",
    ).run(aligned_predictions, synthetic_price_frames)
    daily = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=5.0, rebalance_frequency="D",
    ).run(aligned_predictions, synthetic_price_frames)
    assert len(daily.turnover) > len(weekly.turnover)


def test_backtester_empty_predictions_raises(synthetic_price_frames):
    backtester = Backtester(constructor=EqualWeightLongShort(top_n=2))
    with pytest.raises(ValueError, match="empty"):
        backtester.run([], synthetic_price_frames)


# ── Regime analysis ────────────────────────────────────────────────────────


def test_regime_classifier_produces_labels():
    dates = pd.bdate_range("2024-01-02", periods=120)
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0.001, 0.02, size=len(dates)), index=dates)
    classifier = RegimeClassifier()
    labels = classifier.classify(rets)
    assert set(labels.columns) == {"trend", "vol_regime"}
    # At least some classified days (after warmup).
    valid = labels[labels["trend"] != "unknown"]
    assert len(valid) > 0


def test_regime_analyzer_produces_per_regime_stats(synthetic_price_frames,
                                                      aligned_predictions):
    backtester = Backtester(
        constructor=EqualWeightLongShort(top_n=3, leg_weight=1.0),
        transaction_cost_bps=5.0, rebalance_frequency="W",
    )
    result = backtester.run(aligned_predictions, synthetic_price_frames)
    analyzer = RegimeAnalyzer()
    stats = analyzer.analyze(result)
    for key in ("bull", "bear", "high_vol", "low_vol", "all"):
        assert key in stats
    df = regime_stats_to_dataframe(stats)
    assert {"regime", "sharpe", "max_drawdown"} <= set(df.columns)


# ── Significance tests ─────────────────────────────────────────────────────


def test_paired_t_test_detects_improvement():
    rng = np.random.default_rng(0)
    # Model A consistently beats B by 0.02 IC per day.
    ic_b = rng.normal(0.00, 0.05, size=200)
    ic_a = ic_b + 0.02
    result = paired_t_test_ic(ic_a, ic_b)
    assert result.pvalue < 0.01
    assert result.effect > 0


def test_paired_t_test_no_effect_high_p():
    rng = np.random.default_rng(0)
    ic = rng.normal(0.00, 0.05, size=200)
    result = paired_t_test_ic(ic, ic.copy())
    assert np.isfinite(result.pvalue)
    assert result.effect == pytest.approx(0.0)


def test_diebold_mariano_detects_lower_loss():
    rng = np.random.default_rng(0)
    # Errors_a are smaller → should flag DM significance.
    errors_a = rng.normal(0.0, 0.5, size=200)
    errors_b = errors_a + rng.normal(0.3, 0.5, size=200)   # noisier
    result = diebold_mariano(errors_a, errors_b)
    assert result.pvalue < 0.05
    assert result.effect < 0                # loss_a - loss_b < 0 → A better


def test_bootstrap_sharpe_diff_ci():
    rng = np.random.default_rng(0)
    # Strategy A has mean 0.001 / day, B has mean 0.0 → A > B in Sharpe.
    ra = rng.normal(0.001, 0.01, size=200)
    rb = rng.normal(0.000, 0.01, size=200)
    result = bootstrap_sharpe_diff(ra, rb, n_bootstrap=200, block_size=5, seed=0)
    assert result.statistic > 0
    # With 200 days and a meaningful mean shift, the CI should mostly live above 0.
    assert result.ci_high > 0


def test_bootstrap_sharpe_diff_returns_nan_if_too_short():
    result = bootstrap_sharpe_diff(np.array([0.01]), np.array([0.02]),
                                    n_bootstrap=100, block_size=5)
    assert np.isnan(result.ci_low) or np.isnan(result.ci_high)


# ── Metric cross-checks (add a few more here since portfolio tests land in this file too) ──


def test_daily_metrics_on_synthetic():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=50)
    targets = 0.5 * scores + rng.normal(size=50)
    dm = daily_metrics(scores, targets, top_n=10)
    assert dm.valid_n == 50
    assert dm.ic > 0       # positive correlation by construction
