"""Phase 2 feature tests. Pure-numeric — no network, no torch required."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from constellation_quant.features import (
    FeatureComputeRequest,
    FeatureEngine,
    FundamentalFeatures,
    Normalizer,
    SentimentFeatures,
    TechnicalFeatures,
    cross_sectional_zscore,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def price_frame_factory():
    """Produce reproducible synthetic OHLCV frames."""
    def _make(n_days: int = 250, seed: int = 0, start_price: float = 100.0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-02", periods=n_days)
        returns = rng.normal(0.0005, 0.015, size=n_days)
        close = start_price * np.exp(returns.cumsum())
        return pd.DataFrame({
            "date":         dates,
            "open":         close * (1 + rng.normal(0, 0.005, size=n_days)),
            "high":         close * (1 + rng.uniform(0, 0.01, size=n_days)),
            "low":          close * (1 - rng.uniform(0, 0.01, size=n_days)),
            "close":        close,
            "adj_close":    close,
            "volume":       rng.integers(1_000_000, 10_000_000, size=n_days),
            "dividends":    0.0,
            "stock_splits": 0.0,
        })
    return _make


@pytest.fixture
def price_frames_universe(price_frame_factory) -> Dict[str, pd.DataFrame]:
    return {
        f"T{i:03d}": price_frame_factory(seed=i)
        for i in range(5)
    }


# ── Technical features ─────────────────────────────────────────────────────


def test_technical_feature_names_stable():
    tf = TechnicalFeatures()
    names = tf.feature_names()
    assert "ret_1d" in names
    assert "rsi_14" in names
    assert "macd" in names and "macd_signal" in names and "macd_hist" in names
    # Stable ordering across two calls.
    assert tf.feature_names() == names


def test_technical_log_returns_correctness(price_frame_factory):
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=30, seed=0)
    feats = tf.compute_one(df)
    px = df.set_index("date")["adj_close"]
    expected_r1 = np.log(px / px.shift(1))
    # Last value should match exactly (within floating-point).
    pd.testing.assert_series_equal(
        feats["ret_1d"].dropna(),
        expected_r1.dropna().rename("ret_1d"),
        rtol=1e-10,
        atol=0,
    )


def test_technical_rsi_bounded(price_frame_factory):
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=100)
    feats = tf.compute_one(df)
    rsi = feats["rsi_14"].dropna()
    assert ((rsi >= 0) & (rsi <= 100)).all(), "RSI must lie in [0, 100]"


def test_technical_bollinger_width_positive(price_frame_factory):
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=100)
    feats = tf.compute_one(df)
    bbw = feats["bbw_20"].dropna()
    assert (bbw >= 0).all(), "Bollinger width must be >= 0"


def test_technical_no_future_leakage(price_frame_factory):
    """Modifying a future row must NOT change feature values at earlier dates."""
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=60)
    orig = tf.compute_one(df.copy())

    # Corrupt the last row; recompute.
    df2 = df.copy()
    df2.loc[len(df2) - 1, "adj_close"] *= 2
    mod = tf.compute_one(df2)

    # Every row EXCEPT the last should be identical.
    compared = (orig.iloc[:-1].fillna(0) == mod.iloc[:-1].fillna(0)).all().all()
    assert compared, "technical features leak future information"


def test_technical_missing_column_raises(price_frame_factory):
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=30).drop(columns=["volume"])
    with pytest.raises(KeyError, match="volume"):
        tf.compute_one(df)


# ── Cross-sectional z-score ────────────────────────────────────────────────


def test_cross_sectional_zscore_per_date():
    idx = pd.to_datetime(["2020-01-02", "2020-01-03"])
    frames = {
        "A": pd.DataFrame({"pe": [10.0, 20.0]}, index=idx),
        "B": pd.DataFrame({"pe": [20.0, 10.0]}, index=idx),
        "C": pd.DataFrame({"pe": [30.0, 40.0]}, index=idx),
    }
    out = cross_sectional_zscore(frames)
    # On each date, the three z-scored values must sum to ~0 (mean-centred).
    for d in idx:
        vals = [out[t].loc[d, "pe"] for t in ["A", "B", "C"]]
        assert abs(sum(vals)) < 1e-9


# ── Fundamental features ───────────────────────────────────────────────────


def test_fundamental_derives_ratios(price_frame_factory):
    # Quarterly reports aligned with the first year of the price series
    # (the price factory starts on 2020-01-02).
    dates = pd.to_datetime(["2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30"])
    long_fund = pd.DataFrame({
        "date":   list(dates) * 5,
        "metric": sum(([m] * 4 for m in [
            "total_revenue", "net_income", "total_debt", "stockholders_equity",
            "shares_outstanding",
        ]), []),
        "value":  [1e9, 1.1e9, 1.2e9, 1.3e9,      # revenue
                   1e8, 1.05e8, 1.1e8, 1.15e8,    # net income
                   5e8, 5e8, 5e8, 5e8,            # total debt
                   2e9, 2.05e9, 2.1e9, 2.15e9,   # equity
                   1e7, 1e7, 1e7, 1e7],           # shares
    })
    price_df = price_frame_factory(n_days=300, start_price=50.0)
    ff = FundamentalFeatures({"report_lag_days": 0})
    out = ff.compute_one(long_fund, price_df)
    assert "pe" in out.columns
    assert "pb" in out.columns
    assert "de" in out.columns
    assert "roe" in out.columns
    # ROE should be positive and sane for this synthetic data (NI / equity ≈ 5%).
    roe_mean = out["roe"].dropna().mean()
    assert 0.03 < roe_mean < 0.08, f"ROE out of expected range: {roe_mean}"


def test_fundamental_report_lag_shifts_values(price_frame_factory):
    # Keep the quarter-end well inside the price window so both lag variants
    # have rows to show.
    dates = pd.to_datetime(["2019-12-31"])
    long_fund = pd.DataFrame({
        "date":   list(dates) * 3,
        "metric": ["total_revenue", "shares_outstanding", "stockholders_equity"],
        "value":  [1e9, 1e7, 2e9],
    })
    price = price_frame_factory(n_days=300)
    ff_0  = FundamentalFeatures({"report_lag_days": 0 })
    ff_45 = FundamentalFeatures({"report_lag_days": 45})

    lag_0  = ff_0.compute_one(long_fund, price).dropna(how="all")
    lag_45 = ff_45.compute_one(long_fund, price).dropna(how="all")
    assert not lag_0.empty and not lag_45.empty

    # First date with data lands ~45 days later with the lag applied.
    first_0  = lag_0.index.min()
    first_45 = lag_45.index.min()
    assert (first_45 - first_0).days >= 30


# ── Sentiment features ─────────────────────────────────────────────────────


def test_sentiment_handles_missing_sources():
    sf = SentimentFeatures()
    empty = pd.DataFrame(columns=["date", "source", "score", "volume"])
    out = sf.compute_one(empty, daily_index=pd.bdate_range("2024-01-01", periods=10))
    assert out["sent_composite"].iloc[0] == 0.0   # missing filled with neutral


def test_sentiment_composite_weighted_average():
    sf = SentimentFeatures({
        "sources": {
            "finviz":     {"weight": 0.5},
            "stocktwits": {"weight": 0.5},
            "reddit":     {"weight": 0.0},
        }
    })
    df = pd.DataFrame({
        "date":   pd.to_datetime(["2024-01-02"] * 2),
        "source": ["finviz", "stocktwits"],
        "score":  [0.4, -0.2],
        "volume": [5, 10],
    })
    out = sf.compute_one(df)
    # weighted mean = (0.5*0.4 + 0.5*-0.2) / (0.5 + 0.5) = 0.1
    assert abs(out["sent_composite"].iloc[0] - 0.1) < 1e-9


# ── Normalizer ─────────────────────────────────────────────────────────────


def test_normalizer_no_leakage(price_frame_factory):
    """Z-score of row t must not depend on row t's own value.

    Proof: corrupt row t+1 in the raw features → z-score at row t must be unchanged.
    """
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=300, seed=1)
    feats = {"A": tf.compute_one(df)}

    norm = Normalizer(rolling_window=50, winsorize_std=10.0)
    norm.fit(feats, train_end=feats["A"].index[-1])
    ref = norm.transform(feats, extend=True)["A"]

    # Corrupt row 150 — the z-score at row 149 must be unchanged.
    feats_mod = {"A": feats["A"].copy()}
    feats_mod["A"].iloc[150] = feats_mod["A"].iloc[150] * 100
    mod = norm.transform(feats_mod, extend=True)["A"]

    idx_149 = ref.index[149]
    assert np.allclose(
        ref.loc[idx_149].fillna(0).values,
        mod.loc[idx_149].fillna(0).values,
        equal_nan=True,
    ), "Row t's z-score changed when t+1 was modified — leakage!"


def test_normalizer_winsorizes(price_frame_factory):
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=200, seed=2)
    feats = {"A": tf.compute_one(df)}
    norm = Normalizer(rolling_window=30, winsorize_std=2.0)
    out = norm.fit_transform(feats)["A"]
    assert (out.abs() <= 2.0).all().all()


def test_normalizer_save_load_roundtrip(tmp_path, price_frame_factory):
    tf = TechnicalFeatures()
    df = price_frame_factory(n_days=150, seed=3)
    feats = {"A": tf.compute_one(df)}
    norm = Normalizer(rolling_window=30)
    norm.fit(feats, train_end=feats["A"].index[-1])

    p = tmp_path / "norm.json"
    norm.save(p)
    restored = Normalizer.load(p)
    assert restored.state.feature_columns == norm.state.feature_columns
    assert restored.state.train_end == norm.state.train_end


def test_normalizer_transform_before_fit_raises():
    norm = Normalizer()
    with pytest.raises(RuntimeError, match="before fit"):
        norm.transform({"A": pd.DataFrame({"x": [1, 2, 3]})})


# ── FeatureEngine end-to-end ───────────────────────────────────────────────


def test_feature_engine_technical_only(price_frames_universe):
    cfg = {
        "technical":   {"enabled": True,  "indicators": {}},
        "fundamental": {"enabled": False},
        "sentiment":   {"enabled": False},
        "graph_derived": {"enabled": False},
        "normalization": {"rolling_zscore_window": 30, "winsorize_std": 3.0},
    }
    engine = FeatureEngine(cfg)
    req = FeatureComputeRequest(price_frames=price_frames_universe)
    out = engine.compute(req, fit=True,
                         train_end=price_frames_universe["T000"]["date"].max())

    assert set(out.keys()) == set(price_frames_universe.keys())
    for ticker, df in out.items():
        assert not df.empty
        # Every value must be finite after normalisation.
        arr = df.to_numpy()
        assert np.isfinite(arr).all()


def test_feature_engine_feature_names(price_frames_universe):
    cfg = {
        "technical": {"enabled": True, "indicators": {}},
        "fundamental": {"enabled": False},
        "sentiment":   {"enabled": False},
        "graph_derived": {"enabled": False},
    }
    engine = FeatureEngine(cfg)
    names = engine.feature_names()
    assert "ret_1d" in names and "rsi_14" in names
    assert all("sent_" not in n for n in names)  # sentiment disabled
