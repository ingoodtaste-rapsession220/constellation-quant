"""Phase 6 tests: forward-testing pipeline + live IC tracker."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import pytest

from constellation_quant.forward_testing import (
    ForwardTestConfig,
    ForwardTestPipeline,
    LiveICTracker,
    PredictionRecord,
    PredictionsLog,
    ResultRecord,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def price_frames() -> Dict[str, pd.DataFrame]:
    """6 tickers × 120 trading days — a small but realistic-scale universe."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2024-01-02", periods=120)
    out: Dict[str, pd.DataFrame] = {}
    for i in range(6):
        r = rng.normal(0.0003, 0.015, size=len(dates))
        close = 100 * np.exp(r.cumsum())
        out[f"T{i:02d}"] = pd.DataFrame({
            "date":      dates,
            "open":      close, "high": close, "low": close,
            "close":     close, "adj_close": close, "volume": 1_000_000.0,
            "dividends": 0.0, "stock_splits": 0.0,
        })
    return out


# ── PredictionsLog ─────────────────────────────────────────────────────────


def test_predictions_log_append_and_iter(tmp_path):
    log = PredictionsLog(tmp_path)
    rec = PredictionRecord(
        date=pd.Timestamp("2024-06-03"),
        tickers=["AAPL", "MSFT"],
        scores=np.array([0.5, -0.2]),
        horizon=5,
    )
    assert log.append(rec) is True
    # Idempotent — re-appending the same date returns False and is a no-op.
    assert log.append(rec) is False

    rows = list(log.iter_predictions())
    assert len(rows) == 1
    assert rows[0].date == pd.Timestamp("2024-06-03")
    assert rows[0].tickers == ["AAPL", "MSFT"]
    assert np.allclose(rows[0].scores, [0.5, -0.2])


def test_predictions_log_multiple_dates(tmp_path):
    log = PredictionsLog(tmp_path)
    for d in ["2024-06-03", "2024-06-04", "2024-06-05"]:
        log.append(PredictionRecord(
            date=pd.Timestamp(d), tickers=["A", "B"],
            scores=np.array([1.0, 2.0]), horizon=5,
        ))
    assert len(list(log.iter_predictions())) == 3


def test_predictions_on_returns_none_for_missing(tmp_path):
    log = PredictionsLog(tmp_path)
    assert log.predictions_on(pd.Timestamp("2024-06-03")) is None


def test_results_log_write_and_dedupe(tmp_path):
    log = PredictionsLog(tmp_path)
    r = ResultRecord(
        date=pd.Timestamp("2024-06-03"),
        ic=0.05, hit_rate=1.0, spread=0.01, n_valid=50, horizon=5,
    )
    log.append_result(r)
    log.append_result(r)                # same date → dedup, not duplicate
    results = list(log.iter_results())
    assert len(results) == 1
    assert results[0].ic == pytest.approx(0.05)


def test_results_frame_is_indexed_by_date(tmp_path):
    log = PredictionsLog(tmp_path)
    for i, d in enumerate(["2024-06-03", "2024-06-04"]):
        log.append_result(ResultRecord(
            date=pd.Timestamp(d), ic=0.02 + i * 0.01, hit_rate=0.5 + i * 0.1,
            spread=0.01, n_valid=50, horizon=5,
        ))
    df = log.results_frame()
    assert df.index.is_monotonic_increasing
    assert "ic" in df.columns and len(df) == 2


def test_prediction_record_json_roundtrip():
    rec = PredictionRecord(
        date=pd.Timestamp("2024-06-03"),
        tickers=["AAPL", "MSFT"],
        scores=np.array([0.5, -0.2]),
        horizon=5,
    )
    line = rec.to_json_line()
    restored = PredictionRecord.from_json_line(line)
    assert restored.date == rec.date
    assert restored.tickers == rec.tickers
    assert np.allclose(restored.scores, rec.scores)


def test_result_record_nan_roundtrip():
    rec = ResultRecord(
        date=pd.Timestamp("2024-06-03"),
        ic=float("nan"), hit_rate=float("nan"),
        spread=float("nan"), n_valid=0, horizon=5,
    )
    line = rec.to_json_line()
    restored = ResultRecord.from_json_line(line)
    assert np.isnan(restored.ic)
    assert np.isnan(restored.hit_rate)


# ── LiveICTracker back-scoring ─────────────────────────────────────────────


def test_live_ic_tracker_backscoring(tmp_path, price_frames):
    """Predictions correlated with forward returns → positive IC."""
    log = PredictionsLog(tmp_path)
    dates = price_frames["T00"]["date"].reset_index(drop=True)
    # Build wide adj_close directly from the frames, then seed predictions
    # with the exact future returns (perfect-foresight signal).
    adj = pd.concat(
        [f.set_index("date")["adj_close"].rename(t) for t, f in price_frames.items()],
        axis=1,
    )
    horizon = 5
    fwd = np.log(adj.shift(-horizon) / adj)

    # Plant predictions at a handful of dates early in the series.
    pred_dates = dates[20:30]
    tickers = sorted(price_frames)
    for d in pred_dates:
        fwd_row = fwd.loc[d]
        # Perfect scores → fwd returns (with tiny noise so spearman isn't 1.0).
        noise = np.random.default_rng(int(d.dayofyear)).normal(0, 0.0005, size=len(tickers))
        log.append(PredictionRecord(
            date=pd.Timestamp(d),
            tickers=list(tickers),
            scores=fwd_row.reindex(tickers).to_numpy() + noise,
            horizon=horizon,
        ))

    tracker = LiveICTracker(log)
    n = tracker.rescore_all(price_frames, top_n=2)
    assert n == len(pred_dates)

    results_df = log.results_frame()
    assert (results_df["ic"] > 0).all(), f"expected all positive IC, got:\n{results_df}"


def test_live_ic_tracker_skips_future_predictions(tmp_path, price_frames):
    """Predictions too close to the end of the price series can't be scored yet."""
    log = PredictionsLog(tmp_path)
    dates = price_frames["T00"]["date"].reset_index(drop=True)
    latest_date = dates.max()

    log.append(PredictionRecord(
        date=pd.Timestamp(latest_date),
        tickers=sorted(price_frames), scores=np.zeros(6), horizon=5,
    ))
    tracker = LiveICTracker(log)
    assert tracker.rescore_all(price_frames, top_n=2) == 0


def test_live_ic_summary_reports_aggregates(tmp_path):
    log = PredictionsLog(tmp_path)
    for i, d in enumerate(pd.bdate_range("2024-06-03", periods=40)):
        log.append_result(ResultRecord(
            date=pd.Timestamp(d),
            ic=0.03 + i * 0.0005, hit_rate=0.55, spread=0.01,
            n_valid=50, horizon=5,
        ))
    tracker = LiveICTracker(log)
    summary = tracker.summarise()
    assert summary.n_scored == 40
    assert summary.mean_ic_all > 0.03
    assert np.isfinite(summary.mean_ic_30d)


# ── ForwardTestPipeline ────────────────────────────────────────────────────


def test_pipeline_predict_appends_record(tmp_path):
    cfg = ForwardTestConfig(log_dir=tmp_path / "logs", horizon=5, top_n=2)
    pipeline = ForwardTestPipeline(cfg)

    def scorer(pred_date, tickers, frames):
        return np.arange(len(tickers), dtype=np.float64)

    rec = pipeline.predict(
        pred_date=pd.Timestamp("2024-06-03"),
        tickers=["AAPL", "MSFT"],
        feature_frames={},
        scorer=scorer,
    )
    assert rec.tickers == ["AAPL", "MSFT"]
    assert len(list(pipeline.log.iter_predictions())) == 1


def test_pipeline_predict_idempotent(tmp_path):
    cfg = ForwardTestConfig(log_dir=tmp_path / "logs", horizon=5, top_n=2)
    pipeline = ForwardTestPipeline(cfg)

    calls: List[int] = []
    def scorer(pred_date, tickers, frames):
        calls.append(1)
        return np.zeros(len(tickers))

    pipeline.predict(pd.Timestamp("2024-06-03"), ["A", "B"], {}, scorer)
    pipeline.predict(pd.Timestamp("2024-06-03"), ["A", "B"], {}, scorer)
    # Scorer called only for the first call; second is a short-circuit.
    assert len(calls) == 1


def test_pipeline_scorer_shape_mismatch_raises(tmp_path):
    cfg = ForwardTestConfig(log_dir=tmp_path / "logs", horizon=5, top_n=2)
    pipeline = ForwardTestPipeline(cfg)

    def bad_scorer(pred_date, tickers, frames):
        return np.zeros(len(tickers) + 1)   # wrong shape

    with pytest.raises(ValueError, match="expected"):
        pipeline.predict(pd.Timestamp("2024-06-03"), ["A", "B"], {}, bad_scorer)


def test_pipeline_summary_end_to_end(tmp_path, price_frames):
    """Predict a handful of dates, back-score, summarise — every stage runs."""
    cfg = ForwardTestConfig(log_dir=tmp_path / "logs", horizon=5, top_n=2)
    pipeline = ForwardTestPipeline(cfg)

    dates = price_frames["T00"]["date"].reset_index(drop=True)
    adj = pd.concat(
        [f.set_index("date")["adj_close"].rename(t) for t, f in price_frames.items()],
        axis=1,
    )
    fwd = np.log(adj.shift(-5) / adj)

    def scorer(pred_date, tickers, frames):
        row = fwd.loc[pred_date]
        return row.reindex(tickers).to_numpy(dtype=np.float64)

    for d in dates[20:25]:
        pipeline.predict(d, sorted(price_frames), {}, scorer)

    assert pipeline.rescore(price_frames) == 5
    summary = pipeline.summary()
    assert summary.n_scored == 5
    assert summary.mean_ic_all > 0
