"""Forward-testing orchestration: ingest → feature → predict → log → back-score.

Production flow for paper-trading the model on live data:

    1. Incrementally download today's OHLCV for every current member.
    2. Run the feature engine over the rolling window.
    3. Score the current universe with a loaded checkpoint.
    4. Write a `PredictionRecord` into the predictions log.
    5. Back-score every old prediction that now has `horizon` trading days
       of forward data — yields new `ResultRecord`s.
    6. Emit a rolling `LiveICSummary` for monitoring.

Each step is wrapped so the CLI (`scripts/forward_test.py`) can run steps
à la carte (`--only predict`, `--only rescore`, etc.). That matches the
realistic scheduling: prediction runs at market close, rescoring runs after
settlement a few days later.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as _date
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from constellation_quant.forward_testing.live_ic_tracker import (
    LiveICSummary,
    LiveICTracker,
)
from constellation_quant.forward_testing.predictions_log import (
    PredictionRecord,
    PredictionsLog,
)
from constellation_quant.utils import get_logger

log = get_logger(__name__)


# Callback type for tests + production to inject an inference function:
#   fn(pred_date, tickers, feature_frames) -> np.ndarray of scores
ScorerFn = Callable[[pd.Timestamp, Sequence[str], Mapping[str, pd.DataFrame]], np.ndarray]


@dataclass
class ForwardTestConfig:
    """Minimal config (the rest flows from the training/model configs)."""
    log_dir:   Path                        # e.g. outputs/forward_test/
    horizon:   int = 5                     # must match model horizon
    top_n:     int = 50                    # used for back-scored hit rate + spread


class ForwardTestPipeline:
    """End-to-end forward-testing orchestrator.

    Callers are expected to construct this once, then invoke:
        .predict(date, tickers, feature_frames, scorer_fn)
        .rescore(price_frames)
        .summary()
    """

    def __init__(self, config: ForwardTestConfig):
        self.cfg = config
        self.log = PredictionsLog(config.log_dir)
        self.tracker = LiveICTracker(self.log)

    # ── Steps ──────────────────────────────────────────────────────────

    def predict(
        self,
        pred_date: pd.Timestamp,
        tickers: Sequence[str],
        feature_frames: Mapping[str, pd.DataFrame],
        scorer: ScorerFn,
    ) -> PredictionRecord:
        """Run the scoring function for `pred_date` and append to the log.

        Idempotent — a second call on the same date is a no-op.
        """
        existing = self.log.predictions_on(pred_date)
        if existing is not None:
            log.info("Prediction already logged for {}; skipping.", pred_date.date())
            return existing

        scores = np.asarray(scorer(pred_date, tickers, feature_frames), dtype=np.float64)
        if scores.shape != (len(tickers),):
            raise ValueError(
                f"scorer returned shape {scores.shape}, expected ({len(tickers)},)"
            )
        record = PredictionRecord(
            date=pd.Timestamp(pred_date).normalize(),
            tickers=list(tickers),
            scores=scores,
            horizon=self.cfg.horizon,
        )
        written = self.log.append(record)
        if written:
            log.info("Logged prediction for {} ({} tickers).",
                     record.date.date(), len(tickers))
        return record

    def rescore(self, price_frames: Mapping[str, pd.DataFrame]) -> int:
        """Back-score every predictable row with enough forward data."""
        return self.tracker.rescore_all(price_frames, top_n=self.cfg.top_n)

    def summary(self) -> LiveICSummary:
        return self.tracker.summarise()

    # ── Diagnostics ────────────────────────────────────────────────────

    def recent_predictions(self, n: int = 5) -> pd.DataFrame:
        """Last `n` predictions (metadata only)."""
        df = self.log.predictions_frame()
        return df.tail(n)

    def recent_results(self, n: int = 30) -> pd.DataFrame:
        df = self.log.results_frame()
        return df.tail(n)
