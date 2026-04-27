"""Append-only log of daily model predictions + retrospective back-scoring.

The forward-test pipeline writes one JSON line per prediction date:

    {"date": "2025-06-03", "tickers": [...], "scores": [...], "horizon": 5}

A separate "results" file ties predictions to realised forward returns once
they're knowable (i.e. the current date has moved `horizon` trading days
past the prediction date). Keeping the two files separate means the
prediction log stays append-only — you never rewrite historical rows.

    predictions.jsonl   — append-only; one row per prediction date
    results.jsonl       — back-scored rows; computed on a schedule
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PredictionRecord:
    """One row of the predictions log."""
    date:    pd.Timestamp
    tickers: List[str]
    scores:  np.ndarray
    horizon: int

    def to_json_line(self) -> str:
        return json.dumps({
            "date":    self.date.date().isoformat(),
            "tickers": list(self.tickers),
            "scores":  [float(x) for x in self.scores],
            "horizon": int(self.horizon),
        })

    @classmethod
    def from_json_line(cls, line: str) -> "PredictionRecord":
        obj = json.loads(line)
        return cls(
            date=pd.Timestamp(obj["date"]),
            tickers=list(obj["tickers"]),
            scores=np.asarray(obj["scores"], dtype=np.float64),
            horizon=int(obj["horizon"]),
        )


@dataclass
class ResultRecord:
    """A back-scored comparison of one prediction date to realised returns."""
    date:       pd.Timestamp
    ic:         float                        # Spearman
    hit_rate:   float
    spread:     float                        # long-short mean return
    n_valid:    int
    horizon:    int

    def to_json_line(self) -> str:
        return json.dumps({
            "date":     self.date.date().isoformat(),
            "ic":       None if np.isnan(self.ic) else float(self.ic),
            "hit_rate": None if np.isnan(self.hit_rate) else float(self.hit_rate),
            "spread":   None if np.isnan(self.spread) else float(self.spread),
            "n_valid":  int(self.n_valid),
            "horizon":  int(self.horizon),
        })

    @classmethod
    def from_json_line(cls, line: str) -> "ResultRecord":
        obj = json.loads(line)
        return cls(
            date=pd.Timestamp(obj["date"]),
            ic=float(obj["ic"]) if obj["ic"] is not None else float("nan"),
            hit_rate=float(obj["hit_rate"]) if obj["hit_rate"] is not None else float("nan"),
            spread=float(obj["spread"]) if obj["spread"] is not None else float("nan"),
            n_valid=int(obj["n_valid"]),
            horizon=int(obj["horizon"]),
        )


class PredictionsLog:
    """Append-only JSONL log of daily predictions with idempotent writes."""

    PREDICTIONS_NAME = "predictions.jsonl"
    RESULTS_NAME = "results.jsonl"

    def __init__(self, log_dir: Path):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.predictions_path = self.dir / self.PREDICTIONS_NAME
        self.results_path = self.dir / self.RESULTS_NAME

    # ── Predictions (forward) ──────────────────────────────────────────

    def append(self, record: PredictionRecord) -> bool:
        """Append a prediction record unless one already exists for that date.

        Returns True if the record was written, False if it was a duplicate.
        """
        existing = {r.date for r in self.iter_predictions()}
        if record.date in existing:
            return False
        with self.predictions_path.open("a") as f:
            f.write(record.to_json_line() + "\n")
        return True

    def iter_predictions(self) -> Iterator[PredictionRecord]:
        if not self.predictions_path.exists():
            return
        with self.predictions_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield PredictionRecord.from_json_line(line)

    def predictions_on(self, date: pd.Timestamp) -> Optional[PredictionRecord]:
        target = pd.Timestamp(date).normalize()
        for r in self.iter_predictions():
            if r.date == target:
                return r
        return None

    # ── Results (retrospective) ────────────────────────────────────────

    def append_result(self, record: ResultRecord) -> None:
        # Results file is authoritative per date — we dedupe by rewriting when
        # re-scoring an existing date. Rare path; cheap for forward-test sizes.
        existing: Dict[pd.Timestamp, ResultRecord] = {
            r.date: r for r in self.iter_results()
        }
        existing[record.date] = record
        with self.results_path.open("w") as f:
            for d in sorted(existing.keys()):
                f.write(existing[d].to_json_line() + "\n")

    def iter_results(self) -> Iterator[ResultRecord]:
        if not self.results_path.exists():
            return
        with self.results_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield ResultRecord.from_json_line(line)

    def results_frame(self) -> pd.DataFrame:
        rows = [
            {"date": r.date, "ic": r.ic, "hit_rate": r.hit_rate,
             "spread": r.spread, "n_valid": r.n_valid, "horizon": r.horizon}
            for r in self.iter_results()
        ]
        if not rows:
            return pd.DataFrame(columns=["date", "ic", "hit_rate", "spread",
                                          "n_valid", "horizon"])
        return pd.DataFrame(rows).set_index("date").sort_index()

    def predictions_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.iter_predictions():
            rows.append({"date": r.date, "n_tickers": len(r.tickers),
                          "horizon": r.horizon})
        if not rows:
            return pd.DataFrame(columns=["date", "n_tickers", "horizon"])
        return pd.DataFrame(rows).set_index("date").sort_index()
