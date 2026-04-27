"""Live IC tracker — reconciles yesterday's predictions against realised returns.

Given the prediction log and today's price data, this module:

    1. Finds every prediction date D such that we now have H trading days of
       data past D and haven't yet scored it.
    2. For each such date, computes the realised H-day forward return for
       the predicted tickers.
    3. Writes a `ResultRecord` (IC, hit rate, long-short spread) back into
       the predictions log's `results.jsonl`.
    4. Aggregates rolling IC statistics (mean, IR, trailing 30d / 90d).

All mask-aware; tickers that delisted between prediction and scoring are
excluded rather than dragged in with stale prices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from constellation_quant.evaluation.metrics import (
    hit_rate as _hit_rate_fn,
    long_short_spread,
    spearman_corr,
)
from constellation_quant.forward_testing.predictions_log import (
    PredictionRecord,
    PredictionsLog,
    ResultRecord,
)
from constellation_quant.utils import get_logger

log = get_logger(__name__)


@dataclass
class LiveICSummary:
    """Rolling IC summary computed from the results log."""
    n_scored:        int
    mean_ic_all:     float
    mean_ic_30d:     float
    mean_ic_90d:     float
    ic_ir_all:       float
    hit_rate_all:    float
    first_date:      Optional[pd.Timestamp] = None
    last_date:       Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, float]:
        out = {
            "n_scored":     self.n_scored,
            "mean_ic_all":  self.mean_ic_all,
            "mean_ic_30d":  self.mean_ic_30d,
            "mean_ic_90d":  self.mean_ic_90d,
            "ic_ir_all":    self.ic_ir_all,
            "hit_rate_all": self.hit_rate_all,
        }
        if self.first_date is not None:
            out["first_date"] = self.first_date.date().isoformat()
        if self.last_date is not None:
            out["last_date"] = self.last_date.date().isoformat()
        return out


class LiveICTracker:
    """Reconcile predictions → realised returns → rolling IC stats."""

    def __init__(self, log: PredictionsLog):
        self.log = log

    # ── Back-scoring ───────────────────────────────────────────────────

    def rescore_all(self, price_frames: Mapping[str, pd.DataFrame],
                     top_n: int = 50) -> int:
        """Score every prediction row with enough forward data to be knowable.

        Returns the number of rows freshly scored.
        """
        prices_wide = _wide_adj_close(price_frames)
        existing_results = {r.date for r in self.log.iter_results()}
        written = 0

        for pred in self.log.iter_predictions():
            if pred.date in existing_results:
                continue
            result = self._score_one(pred, prices_wide, top_n=top_n)
            if result is None:
                continue
            self.log.append_result(result)
            written += 1
        if written:
            log.info("Back-scored {} prediction row(s).", written)
        return written

    def _score_one(
        self,
        pred: PredictionRecord,
        prices_wide: pd.DataFrame,
        top_n: int,
    ) -> Optional[ResultRecord]:
        calendar = prices_wide.index
        # Find the trading day that is prediction_date.
        loc = calendar.searchsorted(pred.date)
        if loc == len(calendar) or calendar[loc] != pd.Timestamp(pred.date):
            # Prediction date not in the calendar — wait for data.
            return None
        target_pos = loc + pred.horizon
        if target_pos >= len(calendar):
            # Not enough forward data yet — score next cycle.
            return None

        p0 = prices_wide.iloc[loc]
        pN = prices_wide.iloc[target_pos]

        targets = np.log(pN / p0)
        target_series = targets.reindex(pred.tickers).to_numpy(dtype=np.float64)
        valid = np.isfinite(target_series) & np.isfinite(pred.scores)
        if valid.sum() < 5:
            return None

        ic = spearman_corr(pred.scores[valid], target_series[valid])
        hr = _hit_rate_fn(pred.scores[valid], target_series[valid], top_n=top_n)
        spread = long_short_spread(pred.scores[valid], target_series[valid], top_n=top_n)
        return ResultRecord(
            date=pred.date,
            ic=float(ic),
            hit_rate=float(hr),
            spread=float(spread),
            n_valid=int(valid.sum()),
            horizon=pred.horizon,
        )

    # ── Aggregation ────────────────────────────────────────────────────

    def summarise(self) -> LiveICSummary:
        """Rolling IC stats across the whole results log."""
        df = self.log.results_frame()
        if df.empty:
            return LiveICSummary(
                n_scored=0, mean_ic_all=float("nan"),
                mean_ic_30d=float("nan"), mean_ic_90d=float("nan"),
                ic_ir_all=float("nan"), hit_rate_all=float("nan"),
            )

        valid = df.dropna(subset=["ic"])
        mean_all = float(valid["ic"].mean()) if not valid.empty else float("nan")
        std_all = float(valid["ic"].std(ddof=0)) if not valid.empty else float("nan")
        ir_all = float(mean_all / std_all) if std_all > 1e-12 else float("nan")
        hr_all = float(valid["hit_rate"].mean()) if not valid.empty else float("nan")

        last30 = valid.tail(30)
        last90 = valid.tail(90)

        return LiveICSummary(
            n_scored=len(valid),
            mean_ic_all=mean_all,
            mean_ic_30d=float(last30["ic"].mean()) if not last30.empty else float("nan"),
            mean_ic_90d=float(last90["ic"].mean()) if not last90.empty else float("nan"),
            ic_ir_all=ir_all,
            hit_rate_all=hr_all,
            first_date=valid.index.min() if not valid.empty else None,
            last_date=valid.index.max() if not valid.empty else None,
        )


def _wide_adj_close(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Build wide adj_close frame indexed by date, columns = tickers."""
    cols: List[pd.Series] = []
    for ticker, df in frames.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.set_index("date")
        px = frame["adj_close"].astype(float).sort_index()
        cols.append(px.rename(ticker))
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()
