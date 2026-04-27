"""Statistical evaluation metrics.

All functions operate on numpy / torch inputs and are mask-aware. The
primary metric is the Information Coefficient (Spearman rank correlation
between predicted scores and actual forward returns) — consistent IC > 0.03
is the bar for "genuinely useful" in cross-sectional equity prediction.

Every public function returns a plain Python scalar so the trainer can log
them directly to wandb / stdout without array-vs-tensor juggling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class DailyMetrics:
    """Per-date evaluation summary."""
    date: object
    ic:  float
    hit_rate: float
    long_short_spread: float
    valid_n: int


@dataclass
class AggregateMetrics:
    """IC aggregate over a full evaluation period."""
    mean_ic:            float
    std_ic:             float
    ic_ir:              float          # mean / std
    hit_rate:           float
    long_short_spread:  float
    n_days:             int
    per_day:            List[DailyMetrics] = field(default_factory=list)

    def to_dict(self) -> Dict[str, float]:
        return {
            "ic_mean":            self.mean_ic,
            "ic_std":             self.std_ic,
            "ic_ir":              self.ic_ir,
            "hit_rate":           self.hit_rate,
            "long_short_spread":  self.long_short_spread,
            "n_days":             float(self.n_days),
        }


# ── Primitive computations ─────────────────────────────────────────────────


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation. Returns nan if either rankings are constant."""
    n = len(pred)
    if n < 2:
        return float("nan")
    # Rankings break ties by averaging — matches scipy.stats.spearmanr default.
    pred_rank = _rankdata_avg(pred)
    target_rank = _rankdata_avg(target)
    if np.allclose(pred_rank, pred_rank[0]) or np.allclose(target_rank, target_rank[0]):
        return float("nan")
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def _rankdata_avg(a: np.ndarray) -> np.ndarray:
    """Average-rank ties — same behaviour as scipy.stats.rankdata(method='average')."""
    order = a.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1)
    # Average ties.
    unique_vals, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        tie_sums = np.zeros_like(unique_vals, dtype=np.float64)
        np.add.at(tie_sums, inverse, ranks)
        avg_ranks = tie_sums / counts
        ranks = avg_ranks[inverse]
    return ranks


def hit_rate(pred: np.ndarray, target: np.ndarray, top_n: int = 50) -> float:
    """Fraction of days (1 here — per-day metric) on which the predicted top N
    outperformed the predicted bottom N on average."""
    n = len(pred)
    if n < 2 * top_n:
        top_n = max(n // 4, 1)
    order = pred.argsort()
    bottom = target[order[:top_n]].mean()
    top    = target[order[-top_n:]].mean()
    return float(top > bottom)


def long_short_spread(pred: np.ndarray, target: np.ndarray, top_n: int = 50) -> float:
    """Mean return of predicted top N minus mean of predicted bottom N."""
    n = len(pred)
    if n < 2 * top_n:
        top_n = max(n // 4, 1)
    order = pred.argsort()
    bottom = target[order[:top_n]].mean()
    top    = target[order[-top_n:]].mean()
    return float(top - bottom)


# ── Mask-aware daily metric ────────────────────────────────────────────────


def daily_metrics(
    scores: np.ndarray,
    targets: np.ndarray,
    mask: Optional[np.ndarray] = None,
    top_n: int = 50,
    date: object = None,
) -> DailyMetrics:
    scores = _to_numpy(scores)
    targets = _to_numpy(targets)
    if mask is not None:
        mask = _to_numpy(mask).astype(bool)
        scores = scores[mask]
        targets = targets[mask]

    finite = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[finite]
    targets = targets[finite]

    n = len(scores)
    if n < 2:
        return DailyMetrics(date=date, ic=float("nan"), hit_rate=float("nan"),
                             long_short_spread=float("nan"), valid_n=n)

    return DailyMetrics(
        date=date,
        ic=spearman_corr(scores, targets),
        hit_rate=hit_rate(scores, targets, top_n=top_n),
        long_short_spread=long_short_spread(scores, targets, top_n=top_n),
        valid_n=n,
    )


def aggregate_metrics(daily: Sequence[DailyMetrics]) -> AggregateMetrics:
    ics = np.array([d.ic for d in daily if np.isfinite(d.ic)], dtype=np.float64)
    if ics.size == 0:
        return AggregateMetrics(
            mean_ic=float("nan"),
            std_ic=float("nan"),
            ic_ir=float("nan"),
            hit_rate=float("nan"),
            long_short_spread=float("nan"),
            n_days=0,
            per_day=list(daily),
        )
    mean = float(ics.mean())
    std = float(ics.std(ddof=0))
    ir = float(mean / std) if std > 1e-12 else float("nan")
    hit_rates = np.array([d.hit_rate for d in daily if np.isfinite(d.hit_rate)])
    spreads   = np.array([d.long_short_spread for d in daily if np.isfinite(d.long_short_spread)])
    return AggregateMetrics(
        mean_ic=mean,
        std_ic=std,
        ic_ir=ir,
        hit_rate=float(hit_rates.mean()) if hit_rates.size else float("nan"),
        long_short_spread=float(spreads.mean()) if spreads.size else float("nan"),
        n_days=int(ics.size),
        per_day=list(daily),
    )
