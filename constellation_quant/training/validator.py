"""Validation loop — no grads, returns a dict of metrics for the trainer.

Iterates the val Dataset in chronological order, runs the model, accumulates
per-date IC / hit-rate / long-short spread, and aggregates at the end.
Per-sector IC breakdown is produced when a sector tensor is available.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch

from constellation_quant.evaluation.metrics import (
    DailyMetrics,
    aggregate_metrics,
    daily_metrics,
    spearman_corr,
)
from constellation_quant.utils import get_logger

log = get_logger(__name__)


class Validator:
    """Runs a model over a chronological Dataset and computes IC + friends."""

    def __init__(
        self,
        device: torch.device,
        top_n: int = 50,
        per_sector_breakdown: bool = True,
    ):
        self.device = device
        self.top_n = top_n
        self.per_sector_breakdown = per_sector_breakdown

    @torch.no_grad()
    def evaluate(
        self,
        model,
        dataset,
        prepare_sample,          # callable: sample -> (features, mask, sectors, edges, targets, extras)
    ) -> Dict[str, Any]:
        model.eval()
        daily: List[DailyMetrics] = []
        sector_buckets: Dict[int, List[DailyMetrics]] = {}

        for i in range(len(dataset)):
            sample = dataset[i]
            inputs = prepare_sample(sample, self.device)
            out = model(
                features=inputs["features"],
                mask=inputs["mask"],
                edges=inputs.get("edges"),
                sector_indices=inputs.get("sector_indices"),
                slow_features=inputs.get("slow_features"),
            )
            scores = out.scores.detach().cpu().numpy()
            targets = inputs["targets"].detach().cpu().numpy()
            mask = inputs["mask"].detach().cpu().numpy().astype(bool)
            date = sample.get("date")

            dm = daily_metrics(scores, targets, mask=mask, top_n=self.top_n, date=date)
            daily.append(dm)

            if self.per_sector_breakdown:
                sectors_t = sample.get("sectors")
                if sectors_t is not None:
                    self._accumulate_sector_ic(
                        scores, targets, mask,
                        sectors_t.detach().cpu().numpy(),
                        date, sector_buckets,
                    )

        agg = aggregate_metrics(daily)
        out: Dict[str, Any] = agg.to_dict()
        if self.per_sector_breakdown and sector_buckets:
            per_sector: Dict[str, float] = {}
            for s_idx, bucket in sector_buckets.items():
                sub_agg = aggregate_metrics(bucket)
                per_sector[f"ic_sector_{int(s_idx)}"] = sub_agg.mean_ic
            out["per_sector_ic"] = per_sector
        return out

    @staticmethod
    def _accumulate_sector_ic(
        scores: np.ndarray,
        targets: np.ndarray,
        mask: np.ndarray,
        sectors: np.ndarray,
        date: Any,
        buckets: Dict[int, List[DailyMetrics]],
    ) -> None:
        for s in np.unique(sectors):
            if s == 0:        # 0 = unknown sector (convention from Dataset)
                continue
            sel = mask & (sectors == s)
            if sel.sum() < 3:
                continue
            ic = spearman_corr(scores[sel], targets[sel])
            if np.isfinite(ic):
                buckets.setdefault(int(s), []).append(
                    DailyMetrics(
                        date=date, ic=ic,
                        hit_rate=float("nan"),
                        long_short_spread=float("nan"),
                        valid_n=int(sel.sum()),
                    )
                )
