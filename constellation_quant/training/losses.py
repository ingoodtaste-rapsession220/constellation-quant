"""Loss functions.

    - ListMLE : primary ranking loss. Optimises the likelihood of the true
                return ordering under the predicted score distribution.
                Numerically stable via log-sum-exp; supports masking.

    - LambdaRank : pairwise ranking alternative for ablation.

    - MSE  : auxiliary magnitude loss (return + volatility heads).

    - MultiTaskLoss : weighted sum of per-head losses according to
                      `training_config.yaml > losses`.

All losses accept a `mask` tensor so padded stock slots don't contribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Ranking losses ─────────────────────────────────────────────────────────


class ListMLELoss(nn.Module):
    """ListMLE (Xia et al., 2008) — listwise ranking loss.

    Given predicted scores `s` and true labels `y` (forward returns here),
    the probability of the true ordering π under the Plackett-Luce model is:

        P(π | s) = ∏_t [ exp(s_{π_t}) / Σ_{k=t}^{N} exp(s_{π_k}) ]

    The negative log-likelihood is a convex function of `s`. We compute it
    stably via log-sum-exp, applied over the descending-by-y ordering.
    """

    def __init__(self, eps: float = 1e-9):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        scores:  torch.Tensor,                # (N,) or (B, N)
        targets: torch.Tensor,                # same shape
        mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        B, N = scores.shape

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)

        # Mask invalid slots by pushing their score to -inf (excluded from
        # the log-sum-exp denominators and skipped in the ordering).
        scores_masked = scores.masked_fill(~mask, float("-inf"))
        # For the ordering we need to sort by target descending within the mask.
        # Absent slots get target = -inf so they sort to the very end.
        targets_for_sort = targets.masked_fill(~mask, float("-inf"))
        _, order = targets_for_sort.sort(dim=1, descending=True)
        sorted_scores = torch.gather(scores_masked, 1, order)
        sorted_mask   = torch.gather(mask.float(), 1, order)

        # At each position t, the "denominator" is logsumexp(scores[t:]).
        reversed_scores = torch.flip(sorted_scores, dims=[1])
        cum_logsumexp = torch.logcumsumexp(reversed_scores, dim=1)
        denom = torch.flip(cum_logsumexp, dims=[1])

        valid_count = sorted_mask.sum(dim=1).clamp_min(1.0)
        # At padded positions sorted_scores and denom are both -inf, so their
        # difference is NaN. nan * 0 is still NaN in IEEE 754 and poisons the
        # sum — wipe non-finite diffs to 0 before masking.
        diff = denom - sorted_scores
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        loss = (diff * sorted_mask).sum(dim=1) / valid_count

        # If a row has zero valid slots, ignore it.
        finite = torch.isfinite(loss)
        if not finite.any():
            return scores.sum() * 0.0          # zero with grad_fn
        return loss[finite].mean()


class LambdaRankLoss(nn.Module):
    """Pairwise ranking loss — a simpler ablation alternative to ListMLE.

    For every pair (i, j) in the batch with y_i > y_j, penalises
    `max(0, margin - (s_i - s_j))`. Hinge-style.
    """

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        scores: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        B, N = scores.shape
        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)

        total_loss = scores.new_zeros(())
        total_pairs = 0
        for b in range(B):
            valid = mask[b]
            s = scores[b][valid]
            y = targets[b][valid]
            if s.numel() < 2:
                continue
            s_diff = s.unsqueeze(0) - s.unsqueeze(1)
            y_diff = y.unsqueeze(0) - y.unsqueeze(1)
            positive_pair = y_diff > 0
            pair_loss = F.relu(self.margin - s_diff) * positive_pair.float()
            total_loss = total_loss + pair_loss.sum()
            total_pairs += positive_pair.sum().item()
        if total_pairs == 0:
            return scores.sum() * 0.0
        return total_loss / total_pairs


# ── Auxiliary losses ───────────────────────────────────────────────────────


class ICMaximizationLoss(nn.Module):
    """Information-Coefficient maximisation loss (a.k.a. Pearson IC loss).

    Per prediction date, computes the Pearson correlation between the model's
    scores and the true forward returns across the cross-section of valid
    stocks, then returns its negation so that *minimising* this loss
    *maximises* IC — exactly the metric we evaluate on.

    Compared with ListMLE this loss
      - allocates gradient proportional to how each stock's score deviates
        from the cross-sectional mean, weighted by how its target deviates
        from the cross-sectional mean — naturally putting more pressure
        on the *extremes* of the distribution (the long-short tails)
        than on the noisy middle;
      - is shift-invariant in scores and targets (subtracting the mean
        of either is a no-op);
      - is scale-invariant (dividing by std is normalised away);
      - directly optimises the metric used in `aggregate_metrics`.

    Vectorised over the batch of dates so the per-date Pearson is
    computed in parallel without a Python loop.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        scores:  torch.Tensor,                # (N,) or (B, N)
        targets: torch.Tensor,                # same shape
        mask:    Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        B, N = scores.shape

        if mask is None:
            mask = torch.ones_like(scores, dtype=torch.bool)
        m = mask.float()
        n_valid = m.sum(dim=1, keepdim=True).clamp_min(1.0)             # (B,1)

        # Per-date means computed only over valid positions. Padded slots
        # contribute zero before/after centring (because we multiply by
        # `m` again right after subtracting the mean).
        s_mean = (scores * m).sum(dim=1, keepdim=True) / n_valid
        t_mean = (targets * m).sum(dim=1, keepdim=True) / n_valid
        s_c = (scores - s_mean) * m
        t_c = (targets - t_mean) * m

        num   = (s_c * t_c).sum(dim=1)                                  # (B,)
        denom = torch.sqrt(
            (s_c * s_c).sum(dim=1) * (t_c * t_c).sum(dim=1) + self.eps
        )
        corr = num / (denom + self.eps)                                 # (B,)

        # Need at least 3 valid points for Pearson to be meaningful.
        valid = n_valid.squeeze(-1) >= 3.0
        if not valid.any():
            return scores.sum() * 0.0          # zero with grad_fn
        return -corr[valid].mean()             # negate: minimise → maximise IC


class MaskedMSELoss(nn.Module):
    """MSE that ignores padded entries."""

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is None:
            return F.mse_loss(pred, target)
        mask_f = mask.float()
        denom = mask_f.sum().clamp_min(1.0)
        sq = (pred - target).pow(2) * mask_f
        return sq.sum() / denom


# ── Multi-task combiner ────────────────────────────────────────────────────


@dataclass
class LossComponents:
    """Returned by MultiTaskLoss so we can log each component separately."""
    total: torch.Tensor
    per_component: Dict[str, torch.Tensor]


class MultiTaskLoss(nn.Module):
    """Weighted sum of ranking + return + volatility losses.

    Builds from `training_config.yaml > losses`, e.g.:
        {
            "ranking":    {"name": "listmle", "weight": 1.0},
            "return":     {"name": "mse",     "weight": 0.1},
            "volatility": {"name": "mse",     "weight": 0.05},
        }
    """

    def __init__(self, config: Mapping[str, Mapping[str, object]]):
        super().__init__()
        self.weights: Dict[str, float] = {}
        self.modules_by_name = nn.ModuleDict()

        for name, sub in config.items():
            sub = dict(sub)
            loss_name = str(sub.get("name", "")).lower()
            weight = float(sub.get("weight", 0.0))
            if weight == 0.0:
                continue
            self.weights[name] = weight

            if loss_name == "listmle":
                self.modules_by_name[name] = ListMLELoss()
            elif loss_name == "lambdarank":
                self.modules_by_name[name] = LambdaRankLoss()
            elif loss_name in ("ic_max", "ic_maximisation", "ic_maximization", "ic"):
                self.modules_by_name[name] = ICMaximizationLoss()
            elif loss_name == "mse":
                self.modules_by_name[name] = MaskedMSELoss()
            else:
                raise ValueError(f"Unknown loss name {loss_name!r} for key {name!r}")

    def forward(
        self,
        predictions: Mapping[str, torch.Tensor],
        targets: Mapping[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> LossComponents:
        total = None
        per_component: Dict[str, torch.Tensor] = {}
        for name, mod in self.modules_by_name.items():
            if name not in predictions or name not in targets:
                continue
            val = mod(predictions[name], targets[name], mask=mask)
            per_component[name] = val.detach()
            weighted = self.weights[name] * val
            total = weighted if total is None else total + weighted
        if total is None:
            raise ValueError("MultiTaskLoss: no active components.")
        return LossComponents(total=total, per_component=per_component)
