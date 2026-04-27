"""Primary output head: MLP → single scalar ranking score per stock.

The last layer optionally applies temperature scaling (a learned scalar
multiplier) that sharpens or softens the score distribution at inference
time — useful if the ranking loss collapses scores into a narrow band.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class RankingHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: List[int] = (128, 64),
        dropout: float = 0.1,
        temperature_scaling: bool = True,
        batch_norm: bool = True,
    ):
        super().__init__()
        # Use LayerNorm rather than BatchNorm1d: the head operates on
        # (N_stocks, d) tensors where N_stocks mixes real and padded slots,
        # so BN running stats get polluted and shift between train and eval.
        # LayerNorm normalises per-stock and is invariant to the padding.
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batch_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

        if temperature_scaling:
            # Initialised to 1 (no scaling); learnable.
            self.temperature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("temperature", torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, in_dim) -> (N,) scalar score per node."""
        score = self.mlp(x).squeeze(-1)
        return score * self.temperature
