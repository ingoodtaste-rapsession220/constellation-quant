"""Auxiliary head: predict 5-day forward realised volatility (always positive).

Uses Softplus on the output to guarantee positivity. Feeds into risk-parity
portfolio construction downstream (scale positions inversely by predicted vol).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class VolatilityHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: List[int] = (128,),
        dropout: float = 0.1,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.mlp(x).squeeze(-1)
        return F.softplus(raw)                # strictly positive
