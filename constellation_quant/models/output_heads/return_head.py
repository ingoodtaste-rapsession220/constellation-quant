"""Auxiliary head: predict continuous 5-day forward return (MSE-trained).

Acts as a regulariser on the embeddings — forcing them to carry magnitude
information, not just ordering.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ReturnHead(nn.Module):
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
        """x: (N, d) -> (N,) predicted return."""
        return self.mlp(x).squeeze(-1)
