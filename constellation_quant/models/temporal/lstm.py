"""LSTM temporal encoder — the "does attention matter?" ablation baseline.

2-layer bidirectional LSTM producing a single vector per input sequence.
Interface matches the Informer encoder: `(B, L, F) -> (B, d_model)`, so the
master model can swap it in via the factory without touching anything else.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """2-layer bidirectional LSTM with a final linear projection to d_model."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
        **_: object,      # ignored — keeps factory args compatible
    ):
        super().__init__()
        hidden_size = d_model // (2 if bidirectional else 1)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.out_proj = nn.Linear(
            hidden_size * (2 if bidirectional else 1), d_model,
        )
        self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) -> (B, d_model)."""
        out, _ = self.lstm(x)           # (B, L, hidden * n_directions)
        return self.out_proj(out[:, -1])
