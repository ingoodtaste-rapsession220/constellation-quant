"""Standard Transformer encoder — full O(L²) attention.

The "does ProbSparse sacrifice quality?" ablation baseline. Uses
`nn.TransformerEncoder` under the hood with a learned positional embedding
and an attention-weighted mean pool identical to the Informer's, so any
performance delta is isolated to attention pattern alone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from constellation_quant.models.temporal.informer import (
    AttentionPooling,
    LearnablePositionalEncoding,
)


class TransformerEncoder(nn.Module):
    """Drop-in replacement for the Informer encoder with dense attention."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        pooling: str = "attention_weighted_mean",
        max_len: int = 512,
        **_: object,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pe = LearnablePositionalEncoding(max_len=max_len, d_model=d_model)
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.pooling_kind = pooling
        if pooling == "attention_weighted_mean":
            self.pool = AttentionPooling(d_model)
        else:
            self.pool = None
        self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_dropout(self.pe(self.input_proj(x)))
        h = self.encoder(h)
        if self.pool is not None:
            return self.pool(h)
        if self.pooling_kind == "last":
            return h[:, -1]
        return h.mean(dim=1)
