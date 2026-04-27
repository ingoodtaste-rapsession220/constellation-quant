"""Temporal Convolutional Network — the "attention isn't needed" baseline.

Stack of dilated causal convolutions, following Bai et al. 2018. Dilation
doubles at each layer so the receptive field grows as 2^N, covering a
60-day window with 6 layers (dilation = 1, 2, 4, 8, 16, 32). Each block is:
    causal Conv1d → BatchNorm1d → GELU → Dropout → Conv1d → residual → GELU

No attention anywhere; no quadratic scaling. Meant to test how much of the
Informer's value comes from attention vs just having enough receptive
field over the lookback.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _CausalConv1d(nn.Conv1d):
    """1-D conv where output[t] sees only input[:t]. Uses left-padding only."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        self._left_pad = (kernel_size - 1) * dilation
        super().__init__(
            in_ch, out_ch, kernel_size=kernel_size,
            padding=0, dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self._left_pad, 0))
        return super().forward(x)


class _TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = _CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(F.gelu(self.norm1(self.conv1(x))))
        x = self.dropout(F.gelu(self.norm2(self.conv2(x))))
        return F.gelu(x + residual)


class TCNEncoder(nn.Module):
    """Stacked dilated causal conv blocks → last-step projection."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        num_blocks: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(n_features, d_model, kernel_size=1)
        dilations: List[int] = [2 ** i for i in range(num_blocks)]
        self.blocks = nn.ModuleList([
            _TCNBlock(d_model, kernel_size, d, dropout) for d in dilations
        ])
        self.out_proj = nn.Linear(d_model, d_model)
        self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) -> (B, d_model)."""
        h = self.input_proj(x.transpose(1, 2))        # (B, d, L)
        for block in self.blocks:
            h = block(h)
        return self.out_proj(h[:, :, -1])              # pick the last time step
