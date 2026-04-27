"""Simplified Mamba / selective state-space model baseline.

Not the full paper implementation — we use a straightforward S4-style linear
SSM with input-dependent gating and a depthwise conv, which captures the
essence of selective state-space models without the custom CUDA kernels.
Acts as the "linear-time alternative to attention" ablation.

Per-layer flow:
    (B, L, d)  ─► depthwise conv  ─► SSM (diagonal state)  ─► gate (sigmoid·silu)
                                                                       │
                                                                       ▼
                                                                  (B, L, d)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DiagonalSSM(nn.Module):
    """Diagonal state-space: y_t = x_t + alpha · y_{t-1}.

    alpha is per-channel, learnable, and constrained to (0, 1) via sigmoid
    so the recurrence is stable. This is a linear-time drop-in for attention
    over the temporal dim — no quadratic cost.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.zeros(d_model))   # sigmoid(0) = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d). Returns (B, L, d)."""
        alpha = torch.sigmoid(self.log_alpha)                  # (d,)
        B, L, D = x.shape
        outputs = torch.empty_like(x)
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(L):
            state = x[:, t] + alpha * state
            outputs[:, t] = state
        return outputs


class _MambaBlock(nn.Module):
    """SSM + conv + gating, following the Mamba-style block layout."""

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1,
                 conv_kernel: int = 4):
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_state)         # split: (x, gate)
        self.dw_conv = nn.Conv1d(
            d_state, d_state, kernel_size=conv_kernel,
            groups=d_state, padding=conv_kernel - 1,
        )
        self.ssm = _DiagonalSSM(d_state)
        self.out_proj = nn.Linear(d_state, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)                                    # (B, L, 2·d_state)
        x_part, z_part = xz.chunk(2, dim=-1)

        # Depthwise causal conv on x.
        x_conv = self.dw_conv(x_part.transpose(1, 2))[:, :, : x_part.size(1)]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM + output gate.
        h = self.ssm(x_conv) * F.silu(z_part)
        out = self.dropout(self.out_proj(h))
        return residual + out


class MambaEncoder(nn.Module):
    """Simplified Mamba encoder. Same I/O contract as the Informer."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 3,
        dropout: float = 0.1,
        **_: object,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.blocks = nn.ModuleList([
            _MambaBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self._output_dim = d_model

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        return h[:, -1]                      # last-step pool
