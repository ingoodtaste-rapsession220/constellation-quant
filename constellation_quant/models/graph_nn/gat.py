"""Graph Attention Network — learned edge weights via self-attention.

4-head attention by default. Like the GCN stack, uses residual connections
and LayerNorm. Attention weights are interpretable post-training: inspect
which neighbour a node attends to hardest to surface learned dependencies.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class GATStack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        **_: object,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for GATStack")
        self.dropout = dropout
        self.residual = residual

        # Each head produces `hidden_dim / attention_heads` features; GATv2Conv
        # then concatenates them back to `hidden_dim`.
        head_dim = hidden_dim // attention_heads
        if head_dim * attention_heads != hidden_dim:
            raise ValueError(
                f"hidden_dim {hidden_dim} not divisible by attention_heads {attention_heads}"
            )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev = in_dim
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(prev, head_dim, heads=attention_heads,
                          dropout=dropout, concat=True, add_self_loops=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            prev = hidden_dim

        self.input_proj = (
            nn.Linear(in_dim, hidden_dim)
            if (residual and in_dim != hidden_dim) else None
        )
        self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,   # ignored — GAT learns weights
    ) -> torch.Tensor:
        if self.residual and self.input_proj is not None:
            skip = self.input_proj(x)
        elif self.residual:
            skip = x
        else:
            skip = None

        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index)
            h = norm(F.gelu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            if skip is not None:
                h = h + skip
                skip = h
        return h
