"""Standard multi-layer Graph Convolutional Network.

Baseline for answering "does anything beyond simple averaging help?". Each
layer is a symmetric-normalised neighbour aggregation + linear + non-linearity
+ residual. Residuals stabilise deeper stacks — without them, a 3-layer GCN
on a 500-node graph tends to over-smooth.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GCNConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class GCNStack(nn.Module):
    """Residual GCN stack.

    forward(x, edge_index, edge_weight=None) -> updated node embeddings
        x          : (N_total, in_dim)
        edge_index : (2, E) int64
        edge_weight: (E,)   optional float

    Output has the same N_total; feature dim becomes `hidden_dim`.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        residual: bool = True,
        **_: object,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for GCNStack")
        self.dropout = dropout
        self.residual = residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev = in_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(prev, hidden_dim, add_self_loops=True))
            self.norms.append(nn.LayerNorm(hidden_dim))
            prev = hidden_dim

        # Residual requires matching dims at input; project if needed.
        self.input_proj = nn.Linear(in_dim, hidden_dim) if (residual and in_dim != hidden_dim) else None
        self._output_dim = hidden_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.residual and self.input_proj is not None:
            skip = self.input_proj(x)
        elif self.residual:
            skip = x
        else:
            skip = None

        h = x
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h = conv(h, edge_index, edge_weight=edge_weight)
            h = norm(F.gelu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            if skip is not None:
                h = h + skip
                skip = h
        return h
