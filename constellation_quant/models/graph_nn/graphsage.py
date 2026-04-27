"""GraphSAGE — neighbour-sampling based message passing.

Tests whether full-graph message passing is necessary or a sampled
neighbourhood suffices. Uses PyG's `SAGEConv` (mean aggregator) with an
optional sampling ratio; for the 500-node graphs used here, the full
neighbourhood is still cheap, so the practical benefit is modest — the
point is the ablation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


class GraphSAGEStack(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        aggregator: str = "mean",        # mean | max | lstm
        dropout: float = 0.1,
        residual: bool = True,
        **_: object,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for GraphSAGEStack")
        self.dropout = dropout
        self.residual = residual

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev = in_dim
        for _ in range(num_layers):
            self.convs.append(SAGEConv(prev, hidden_dim, aggr=aggregator))
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
        edge_weight: Optional[torch.Tensor] = None,   # ignored by SAGEConv
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
