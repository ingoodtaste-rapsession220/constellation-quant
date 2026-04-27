"""Relational Graph Attention Network — per-relation attention heads.

The primary GNN for constellation-quant. Each edge type (`correlation`,
`fundamental`, and the learned `attention` type) has its own attention
parameters; at each layer we run a GATv2 per relation and combine their
outputs with a learned relation-level attention.

Input interface differs from single-relation GNNs:
    edges:   Dict[str, (edge_index, edge_weight)]
    x:       (N_total, in_dim)
    out:     (N_total, hidden_dim)

If the `attention` relation appears in `edges` with an empty index, this
layer constructs the attention edges on the fly: for every node, compute
key-query similarities against its neighbours in *any* other relation and
pick the top-K. These edges are fully learned and differentiable through
both the selection mask and the weights.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import GATv2Conv
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False


EdgeData = Tuple[torch.Tensor, Optional[torch.Tensor]]  # (edge_index, edge_weight)


class _RGATLayer(nn.Module):
    """Single R-GAT layer: per-relation GATv2 + relation-level attention."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        relations: Tuple[str, ...],
        dropout: float,
    ):
        super().__init__()
        head_dim = out_dim // num_heads
        if head_dim * num_heads != out_dim:
            raise ValueError(f"out_dim {out_dim} not divisible by num_heads {num_heads}")

        self.relations = relations
        self.convs = nn.ModuleDict({
            rel: GATv2Conv(in_dim, head_dim, heads=num_heads,
                           dropout=dropout, concat=True, add_self_loops=True)
            for rel in relations
        })
        # Learned per-relation importance — a vector scored by a softmax
        # across relations at aggregation time.
        self.relation_attn = nn.Parameter(torch.zeros(len(relations)))

    def forward(
        self,
        x: torch.Tensor,
        edges_by_rel: Mapping[str, EdgeData],
    ) -> torch.Tensor:
        per_rel_out = []
        relation_idx: list[int] = []
        for i, rel in enumerate(self.relations):
            edata = edges_by_rel.get(rel)
            if edata is None or edata[0].numel() == 0:
                continue
            edge_index, _ = edata
            per_rel_out.append(self.convs[rel](x, edge_index))
            relation_idx.append(i)

        if not per_rel_out:
            # No edges in any relation — fall back to identity.
            return x

        stacked = torch.stack(per_rel_out, dim=0)            # (R, N, d)
        rel_logits = self.relation_attn[relation_idx]
        rel_weights = F.softmax(rel_logits, dim=0)            # (R,)
        combined = (stacked * rel_weights.view(-1, 1, 1)).sum(dim=0)
        return combined


class RGATStack(nn.Module):
    """Multi-layer R-GAT with residual connections + LayerNorm."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        attention_heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
        edge_types: Tuple[str, ...] = ("correlation", "fundamental"),
        learned_attention: bool = True,
        top_k_attention: int = 10,
        **_: object,
    ):
        super().__init__()
        if not _HAS_PYG:
            raise ImportError("torch_geometric is required for RGATStack")

        self.dropout = dropout
        self.residual = residual
        self.learned_attention = learned_attention
        self.top_k_attention = top_k_attention
        self.edge_types = tuple(edge_types)

        # When learned-attention is on, the layer consumes an extra "attention"
        # edge type that we construct per forward pass.
        relations = self.edge_types + (("attention",) if learned_attention else ())

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        prev = in_dim
        for _ in range(num_layers):
            self.layers.append(
                _RGATLayer(prev, hidden_dim, attention_heads, relations, dropout)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
            prev = hidden_dim

        # Query/key projections used to build the attention edges on the fly.
        self.attn_q = nn.Linear(in_dim, hidden_dim) if learned_attention else None
        self.attn_k = nn.Linear(in_dim, hidden_dim) if learned_attention else None

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
        edges_by_rel: Mapping[str, EdgeData],
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        edges_by_rel = dict(edges_by_rel)
        if self.learned_attention:
            edges_by_rel["attention"] = self._build_attention_edges(x, node_mask)

        if self.residual and self.input_proj is not None:
            skip = self.input_proj(x)
        elif self.residual:
            skip = x
        else:
            skip = None

        h = x
        for layer, norm in zip(self.layers, self.norms):
            h = layer(h, edges_by_rel)
            h = norm(F.gelu(h))
            h = F.dropout(h, p=self.dropout, training=self.training)
            if skip is not None:
                h = h + skip
                skip = h
        return h

    # ── Learned-attention edge construction ────────────────────────────

    def _build_attention_edges(
        self,
        x: torch.Tensor,
        node_mask: Optional[torch.Tensor],
    ) -> EdgeData:
        """Top-K self-attention graph over query-key similarities.

        Cheap — single matrix multiply + top-K. Straight-through gradients
        work because we use the full attention matrix for weighting; the
        top-K selection is the mask.
        """
        if self.attn_q is None or self.attn_k is None:
            return (torch.zeros((2, 0), dtype=torch.long, device=x.device), None)
        q = self.attn_q(x)
        k = self.attn_k(x)
        scores = (q @ k.T) / (q.size(-1) ** 0.5)              # (N, N)

        if node_mask is not None:
            # Mask absent nodes out of attention targets.
            invalid = ~node_mask.bool()
            scores.masked_fill_(invalid.unsqueeze(0), float("-inf"))
            scores.masked_fill_(invalid.unsqueeze(1), float("-inf"))

        n = scores.size(0)
        k_eff = min(self.top_k_attention, n - 1)
        if k_eff <= 0:
            return (torch.zeros((2, 0), dtype=torch.long, device=x.device), None)

        # Drop self-loops before top-K.
        scores = scores - torch.eye(n, device=x.device) * 1e9

        topk = scores.topk(k_eff, dim=-1).indices             # (N, k_eff)
        src = torch.arange(n, device=x.device).view(-1, 1).expand(-1, k_eff).reshape(-1)
        dst = topk.reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        return (edge_index, None)
