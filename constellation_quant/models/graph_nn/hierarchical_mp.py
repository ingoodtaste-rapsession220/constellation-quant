"""Hierarchical message passing: stocks ↔ sectors ↔ market.

Wraps a base GNN's output with bidirectional hierarchical aggregation.

Flow:

    1. Bottom-up aggregation
       - `sector_h = attention_weighted_mean(stock_h of that sector's members)`
       - `market_h = attention_weighted_mean(sector_h)`

    2. Top-down gated conditioning
       - For each stock i with sector s:
           top_down_i = Linear(concat(sector_h[s], market_h))
           gate_i    = sigmoid( Linear(concat(stock_h[i], top_down_i)) )
           stock_h[i] ← (1 - gate_i) · stock_h[i] + gate_i · top_down_i

    The sigmoid gate means macro information never fully overrides a
    stock-level signal; it just biases it.

Inputs come from the `GraphBuilder` via `BuiltGraph.sector_indices`
(N_stocks,) and `hierarchy.n_sector_nodes`.

Output: updated stock-node embeddings of the same shape.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalMessagePassing(nn.Module):
    """Bidirectional stock↔sector↔market aggregation + top-down gating.

    Args:
        d_model: Stock-embedding dimension (and sector/market dim).
        n_sectors: Number of GICS sector super-nodes.
        unknown_sector_index: Sentinel for stocks with no GICS assignment.
    """

    def __init__(
        self,
        d_model: int,
        n_sectors: int = 11,
        unknown_sector_index: int = -1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_sectors = n_sectors
        self.unknown = unknown_sector_index

        # Bottom-up: scalar attention score per stock for its sector mean,
        # and per sector for the market mean.
        self.stock_attn = nn.Linear(d_model, 1)
        self.sector_attn = nn.Linear(d_model, 1)

        # Top-down: project (sector, market) → condition, then gate.
        self.top_down_proj = nn.Linear(2 * d_model, d_model)
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        stock_h: torch.Tensor,           # (N_stocks, d_model)
        sector_indices: torch.Tensor,    # (N_stocks,) int64 — sector idx per stock, -1 unknown
        node_mask: Optional[torch.Tensor] = None,  # (N_stocks,) bool
    ) -> torch.Tensor:
        device = stock_h.device
        if node_mask is None:
            node_mask = torch.ones(stock_h.size(0), dtype=torch.bool, device=device)

        # 1) Bottom-up: stock → sector.
        sector_h = self._bottom_up_sector(stock_h, sector_indices, node_mask)    # (S, d)

        # 2) Bottom-up: sector → market.
        market_h = self._bottom_up_market(sector_h)                              # (d,)

        # 3) Top-down: condition each stock on its sector + the market.
        return self._top_down(stock_h, sector_h, market_h, sector_indices)

    # ── Components ─────────────────────────────────────────────────────

    def _bottom_up_sector(
        self,
        stock_h: torch.Tensor,
        sector_indices: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-weighted mean of stock embeddings within each sector."""
        N, D = stock_h.shape
        scores = self.stock_attn(stock_h).squeeze(-1)                            # (N,)

        # Mask absent stocks and unknown sectors out of every sector's softmax.
        valid = node_mask & (sector_indices >= 0)
        scores = scores.masked_fill(~valid, float("-inf"))

        sector_h = torch.zeros(self.n_sectors, D, device=stock_h.device, dtype=stock_h.dtype)
        for s in range(self.n_sectors):
            members = (sector_indices == s) & valid
            if not members.any():
                continue
            member_scores = scores[members]
            weights = F.softmax(member_scores, dim=0).unsqueeze(-1)              # (|members|, 1)
            sector_h[s] = (stock_h[members] * weights).sum(dim=0)
        return sector_h

    def _bottom_up_market(self, sector_h: torch.Tensor) -> torch.Tensor:
        """Attention-weighted mean of populated sector embeddings."""
        scores = self.sector_attn(sector_h).squeeze(-1)                          # (S,)
        populated = sector_h.norm(dim=-1) > 0
        if not populated.any():
            return sector_h.mean(dim=0)
        scores = scores.masked_fill(~populated, float("-inf"))
        weights = F.softmax(scores, dim=0).unsqueeze(-1)
        return (sector_h * weights).sum(dim=0)

    def _top_down(
        self,
        stock_h: torch.Tensor,
        sector_h: torch.Tensor,
        market_h: torch.Tensor,
        sector_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gate stock embeddings with their (sector, market) context."""
        N = stock_h.size(0)

        # Pull the per-stock sector embedding; unknown sectors use the
        # market vector alone.
        safe_idx = sector_indices.clamp(min=0)
        per_stock_sector = sector_h.index_select(0, safe_idx)                    # (N, d)
        missing = (sector_indices < 0).unsqueeze(-1)
        per_stock_sector = torch.where(missing, market_h.expand_as(per_stock_sector), per_stock_sector)

        # Concatenate with the market vector and project down.
        market_broadcast = market_h.unsqueeze(0).expand(N, -1)
        top_down = self.top_down_proj(torch.cat([per_stock_sector, market_broadcast], dim=-1))

        # Gate: how much top-down info to blend into each stock.
        gate_val = self.gate(torch.cat([stock_h, top_down], dim=-1))             # (N, d)
        return (1 - gate_val) * stock_h + gate_val * top_down
