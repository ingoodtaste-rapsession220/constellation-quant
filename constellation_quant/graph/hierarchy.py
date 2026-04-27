"""Hierarchical super-node structure: stocks ↔ sectors ↔ market.

Adds 11 GICS sector super-nodes and 1 market super-node on top of the stock
graph. Information flows bidirectionally:

    Bottom-up: stock → sector → market
        (aggregation — sector node init = attention-weighted mean of its
        constituent stocks; market node = attention-weighted mean of sectors)

    Top-down: market → sector → stock
        (conditioning via sigmoid gate — the stock embedding is updated by
        the sector and market embeddings, controlled by a learned gate so
        macro information never fully overrides stock-level signals)

This module produces the **structural** edges. The actual message-passing
and gating live in `models/graph_nn/hierarchical_mp.py` (Phase 3). Here we
return the edge indices plus the required node counts and sector-index
metadata the GNN needs to route messages.

Conventions for global node indexing (used by the GraphBuilder when
assembling the full PyG Data):
    - Stock nodes         : indices [0, N_stocks)
    - Sector super-nodes  : indices [N_stocks, N_stocks + N_sectors)
    - Market super-node   : index   N_stocks + N_sectors
    Total nodes           : N_stocks + N_sectors + 1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence

import numpy as np


# Canonical GICS sector ordering — stable index 0..10 across all runs.
GICS_SECTORS: List[str] = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

MARKET_NODE_NAME = "Market"


@dataclass
class HierarchicalSpec:
    """Structural output of `HierarchyBuilder.build()`."""

    stock_to_sector: np.ndarray      # (2, N_stocks)   int64 global indices
    sector_to_market: np.ndarray     # (2, N_sectors)  int64
    sector_to_stock: np.ndarray      # (2, N_stocks)   reverse
    market_to_sector: np.ndarray     # (2, N_sectors)  reverse
    sector_indices: np.ndarray       # (N_stocks,)     sector index (0..N_sectors-1) per stock, -1 = unknown
    n_stock_nodes: int
    n_sector_nodes: int
    market_node_index: int

    @property
    def total_nodes(self) -> int:
        return self.n_stock_nodes + self.n_sector_nodes + 1


class HierarchyBuilder:
    """Builds the super-node edge structure for a given universe.

    Args:
        sector_map: ticker -> GICS sector label. Unknown tickers are routed
            to the market node only (no sector edge).
        sectors: Optional explicit sector ordering. Defaults to GICS_SECTORS.
        bidirectional: If True (default), emits both upward and downward
            edges so the GNN can do the two-pass hierarchical pass in a
            single message-passing round.
    """

    def __init__(
        self,
        sector_map: Mapping[str, str],
        sectors: Sequence[str] = GICS_SECTORS,
        bidirectional: bool = True,
    ):
        self.sector_map = {k.upper(): v for k, v in sector_map.items() if v}
        self.sectors = list(sectors)
        self.sector_to_idx = {s: i for i, s in enumerate(self.sectors)}
        self.bidirectional = bidirectional

    def build(self, universe_tickers: Sequence[str]) -> HierarchicalSpec:
        tickers = [t.upper() for t in universe_tickers]
        n_stocks = len(tickers)
        n_sectors = len(self.sectors)
        sector_base = n_stocks
        market_idx = n_stocks + n_sectors

        stock_to_sector_src: List[int] = []
        stock_to_sector_dst: List[int] = []
        sector_indices = np.full(n_stocks, fill_value=-1, dtype=np.int64)

        for i, ticker in enumerate(tickers):
            sector = self.sector_map.get(ticker)
            if sector is None or sector not in self.sector_to_idx:
                continue
            s_idx = self.sector_to_idx[sector]
            sector_indices[i] = s_idx
            stock_to_sector_src.append(i)
            stock_to_sector_dst.append(sector_base + s_idx)

        # Every sector node connects to the market node.
        sector_to_market_src = np.arange(sector_base, sector_base + n_sectors, dtype=np.int64)
        sector_to_market_dst = np.full(n_sectors, market_idx, dtype=np.int64)

        stock_to_sector = np.stack([
            np.asarray(stock_to_sector_src, dtype=np.int64),
            np.asarray(stock_to_sector_dst, dtype=np.int64),
        ], axis=0) if stock_to_sector_src else np.zeros((2, 0), dtype=np.int64)

        sector_to_market = np.stack([sector_to_market_src, sector_to_market_dst], axis=0)

        # Reverse directions for top-down flow.
        sector_to_stock = (
            stock_to_sector[[1, 0]] if stock_to_sector.size else np.zeros((2, 0), dtype=np.int64)
        )
        market_to_sector = sector_to_market[[1, 0]]

        return HierarchicalSpec(
            stock_to_sector=stock_to_sector,
            sector_to_market=sector_to_market,
            sector_to_stock=sector_to_stock,
            market_to_sector=market_to_sector,
            sector_indices=sector_indices,
            n_stock_nodes=n_stocks,
            n_sector_nodes=n_sectors,
            market_node_index=market_idx,
        )
