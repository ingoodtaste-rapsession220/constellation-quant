"""Master graph orchestrator.

Reads the model config, instantiates the enabled edge builders (correlation /
sector / fundamental / hierarchical), and for any prediction date assembles
the complete PyG `HeteroData` (or homogeneous `Data`) graph.

Output semantics:

    * Homogeneous mode (no hierarchy): returns a single `Data` with one
      `edge_index` per enabled edge type, stored as
      `edge_index_{type}` + `edge_weight_{type}`. The GNN consumes these
      via a multi-relational message-passing layer (R-GAT in Phase 3).

    * Hierarchical mode: returns a `HeteroData` with node types
      {`stock`, `sector`, `market`} and edge types for each relation
      (`stock -> sector`, `sector -> market`, and their reverses).
      Stock-stock edges keep their multi-relational typing.

Either way, variable-size graphs are handled natively — the caller supplies
the universe (current S&P 500 members on `pred_date`) and we index edges
into it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from constellation_quant.graph.correlation_edges import (
    CorrelationEdgeBuilder,
    EdgeSpec,
    prepare_log_returns,
)
from constellation_quant.graph.fundamental_edges import FundamentalEdgeBuilder
from constellation_quant.graph.hierarchy import HierarchicalSpec, HierarchyBuilder
from constellation_quant.graph.sector_edges import SectorEdgeBuilder
from constellation_quant.utils import get_logger

log = get_logger(__name__)


@dataclass
class BuiltGraph:
    """Bundle of all tensors the trainer needs for one prediction date.

    `node_features` is an (N_total, F) array: stock slots first, then sector
    slots, then the single market slot (if hierarchy is enabled). Entries
    for absent stocks (dynamic-membership padding) are zeros and masked out
    via `node_mask`.

    Edge tensors are grouped by type — the GNN consumes them as per-relation
    inputs to an R-GAT layer.
    """

    node_features: np.ndarray           # (N_total, F) float32
    node_mask: np.ndarray               # (N_total,)   bool  — True for real stocks + super-nodes
    sector_indices: np.ndarray          # (N_stocks,)  int64 — sector per stock, -1 = unknown
    edges: Dict[str, EdgeSpec]          # edge type -> spec  (indices are into full node list)
    hierarchy: Optional[HierarchicalSpec]
    tickers: list
    date: pd.Timestamp


class GraphBuilder:
    """Assemble PyG-ready graph tensors for any prediction date + universe.

    Args:
        model_cfg: Parsed model_config.yaml dict.
        sector_map: ticker -> GICS sector (used by sector + hierarchy builders).
        returns_wide: Wide DataFrame of log returns (rows=date, cols=ticker).
            Required when correlation edges are enabled.
    """

    def __init__(
        self,
        model_cfg: Mapping[str, Any],
        sector_map: Optional[Mapping[str, str]] = None,
        returns_wide: Optional[pd.DataFrame] = None,
    ):
        self.cfg = dict(model_cfg)
        self.graph_cfg = dict(self.cfg.get("graph", {}) or {})
        self.edge_types = list(self.graph_cfg.get("edge_types", []))
        self.hierarchy_enabled = bool(
            (self.cfg.get("hierarchy", {}) or {}).get("enabled", False)
        )
        self.sector_map = dict(sector_map or {})
        self.returns_wide = returns_wide

        self._edge_builders = self._init_edge_builders()
        self._hierarchy_builder = (
            HierarchyBuilder(self.sector_map) if self.hierarchy_enabled else None
        )

    # ── Public API ─────────────────────────────────────────────────────

    def build(
        self,
        pred_date: pd.Timestamp,
        universe_tickers: Sequence[str],
        node_features: np.ndarray,
        fundamental_vectors: Optional[pd.DataFrame] = None,
    ) -> BuiltGraph:
        """Assemble the full graph for one prediction date.

        Args:
            pred_date: The prediction date `t`. Used to slice rolling windows.
            universe_tickers: S&P 500 members on `pred_date` (their canonical
                order determines node indexing 0..N-1).
            node_features: (N_stocks, F) array — per-stock embeddings for
                stocks in the universe. Positions corresponding to padded
                slots (e.g. non-members) should already be zero.
            fundamental_vectors: Optional DataFrame indexed by ticker with
                columns = fundamental features. Required iff "fundamental"
                is in edge_types.
        """
        universe = [t.upper() for t in universe_tickers]
        n_stocks = len(universe)
        if node_features.shape[0] != n_stocks:
            raise ValueError(
                f"node_features has {node_features.shape[0]} rows, "
                f"expected {n_stocks} to match universe_tickers."
            )

        # Per-relation stock-stock edges.
        edges: Dict[str, EdgeSpec] = {}
        for edge_type, builder in self._edge_builders.items():
            edges[edge_type] = self._build_edge_type(
                edge_type, builder, pred_date, universe, fundamental_vectors,
            )

        hierarchy = self._hierarchy_builder.build(universe) if self._hierarchy_builder else None

        # Assemble full node-feature matrix with super-node slots appended.
        full_features, node_mask = self._assemble_node_features(
            node_features, hierarchy, edges,
        )

        return BuiltGraph(
            node_features=full_features,
            node_mask=node_mask,
            sector_indices=(hierarchy.sector_indices if hierarchy is not None
                            else self._default_sector_indices(universe)),
            edges=edges,
            hierarchy=hierarchy,
            tickers=universe,
            date=pd.Timestamp(pred_date),
        )

    def to_pyg(self, built: BuiltGraph) -> Any:
        """Convert a BuiltGraph to PyTorch Geometric `Data` / `HeteroData`.

        Lazy import — keeps `graph_builder.build()` usable without PyG for
        tests that don't need the torch conversion.
        """
        import torch
        try:
            from torch_geometric.data import Data, HeteroData
        except ImportError as exc:
            raise ImportError(
                "torch_geometric is required for GraphBuilder.to_pyg()."
            ) from exc

        if built.hierarchy is None:
            data = Data()
            data.x = torch.from_numpy(built.node_features).float()
            data.node_mask = torch.from_numpy(built.node_mask).bool()
            data.sector_indices = torch.from_numpy(built.sector_indices).long()
            for edge_type, spec in built.edges.items():
                setattr(data, f"edge_index_{edge_type}",
                        torch.from_numpy(spec.edge_index).long())
                setattr(data, f"edge_weight_{edge_type}",
                        torch.from_numpy(spec.edge_weight).float())
            return data

        hd = HeteroData()
        n_stocks = built.hierarchy.n_stock_nodes
        n_sectors = built.hierarchy.n_sector_nodes
        stock_feats  = built.node_features[:n_stocks]
        sector_feats = built.node_features[n_stocks : n_stocks + n_sectors]
        market_feats = built.node_features[-1:]

        hd["stock"].x  = torch.from_numpy(stock_feats).float()
        hd["sector"].x = torch.from_numpy(sector_feats).float()
        hd["market"].x = torch.from_numpy(market_feats).float()
        hd["stock"].mask = torch.from_numpy(built.node_mask[:n_stocks]).bool()
        hd["stock"].sector_indices = torch.from_numpy(built.sector_indices).long()

        # Stock-stock multi-relational edges (local to the stock slice).
        for edge_type, spec in built.edges.items():
            hd["stock", edge_type, "stock"].edge_index = torch.from_numpy(spec.edge_index).long()
            hd["stock", edge_type, "stock"].edge_weight = torch.from_numpy(spec.edge_weight).float()

        # Hierarchical edges: shift indices to be local to the node type.
        stock_to_sector = built.hierarchy.stock_to_sector.copy()
        stock_to_sector[1] -= n_stocks
        sector_to_market = built.hierarchy.sector_to_market.copy()
        sector_to_market[0] -= n_stocks
        sector_to_market[1] -= n_stocks + n_sectors

        hd["stock", "in", "sector"].edge_index = torch.from_numpy(stock_to_sector).long()
        hd["sector", "contains", "stock"].edge_index = torch.from_numpy(stock_to_sector[[1, 0]]).long()
        hd["sector", "in", "market"].edge_index = torch.from_numpy(sector_to_market).long()
        hd["market", "contains", "sector"].edge_index = torch.from_numpy(sector_to_market[[1, 0]]).long()
        return hd

    # ── Internals ──────────────────────────────────────────────────────

    def _init_edge_builders(self) -> Dict[str, Any]:
        builders: Dict[str, Any] = {}
        edges_cfg = self.cfg.get("edges", {}) or {}
        for edge_type in self.edge_types:
            if edge_type == "correlation":
                corr_cfg = dict(edges_cfg.get("correlation", {}) or {})
                builders["correlation"] = CorrelationEdgeBuilder(
                    window=int(corr_cfg.get("window", 30)),
                    threshold=corr_cfg.get("threshold", 0.5),
                    top_k=corr_cfg.get("top_k"),
                    multi_windows=corr_cfg.get("multi_windows"),
                    inverse_vol_weight=bool(corr_cfg.get("inverse_vol_weight", False)),
                )
            elif edge_type == "sector":
                builders["sector"] = SectorEdgeBuilder(self.sector_map)
            elif edge_type == "fundamental":
                fund_cfg = dict(edges_cfg.get("fundamental", {}) or {})
                builders["fundamental"] = FundamentalEdgeBuilder(
                    threshold=float(fund_cfg.get("threshold", 0.7)),
                )
            elif edge_type == "attention":
                # Learned attention edges are constructed by the GNN itself;
                # no static builder at this layer.
                continue
            else:
                log.warning("Unknown edge_type in config: {}", edge_type)
        return builders

    def _build_edge_type(
        self,
        edge_type: str,
        builder: Any,
        pred_date: pd.Timestamp,
        universe: Sequence[str],
        fundamental_vectors: Optional[pd.DataFrame],
    ) -> EdgeSpec:
        if edge_type == "correlation":
            if self.returns_wide is None:
                raise ValueError(
                    "Correlation edges requested but `returns_wide` not supplied "
                    "to GraphBuilder."
                )
            return builder.build(self.returns_wide, pred_date, universe)
        if edge_type == "sector":
            return builder.build(universe)
        if edge_type == "fundamental":
            if fundamental_vectors is None:
                return EdgeSpec(
                    edge_index=np.zeros((2, 0), dtype=np.int64),
                    edge_weight=np.zeros((0,),  dtype=np.float32),
                    num_nodes=len(universe),
                )
            return builder.build(fundamental_vectors, universe)
        raise ValueError(f"Unhandled edge_type {edge_type!r}")

    @staticmethod
    def _assemble_node_features(
        stock_features: np.ndarray,
        hierarchy: Optional[HierarchicalSpec],
        edges: Dict[str, EdgeSpec],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extend stock features with zero-initialised super-node slots."""
        n_stocks, f = stock_features.shape
        stock_mask = np.linalg.norm(stock_features, axis=1) > 0

        if hierarchy is None:
            return stock_features.astype(np.float32), stock_mask

        n_sectors = hierarchy.n_sector_nodes
        super_feats = np.zeros((n_sectors + 1, f), dtype=np.float32)
        # Sector init: mean of constituent stocks that are present.
        for s_idx in range(n_sectors):
            members = np.where((hierarchy.sector_indices == s_idx) & stock_mask)[0]
            if members.size:
                super_feats[s_idx] = stock_features[members].mean(axis=0)
        # Market init: mean of populated sector slots.
        pop_sectors = np.where(np.linalg.norm(super_feats[:n_sectors], axis=1) > 0)[0]
        if pop_sectors.size:
            super_feats[n_sectors] = super_feats[pop_sectors].mean(axis=0)

        full = np.concatenate([stock_features.astype(np.float32), super_feats], axis=0)
        # All super-nodes are always "present" — hierarchy is static.
        super_mask = np.ones(n_sectors + 1, dtype=bool)
        node_mask = np.concatenate([stock_mask, super_mask], axis=0)
        return full, node_mask

    @staticmethod
    def _default_sector_indices(universe: Sequence[str]) -> np.ndarray:
        return np.full(len(universe), -1, dtype=np.int64)


def build_returns_wide(price_frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Convenience passthrough to the prepared log-returns matrix."""
    return prepare_log_returns(dict(price_frames))
