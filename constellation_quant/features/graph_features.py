"""Graph-derived features — each stock's position in the network becomes
a per-stock feature vector.

Computed from the correlation-edge graph (dense enough to yield meaningful
centralities). Four features per stock per date:

    - degree                 : neighbour count in the correlation graph
    - avg_neighbor_return_5d : mean 5-day log return of connected neighbours
    - sector_momentum_5d     : mean 5-day log return of same-sector peers
    - betweenness_centrality : bridge-score between clusters (sampled)

Betweenness is the heavy one — full exact computation is O(V·E). For
universes of ~500 stocks we use `networkx.betweenness_centrality(k=50)`
with random sampling, which is >10× faster and within a few percent of the
exact value.

Output: dict[ticker, DataFrame] matching the shape of the other feature
builders, keyed by ticker and indexed by date.
"""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from constellation_quant.graph.correlation_edges import CorrelationEdgeBuilder, EdgeSpec
from constellation_quant.utils import get_logger

log = get_logger(__name__)


class GraphFeatures:
    """Compute per-stock graph-position features from rolling correlation graphs.

    Args:
        edge_builder: A CorrelationEdgeBuilder — reused across dates.
        betweenness_sample_k: Sample size for approximate betweenness. Set to
            None for exact computation (slow on >200 nodes).
        return_window: Lookback length for the "neighbour return" aggregate.
    """

    def __init__(
        self,
        edge_builder: Optional[CorrelationEdgeBuilder] = None,
        betweenness_sample_k: Optional[int] = 50,
        return_window: int = 5,
    ):
        self.edge_builder = edge_builder or CorrelationEdgeBuilder()
        self.betweenness_k = betweenness_sample_k
        self.return_window = return_window

    def compute(
        self,
        returns_wide: pd.DataFrame,
        dates: Iterable[pd.Timestamp],
        universe_fn,
        sector_map: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute graph features for each (date, ticker).

        Args:
            returns_wide: (T, N) log-return matrix.
            dates: iterable of prediction dates.
            universe_fn: `date -> list[ticker]` returning the S&P 500 members
                on that date.
            sector_map: Optional ticker->sector for sector_momentum feature.

        Returns:
            Dict[ticker, DataFrame]. Each row is a date; columns are the
            four graph-derived features.
        """
        sector_map = {k.upper(): v for k, v in (sector_map or {}).items()}
        rows_by_ticker: Dict[str, Dict[pd.Timestamp, Dict[str, float]]] = {}

        for d in dates:
            universe = [t.upper() for t in universe_fn(d)]
            spec = self.edge_builder.build(returns_wide, d, universe)
            neighbour_lookup = _edge_neighbours(spec)

            avg_returns_by_member = self._recent_avg_return(returns_wide, d, universe)
            betweenness = self._betweenness(spec)
            sector_mom = self._sector_momentum(avg_returns_by_member, universe, sector_map)

            for i, ticker in enumerate(universe):
                neighbours = neighbour_lookup.get(i, np.empty(0, dtype=np.int64))
                degree = int(neighbours.size)
                if degree:
                    avg_nbr_ret = float(
                        np.nanmean([avg_returns_by_member[j] for j in neighbours])
                    )
                else:
                    avg_nbr_ret = 0.0
                rows_by_ticker.setdefault(ticker, {})[pd.Timestamp(d)] = {
                    "degree":                 float(degree),
                    "avg_neighbor_return":    avg_nbr_ret,
                    "sector_momentum":        float(sector_mom.get(ticker, 0.0)),
                    "betweenness_centrality": float(betweenness.get(i, 0.0)),
                }

        return {
            ticker: pd.DataFrame.from_dict(rows, orient="index").sort_index()
            for ticker, rows in rows_by_ticker.items()
        }

    # ── Component pieces ───────────────────────────────────────────────

    def _recent_avg_return(
        self,
        returns_wide: pd.DataFrame,
        pred_date: pd.Timestamp,
        universe: Sequence[str],
    ) -> np.ndarray:
        """Per-stock mean log return over the last `return_window` days."""
        window = returns_wide.loc[returns_wide.index <= pred_date].tail(self.return_window)
        if window.empty:
            return np.zeros(len(universe), dtype=np.float32)
        per_ticker = window.mean(axis=0, skipna=True)
        return np.array(
            [float(per_ticker.get(t, 0.0)) for t in universe],
            dtype=np.float32,
        )

    def _betweenness(self, spec: EdgeSpec) -> Dict[int, float]:
        """Approximate betweenness centrality via networkx (sampled)."""
        if spec.edge_index.size == 0:
            return {}
        try:
            import networkx as nx
        except ImportError:
            log.warning("networkx not available; betweenness will be zero-filled.")
            return {}

        g = nx.Graph()
        g.add_nodes_from(range(spec.num_nodes))
        # Undirected: symmetrise the edge list.
        for src, dst in zip(spec.edge_index[0], spec.edge_index[1]):
            g.add_edge(int(src), int(dst))
        k = self.betweenness_k
        if k is not None:
            k = min(k, max(g.number_of_nodes(), 1))
        return nx.betweenness_centrality(g, k=k, normalized=True)

    @staticmethod
    def _sector_momentum(
        recent_returns: np.ndarray,
        universe: Sequence[str],
        sector_map: Mapping[str, str],
    ) -> Dict[str, float]:
        """Mean recent return of each stock's same-sector peers (excluding itself)."""
        by_sector: Dict[str, list] = {}
        for t, r in zip(universe, recent_returns):
            sector = sector_map.get(t)
            if sector is None:
                continue
            by_sector.setdefault(sector, []).append(float(r))

        sector_means = {s: float(np.mean(vs)) for s, vs in by_sector.items() if vs}
        out: Dict[str, float] = {}
        for t, r in zip(universe, recent_returns):
            sector = sector_map.get(t)
            if sector is None or sector not in sector_means:
                out[t] = 0.0
                continue
            peers = by_sector.get(sector, [])
            if len(peers) <= 1:
                out[t] = 0.0
            else:
                # Remove self from mean: (sum - r) / (n - 1)
                peer_sum = sum(peers) - float(r)
                out[t] = peer_sum / (len(peers) - 1)
        return out


def _edge_neighbours(spec: EdgeSpec) -> Dict[int, np.ndarray]:
    """Group a (2, E) edge_index into src_idx -> neighbour_indices."""
    if spec.edge_index.size == 0:
        return {}
    src, dst = spec.edge_index
    order = np.argsort(src, kind="stable")
    src_sorted = src[order]
    dst_sorted = dst[order]
    unique, starts = np.unique(src_sorted, return_index=True)
    ends = np.append(starts[1:], len(src_sorted))
    return {
        int(u): dst_sorted[s:e].astype(np.int64)
        for u, s, e in zip(unique, starts, ends)
    }
