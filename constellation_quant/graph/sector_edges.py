"""Static sector-based edges — every stock connected to every other stock in
its same GICS sector. Used as the baseline graph in ablation Model B.

Edges are **symmetric and unweighted** (weight=1.0). Self-loops excluded by
default. The edge index is computed per date because membership changes,
but the underlying sector assignments are static.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from constellation_quant.graph.correlation_edges import EdgeSpec


class SectorEdgeBuilder:
    """Build static sector edges given a ticker->sector mapping.

    Args:
        sector_map: ticker -> GICS sector label. Tickers missing from this
            map get no edges (treated as an "unknown" singleton cluster).
        include_self_loops: Default False.
    """

    def __init__(
        self,
        sector_map: Mapping[str, str],
        include_self_loops: bool = False,
    ):
        self.sector_map = {k.upper(): v for k, v in sector_map.items() if v}
        self.include_self_loops = include_self_loops

    def build(self, universe_tickers: Sequence[str]) -> EdgeSpec:
        tickers = [t.upper() for t in universe_tickers]
        n = len(tickers)
        if n < 2:
            return _empty_spec(n)

        # Bucket indices by sector.
        buckets: dict[str, list[int]] = {}
        for idx, ticker in enumerate(tickers):
            sector = self.sector_map.get(ticker)
            if sector is None:
                continue
            buckets.setdefault(sector, []).append(idx)

        src_list, dst_list = [], []
        for indices in buckets.values():
            if len(indices) < 2:
                continue
            arr = np.array(indices, dtype=np.int64)
            src = np.repeat(arr, len(arr))
            dst = np.tile(arr, len(arr))
            mask = src != dst  # drop self edges
            src_list.append(src[mask])
            dst_list.append(dst[mask])

        if not src_list:
            return _empty_spec(n)
        src_all = np.concatenate(src_list)
        dst_all = np.concatenate(dst_list)

        if self.include_self_loops:
            loops = np.arange(n, dtype=np.int64)
            src_all = np.concatenate([src_all, loops])
            dst_all = np.concatenate([dst_all, loops])

        edge_index = np.stack([src_all, dst_all], axis=0)
        weights = np.ones(edge_index.shape[1], dtype=np.float32)
        return EdgeSpec(edge_index=edge_index, edge_weight=weights, num_nodes=n)


def _empty_spec(n: int) -> EdgeSpec:
    return EdgeSpec(
        edge_index=np.zeros((2, 0), dtype=np.int64),
        edge_weight=np.zeros((0,),  dtype=np.float32),
        num_nodes=n,
    )
