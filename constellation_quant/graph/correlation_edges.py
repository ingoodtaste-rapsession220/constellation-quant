"""Rolling-correlation edges — the primary dynamic edge type.

For each trading day `t`, compute the pairwise correlation of log returns
over the trailing `window` days (default 30). Stocks `(i, j)` are connected
when `|ρ_ij| > threshold` (or when `j` is among stock `i`'s top-K most
correlated peers in top-K mode). Edge weight equals the absolute correlation.

Output is in **PyTorch Geometric sparse format**:
    edge_index  : np.ndarray[int64], shape (2, E)   — (source_idx, target_idx)
    edge_weight : np.ndarray[float32], shape (E,)

Both arrays are numpy — the graph builder wraps them into torch tensors when
assembling the final PyG Data object. Keeping this layer numpy-only makes it
testable and cache-friendly without dragging torch into the test path.

Dynamic membership: `universe_tickers` is the list of tickers at date `t` in
their canonical ordering. Nodes not in the universe simply don't appear in
the edge list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from constellation_quant.utils import get_logger

log = get_logger(__name__)


@dataclass
class EdgeSpec:
    """Bundle of edge tensors for a single date."""
    edge_index: np.ndarray   # (2, E) int64
    edge_weight: np.ndarray  # (E,)  float32
    num_nodes: int

    def __len__(self) -> int:
        return int(self.edge_weight.size)


class CorrelationEdgeBuilder:
    """Rolling Pearson correlation → sparse edges.

    Args:
        window: Rolling window length in trading days. Default 30.
        threshold: |ρ| edge cutoff. Ignored when `top_k` is set.
        top_k: If not None, each node keeps its top-K neighbours by |ρ|
            regardless of threshold.
        use_absolute: If True, edge weight = |ρ|; else signed ρ.
        include_self_loops: Keep self edges (always ρ=1). Default False.
    """

    def __init__(
        self,
        window: int = 30,
        threshold: Optional[float] = 0.5,
        top_k: Optional[int] = None,
        use_absolute: bool = True,
        include_self_loops: bool = False,
        multi_windows: Optional[Sequence[int]] = None,
        inverse_vol_weight: bool = False,
    ):
        if window < 2:
            raise ValueError(f"window must be >= 2, got {window}")
        if top_k is not None and top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if threshold is not None and not -1.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [-1, 1], got {threshold}")
        if multi_windows is not None:
            multi_windows = sorted(int(w) for w in multi_windows)
            if any(w < 2 for w in multi_windows):
                raise ValueError(f"multi_windows entries must be >= 2, got {multi_windows}")
        self.window = window
        self.threshold = threshold
        self.top_k = top_k
        self.use_absolute = use_absolute
        self.include_self_loops = include_self_loops
        # multi_windows: instead of a single rolling window, compute correlation
        # at multiple lookback lengths and take the elementwise MIN of |ρ|
        # across them. Sign is taken from the LONGEST window (the most reliable
        # for direction). Kills spurious short-window correlations that don't
        # hold over longer horizons. The longest entry overrides `window` for
        # data-availability checks.
        self.multi_windows = multi_windows
        # inverse_vol_weight: scale each edge by sqrt(stab_i × stab_j) where
        # stab = clip(median_vol / vol, 0, 1). Stocks with above-median vol
        # get their edges down-weighted. Equivalent to "trust correlations
        # from stable stocks more than from noisy ones".
        self.inverse_vol_weight = bool(inverse_vol_weight)

    # ── Public API ─────────────────────────────────────────────────────

    def build(
        self,
        returns: pd.DataFrame,
        pred_date: pd.Timestamp,
        universe_tickers: Sequence[str],
    ) -> EdgeSpec:
        """Build the edge set for a single prediction date.

        Args:
            returns: Wide DataFrame (rows=date, cols=ticker) of log returns.
            pred_date: Date `t` — rolling window spans `[t - W + 1, t]`.
            universe_tickers: Canonical ticker ordering for the date (usually
                the S&P 500 membership on `pred_date`). Node indices in the
                returned `edge_index` refer to positions in this list.

        Returns:
            EdgeSpec with numpy arrays. `edge_index` is empty (shape (2, 0))
            if no rolling window exists yet or no edges cross the threshold.
        """
        # Use the longest required window so multi-window mode always has data.
        windows_used = self.multi_windows if self.multi_windows else [self.window]
        longest = max(windows_used)
        window = returns.loc[returns.index <= pred_date].tail(longest)
        if len(window) < longest:
            return self._empty_spec(len(universe_tickers))

        members_in_data = [t for t in universe_tickers if t in window.columns]
        if len(members_in_data) < 2:
            return self._empty_spec(len(universe_tickers))

        submatrix = window[members_in_data].dropna(axis=1, how="any")
        members = list(submatrix.columns)
        if len(members) < 2:
            return self._empty_spec(len(universe_tickers))

        if self.multi_windows is not None:
            # Compute corr at each lookback; take elementwise min of |ρ|.
            # Sign comes from the longest window — most reliable for direction.
            abs_corrs: List[np.ndarray] = []
            longest_corr: Optional[np.ndarray] = None
            for w in windows_used:
                sub_w = submatrix.tail(w)
                if len(sub_w) < w:
                    continue
                c = sub_w.corr().to_numpy(copy=False)
                abs_corrs.append(np.abs(c))
                longest_corr = c            # last iteration = longest window
            if not abs_corrs or longest_corr is None:
                return self._empty_spec(len(universe_tickers))
            min_abs = np.minimum.reduce(abs_corrs)
            corr = np.sign(longest_corr) * min_abs
        else:
            corr = submatrix.corr().to_numpy(copy=True)
        np.fill_diagonal(corr, 0.0)  # no self edges pre-filter

        if self.inverse_vol_weight:
            # Per-stock realised vol on the longest window. Down-weight edges
            # whose endpoints are unstable (vol > median). Stable pairs keep
            # full strength; unstable pairs shrink toward zero.
            vols = submatrix.std(axis=0, ddof=0).to_numpy(dtype=np.float64)
            vols = np.where(vols < 1e-8, 1e-8, vols)
            median_vol = float(np.median(vols))
            stability = np.clip(median_vol / vols, 0.0, 1.0)
            stab_product = np.sqrt(np.outer(stability, stability))
            corr = corr * stab_product

        weights = np.abs(corr) if self.use_absolute else corr.copy()

        if self.top_k is not None:
            edges = self._top_k_edges(weights, self.top_k)
        else:
            edges = self._threshold_edges(weights, self.threshold)
        if edges.size == 0:
            return self._empty_spec(len(universe_tickers))

        src_local, dst_local = edges
        weight_values = weights[src_local, dst_local].astype(np.float32)

        # Remap local indices (within `members`) to the universe ordering.
        universe_lookup = {t: i for i, t in enumerate(universe_tickers)}
        src_global = np.array([universe_lookup[members[i]] for i in src_local], dtype=np.int64)
        dst_global = np.array([universe_lookup[members[i]] for i in dst_local], dtype=np.int64)
        edge_index = np.stack([src_global, dst_global], axis=0)

        if self.include_self_loops:
            n = len(universe_tickers)
            loops = np.arange(n, dtype=np.int64)
            edge_index = np.concatenate([edge_index, np.stack([loops, loops], axis=0)], axis=1)
            weight_values = np.concatenate([weight_values, np.ones(n, dtype=np.float32)])

        return EdgeSpec(
            edge_index=edge_index,
            edge_weight=weight_values,
            num_nodes=len(universe_tickers),
        )

    def build_many(
        self,
        returns: pd.DataFrame,
        dates: Sequence[pd.Timestamp],
        universe_fn,
    ) -> Dict[pd.Timestamp, EdgeSpec]:
        """Batch build. `universe_fn(date) -> list[ticker]`."""
        out: Dict[pd.Timestamp, EdgeSpec] = {}
        for d in dates:
            out[d] = self.build(returns, d, universe_fn(d))
        return out

    # ── Internals ──────────────────────────────────────────────────────

    @staticmethod
    def _threshold_edges(weights: np.ndarray, threshold: float) -> np.ndarray:
        src, dst = np.where(weights > threshold)
        return np.stack([src, dst], axis=0) if src.size else np.zeros((2, 0), dtype=np.int64)

    @staticmethod
    def _top_k_edges(weights: np.ndarray, k: int) -> np.ndarray:
        n = weights.shape[0]
        k = min(k, n - 1)
        if k <= 0:
            return np.zeros((2, 0), dtype=np.int64)
        # Top-K neighbours per row (excluding self, enforced via diagonal zero).
        topk_idx = np.argpartition(-weights, kth=k - 1, axis=1)[:, :k]
        src = np.repeat(np.arange(n), k).astype(np.int64)
        dst = topk_idx.reshape(-1).astype(np.int64)
        mask = src != dst
        return np.stack([src[mask], dst[mask]], axis=0)

    @staticmethod
    def _empty_spec(n: int) -> EdgeSpec:
        return EdgeSpec(
            edge_index=np.zeros((2, 0), dtype=np.int64),
            edge_weight=np.zeros((0,),  dtype=np.float32),
            num_nodes=n,
        )


def prepare_log_returns(
    price_frames: "Dict[str, pd.DataFrame]",
) -> pd.DataFrame:
    """Convert per-ticker OHLCV frames into a wide log-returns matrix.

    Rows = dates (union of all tickers), columns = tickers. Missing cells
    are NaN and get dropped by the correlation builder per date.
    """
    cols: List[pd.Series] = []
    for ticker, df in price_frames.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.set_index("date")
        px = frame["adj_close"].astype(float).sort_index()
        cols.append(np.log(px / px.shift(1)).rename(ticker))
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()
