"""Phase 2 graph tests. numpy-only core; PyG conversion gated behind importorskip."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pytest

from constellation_quant.graph import (
    CorrelationEdgeBuilder,
    FundamentalEdgeBuilder,
    GICS_SECTORS,
    GraphBuilder,
    HierarchyBuilder,
    SectorEdgeBuilder,
    prepare_log_returns,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def returns_wide() -> pd.DataFrame:
    """60 trading days of log returns for 6 tickers."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-02", periods=60)
    n = 6
    # Build three pairs of highly correlated series plus noise.
    common = rng.normal(0, 0.01, size=len(dates))
    cols = {}
    for i in range(n):
        group = i // 2
        cols[f"T{i:03d}"] = (
            common if group == 0 else 0.0
        ) + rng.normal(0, 0.005, size=len(dates))
    return pd.DataFrame(cols, index=dates)


@pytest.fixture
def sector_map():
    return {
        "T000": "Financials", "T001": "Financials",
        "T002": "Information Technology", "T003": "Information Technology",
        "T004": "Health Care", "T005": "Health Care",
    }


# ── Correlation edges ──────────────────────────────────────────────────────


def test_correlation_edges_threshold_filters(returns_wide):
    builder = CorrelationEdgeBuilder(window=30, threshold=0.5)
    spec = builder.build(returns_wide, returns_wide.index[-1], returns_wide.columns)
    # With 6 nodes and threshold 0.5, edges are symmetric / directed pairs.
    assert spec.edge_index.shape[0] == 2
    # No self-edges.
    assert not np.any(spec.edge_index[0] == spec.edge_index[1])


def test_correlation_edges_density_changes_with_threshold(returns_wide):
    loose = CorrelationEdgeBuilder(window=30, threshold=0.1).build(
        returns_wide, returns_wide.index[-1], returns_wide.columns,
    )
    tight = CorrelationEdgeBuilder(window=30, threshold=0.9).build(
        returns_wide, returns_wide.index[-1], returns_wide.columns,
    )
    assert len(loose) >= len(tight)


def test_correlation_edges_top_k(returns_wide):
    builder = CorrelationEdgeBuilder(window=30, threshold=None, top_k=2)
    spec = builder.build(returns_wide, returns_wide.index[-1], returns_wide.columns)
    # With top_k=2 and 6 nodes, at most 2 edges per source node.
    n_nodes = len(returns_wide.columns)
    for src in range(n_nodes):
        assert np.sum(spec.edge_index[0] == src) <= 2


def test_correlation_returns_empty_without_enough_history():
    rw = pd.DataFrame(
        np.random.randn(10, 3),
        index=pd.bdate_range("2020-01-02", periods=10),
        columns=["A", "B", "C"],
    )
    builder = CorrelationEdgeBuilder(window=30)
    spec = builder.build(rw, rw.index[-1], rw.columns)
    assert spec.edge_index.shape == (2, 0)


def test_correlation_maps_to_universe_indices(returns_wide):
    """Edge indices must refer to universe positions, not returns_wide cols."""
    universe = list(returns_wide.columns)[::-1]  # reverse ordering
    builder = CorrelationEdgeBuilder(window=30, threshold=0.2)
    spec = builder.build(returns_wide, returns_wide.index[-1], universe)
    for idx in spec.edge_index.flatten():
        assert 0 <= idx < len(universe)


def test_correlation_multi_window_min_kills_spurious_edges():
    """A correlation that holds in 10d but not 90d should drop near zero
    under multi-window min mode."""
    # Build a returns frame: A & B are perfectly correlated for last 10 days,
    # uncorrelated noise for the prior 80 days.
    rng = np.random.default_rng(0)
    n = 100
    dates = pd.bdate_range("2020-01-02", periods=n)
    a_long = rng.normal(size=n)
    b_long = rng.normal(size=n)
    # Last 10 days: B = A (perfect correlation)
    b_long[-10:] = a_long[-10:]
    df = pd.DataFrame({"A": a_long, "B": b_long}, index=dates)

    # Single 10-day window: corr(A, B) ≈ 1.0
    single = CorrelationEdgeBuilder(window=10, threshold=0.5)
    spec_s = single.build(df, dates[-1], ["A", "B"])
    assert spec_s.edge_weight.size > 0     # high-corr edge survives

    # Multi-window 10/30/90: 30d & 90d correlations are near zero → min ≈ 0
    multi = CorrelationEdgeBuilder(
        window=30, threshold=0.5, multi_windows=[10, 30, 90],
    )
    spec_m = multi.build(df, dates[-1], ["A", "B"])
    # Robust corr should be much smaller than the spurious 10d-only one.
    if spec_m.edge_weight.size > 0:
        assert spec_m.edge_weight.max() < spec_s.edge_weight.max()


def test_correlation_inverse_vol_downweights_unstable_pairs():
    """High-volatility stocks should have their edges scaled down."""
    rng = np.random.default_rng(1)
    n = 100
    dates = pd.bdate_range("2020-01-02", periods=n)
    # Two pairs of correlated stocks: stable pair (low vol) + unstable pair (high vol).
    base_low  = rng.normal(scale=0.005, size=n)
    base_high = rng.normal(scale=0.05,  size=n)
    df = pd.DataFrame({
        "STABLE_A": base_low + rng.normal(scale=0.001, size=n),
        "STABLE_B": base_low + rng.normal(scale=0.001, size=n),
        "UNSTABLE_A": base_high + rng.normal(scale=0.005, size=n),
        "UNSTABLE_B": base_high + rng.normal(scale=0.005, size=n),
    }, index=dates)

    universe = ["STABLE_A", "STABLE_B", "UNSTABLE_A", "UNSTABLE_B"]
    plain = CorrelationEdgeBuilder(window=60, threshold=0.0)
    weighted = CorrelationEdgeBuilder(window=60, threshold=0.0,
                                       inverse_vol_weight=True)

    spec_plain = plain.build(df, dates[-1], universe)
    spec_w     = weighted.build(df, dates[-1], universe)

    # Find the unstable pair edge in both specs.
    def edge_weight_between(spec, i, j):
        for k in range(spec.edge_weight.size):
            if {spec.edge_index[0, k], spec.edge_index[1, k]} == {i, j}:
                return float(spec.edge_weight[k])
        return 0.0

    unstable_w_plain = edge_weight_between(spec_plain, 2, 3)   # UNSTABLE pair
    unstable_w_scaled = edge_weight_between(spec_w, 2, 3)
    # Inverse-vol scaling MUST reduce the unstable edge weight.
    assert unstable_w_scaled <= unstable_w_plain
    # And the reduction should be meaningful (>=20%).
    assert unstable_w_scaled < 0.8 * unstable_w_plain


def test_prepare_log_returns_wide_format():
    frames = {
        "A": pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=10),
            "adj_close": np.arange(1, 11, dtype=float),
        }),
        "B": pd.DataFrame({
            "date": pd.bdate_range("2020-01-02", periods=10),
            "adj_close": np.arange(10, 0, -1, dtype=float),
        }),
    }
    wide = prepare_log_returns(frames)
    assert set(wide.columns) == {"A", "B"}
    assert len(wide) == 10
    assert wide.iloc[0].isna().all()  # first row is NaN (diff)


# ── Sector edges ───────────────────────────────────────────────────────────


def test_sector_edges_within_cluster_only(sector_map):
    builder = SectorEdgeBuilder(sector_map)
    universe = list(sector_map.keys())
    spec = builder.build(universe)
    src, dst = spec.edge_index
    # Every edge must join two stocks of the same sector.
    tickers_by_idx = {i: universe[i] for i in range(len(universe))}
    for s, d in zip(src, dst):
        assert sector_map[tickers_by_idx[int(s)]] == sector_map[tickers_by_idx[int(d)]]


def test_sector_edges_count_matches_combinatorics(sector_map):
    """Each sector of size k contributes k*(k-1) directed edges (no self)."""
    builder = SectorEdgeBuilder(sector_map)
    spec = builder.build(list(sector_map.keys()))
    # 3 sectors of 2 tickers each → 3 * 2 * 1 = 6 directed edges
    assert spec.edge_index.shape[1] == 6


def test_sector_edges_no_self_edges(sector_map):
    spec = SectorEdgeBuilder(sector_map).build(list(sector_map.keys()))
    assert not np.any(spec.edge_index[0] == spec.edge_index[1])


def test_sector_edges_ignore_unknown_tickers(sector_map):
    spec = SectorEdgeBuilder(sector_map).build(list(sector_map.keys()) + ["UNKNOWN"])
    # UNKNOWN has no sector, so no new edges involve its index.
    unknown_idx = 6  # 7th position
    assert unknown_idx not in spec.edge_index.flatten()


# ── Fundamental edges ──────────────────────────────────────────────────────


def test_fundamental_edges_cosine_similarity_threshold():
    vectors = pd.DataFrame(
        {
            "pe":  [10.0, 10.5, 25.0, 25.2],
            "pb":  [ 2.0,  2.1,  5.0,  5.1],
            "de":  [ 0.5,  0.5,  1.2,  1.3],
        },
        index=["A", "B", "C", "D"],
    )
    builder = FundamentalEdgeBuilder(threshold=0.99)
    spec = builder.build(vectors, ["A", "B", "C", "D"])
    # A-B and C-D are near-identical (cosine ≈ 1); A-C and B-D are not.
    assert spec.edge_index.shape[1] > 0
    # Every edge weight exceeds threshold (cosine sim).
    assert (spec.edge_weight > 0.99).all()


def test_fundamental_edges_empty_on_missing_data():
    vectors = pd.DataFrame({"pe": []}, index=pd.Index([], dtype=object))
    spec = FundamentalEdgeBuilder().build(vectors, ["A", "B"])
    assert spec.edge_index.shape == (2, 0)


def test_fundamental_edges_top_k():
    # Random vectors — each node should end up with exactly k neighbours.
    rng = np.random.default_rng(0)
    vectors = pd.DataFrame(
        rng.normal(size=(10, 4)),
        index=[f"T{i}" for i in range(10)],
        columns=["a", "b", "c", "d"],
    )
    spec = FundamentalEdgeBuilder(top_k=3).build(vectors, list(vectors.index))
    for i in range(10):
        assert np.sum(spec.edge_index[0] == i) <= 3


# ── Hierarchy ──────────────────────────────────────────────────────────────


def test_hierarchy_node_counts(sector_map):
    builder = HierarchyBuilder(sector_map)
    spec = builder.build(list(sector_map.keys()))
    assert spec.n_stock_nodes == len(sector_map)
    assert spec.n_sector_nodes == len(GICS_SECTORS)
    assert spec.total_nodes == len(sector_map) + len(GICS_SECTORS) + 1
    assert spec.market_node_index == len(sector_map) + len(GICS_SECTORS)


def test_hierarchy_sector_edges_connect_stocks_to_correct_sector(sector_map):
    builder = HierarchyBuilder(sector_map)
    tickers = list(sector_map.keys())
    spec = builder.build(tickers)
    # Financials is sector 4 in GICS_SECTORS ordering.
    fin_sector_idx = GICS_SECTORS.index("Financials")
    # T000 / T001 are financials → they should connect to the Financials super-node.
    stock_src = spec.stock_to_sector[0]
    sector_dst = spec.stock_to_sector[1]
    for src_idx, dst_idx in zip(stock_src, sector_dst):
        if tickers[int(src_idx)] in {"T000", "T001"}:
            assert int(dst_idx) == spec.n_stock_nodes + fin_sector_idx


def test_hierarchy_market_connects_to_all_sectors(sector_map):
    spec = HierarchyBuilder(sector_map).build(list(sector_map.keys()))
    # Every sector should connect to the market node exactly once.
    assert spec.sector_to_market.shape[1] == len(GICS_SECTORS)
    assert np.all(spec.sector_to_market[1] == spec.market_node_index)


# ── GraphBuilder orchestration ─────────────────────────────────────────────


@pytest.fixture
def model_cfg_minimal():
    return {
        "graph": {"enabled": True, "gnn_name": "rgat",
                   "edge_types": ["sector"]},
        "edges": {},
        "hierarchy": {"enabled": False},
    }


def test_graph_builder_sector_only(sector_map, model_cfg_minimal):
    universe = list(sector_map.keys())
    builder = GraphBuilder(model_cfg=model_cfg_minimal, sector_map=sector_map)

    node_features = np.random.RandomState(0).randn(len(universe), 8).astype(np.float32)
    built = builder.build(
        pred_date=pd.Timestamp("2020-06-01"),
        universe_tickers=universe,
        node_features=node_features,
    )
    assert "sector" in built.edges
    assert built.node_features.shape == (len(universe), 8)
    assert built.hierarchy is None


def test_graph_builder_with_hierarchy(sector_map, returns_wide):
    cfg = {
        "graph": {"enabled": True, "gnn_name": "rgat",
                   "edge_types": ["correlation", "sector"]},
        "edges": {"correlation": {"window": 30, "threshold": 0.1}},
        "hierarchy": {"enabled": True},
    }
    universe = list(sector_map.keys())
    builder = GraphBuilder(
        model_cfg=cfg,
        sector_map=sector_map,
        returns_wide=returns_wide,
    )
    node_features = np.random.RandomState(1).randn(len(universe), 4).astype(np.float32)
    built = builder.build(
        pred_date=returns_wide.index[-1],
        universe_tickers=universe,
        node_features=node_features,
    )
    assert built.hierarchy is not None
    # Node feature matrix grows by N_sectors + 1.
    expected_rows = len(universe) + len(GICS_SECTORS) + 1
    assert built.node_features.shape == (expected_rows, 4)
    assert "correlation" in built.edges
    assert "sector" in built.edges


def test_graph_builder_dynamic_membership(sector_map, returns_wide):
    """Universe shrinks between two dates; GraphBuilder adapts without error."""
    cfg = {
        "graph": {"enabled": True, "edge_types": ["sector"]},
        "edges": {},
        "hierarchy": {"enabled": False},
    }
    builder = GraphBuilder(cfg, sector_map=sector_map, returns_wide=returns_wide)
    small = builder.build(
        pred_date=returns_wide.index[-1],
        universe_tickers=["T000", "T001", "T002"],
        node_features=np.zeros((3, 4), dtype=np.float32),
    )
    big = builder.build(
        pred_date=returns_wide.index[-1],
        universe_tickers=list(sector_map.keys()),
        node_features=np.zeros((6, 4), dtype=np.float32),
    )
    assert small.node_features.shape == (3, 4)
    assert big.node_features.shape == (6, 4)


def test_graph_builder_to_pyg_roundtrip(sector_map, model_cfg_minimal):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    universe = list(sector_map.keys())
    builder = GraphBuilder(model_cfg=model_cfg_minimal, sector_map=sector_map)
    built = builder.build(
        pred_date=pd.Timestamp("2020-06-01"),
        universe_tickers=universe,
        node_features=np.random.RandomState(0).randn(len(universe), 8).astype(np.float32),
    )
    data = builder.to_pyg(built)
    # Homogeneous → one Data object with edge_index_sector attribute.
    assert hasattr(data, "edge_index_sector")
    assert isinstance(data.edge_index_sector, torch.Tensor)
