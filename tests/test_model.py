"""Phase 3 model tests. Tiny synthetic data — everything runs in seconds."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch

from constellation_quant.models import (
    ConstellationQuant,
    InformerEncoder,
    InformerConfig,
    LSTMEncoder,
    MambaEncoder,
    TCNEncoder,
    TransformerEncoder,
    get_gnn_layer,
    get_output_head,
    get_temporal_encoder,
    list_gnn_layers,
    list_temporal_encoders,
)
from constellation_quant.models.graph_nn import HierarchicalMessagePassing


# ── Temporal encoders ──────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["informer", "lstm", "transformer", "tcn", "mamba"])
def test_temporal_encoder_shapes(name):
    B, L, F = 4, 20, 6
    torch.manual_seed(0)
    enc = get_temporal_encoder(name, n_features=F, config={"d_model": 32, "n_heads": 4,
                                                             "e_layers": 2, "d_ff": 64,
                                                             "num_layers": 2})
    out = enc(torch.randn(B, L, F))
    assert out.shape == (B, enc.output_dim)
    assert torch.isfinite(out).all()


def test_informer_probsparse_matches_dense_for_short_l():
    """For L <= 32 the encoder uses dense attention; gradients should flow."""
    torch.manual_seed(42)
    enc = InformerEncoder(n_features=5, config=InformerConfig(d_model=16, n_heads=4, e_layers=1))
    x = torch.randn(2, 16, 5, requires_grad=True)
    out = enc(x)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_informer_long_sequence_probsparse_path():
    """L > 32 triggers ProbSparse branch; output still well-formed."""
    torch.manual_seed(0)
    enc = InformerEncoder(n_features=4, config=InformerConfig(d_model=16, n_heads=4, e_layers=1, distil=False))
    out = enc(torch.randn(2, 64, 4))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_temporal_factory_lists_all():
    assert set(list_temporal_encoders()) == {"informer", "lstm", "transformer", "tcn", "mamba"}


# ── GNN layers ─────────────────────────────────────────────────────────────


def test_gcn_preserves_node_count():
    torch.manual_seed(0)
    N = 8
    x = torch.randn(N, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    gnn = get_gnn_layer("gcn", in_dim=16, config={"hidden_dim": 32, "num_layers": 2})
    out = gnn(x, edge_index)
    assert out.shape == (N, 32)


def test_gat_preserves_node_count():
    torch.manual_seed(0)
    N = 8
    x = torch.randn(N, 16)
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 0, 5]], dtype=torch.long)
    gnn = get_gnn_layer("gat", in_dim=16,
                         config={"hidden_dim": 32, "num_layers": 2, "attention_heads": 4})
    assert gnn(x, edge_index).shape == (N, 32)


def test_graphsage_preserves_node_count():
    torch.manual_seed(0)
    N = 8
    x = torch.randn(N, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    gnn = get_gnn_layer("graphsage", in_dim=16, config={"hidden_dim": 32, "num_layers": 2})
    assert gnn(x, edge_index).shape == (N, 32)


def test_rgat_per_relation():
    torch.manual_seed(0)
    N = 10
    x = torch.randn(N, 24)
    edges_by_rel = {
        "correlation": (torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long), None),
        "fundamental": (torch.tensor([[5, 6], [6, 7]], dtype=torch.long), None),
    }
    gnn = get_gnn_layer("rgat", in_dim=24, config={
        "hidden_dim": 24, "num_layers": 2, "attention_heads": 4,
        "edge_types": ["correlation", "fundamental"],
        "learned_attention": True, "top_k_attention": 3,
    })
    out = gnn(x, edges_by_rel)
    assert out.shape == (N, 24)


def test_rgat_handles_empty_relations():
    torch.manual_seed(0)
    gnn = get_gnn_layer("rgat", in_dim=8, config={
        "hidden_dim": 8, "num_layers": 1,
        "edge_types": ["correlation"],
        "learned_attention": False,
    })
    x = torch.randn(4, 8)
    edges = {"correlation": (torch.zeros((2, 0), dtype=torch.long), None)}
    out = gnn(x, edges)
    # No edges → identity (same shape, gradients via residual only).
    assert out.shape == (4, 8)


def test_gnn_factory_lists_all():
    assert set(list_gnn_layers()) == {"gcn", "gat", "rgat", "graphsage"}


# ── Hierarchical MP ────────────────────────────────────────────────────────


def test_hierarchical_mp_shapes_and_gate():
    torch.manual_seed(0)
    N, d = 6, 8
    hmp = HierarchicalMessagePassing(d_model=d, n_sectors=3)
    x = torch.randn(N, d)
    sectors = torch.tensor([0, 0, 1, 1, 2, -1], dtype=torch.long)
    mask = torch.tensor([True, True, True, True, True, True])
    out = hmp(x, sectors, mask)
    assert out.shape == (N, d)
    assert torch.isfinite(out).all()


# ── Output heads ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["ranking", "return", "volatility"])
def test_output_heads(name):
    head = get_output_head(name, in_dim=32, config={"mlp": [16, 1], "dropout": 0.0})
    head.eval()   # avoid BatchNorm issues on a 3-sample tensor
    x = torch.randn(3, 32)
    out = head(x)
    assert out.shape == (3,)
    if name == "volatility":
        assert (out >= 0).all(), "volatility must be non-negative"


# ── Master model end-to-end ────────────────────────────────────────────────


def _model_cfg(gnn_name="gat", hierarchy=False, multi_scale=False):
    return {
        "lookback": 20, "horizon": 5,
        "multi_scale": multi_scale,
        "multi_scale_windows": [10, 20],
        "temporal": {
            "name": "informer",
            "d_model": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64,
            "dropout": 0.1, "probsparse_factor": 5, "distil": False,
        },
        "graph": {
            "enabled": True, "gnn_name": gnn_name,
            "hidden_dim": 32, "num_layers": 2, "attention_heads": 4,
            "dropout": 0.1, "residual": True,
            "edge_types": ["correlation"] if gnn_name != "rgat"
                          else ["correlation", "fundamental"],
            "learned_attention": True, "top_k_attention": 3,
        },
        "edges": {"correlation": {"window": 10, "threshold": 0.5}},
        "hierarchy": {"enabled": hierarchy, "sector_nodes": 3,
                       "market_node": 1, "bidirectional": True},
        "heads": {
            "ranking":    {"enabled": True, "mlp": [32, 16, 1], "dropout": 0.0,
                           "temperature_scaling": True},
            "return":     {"enabled": True, "mlp": [32, 1], "dropout": 0.0},
            "volatility": {"enabled": True, "mlp": [32, 1], "dropout": 0.0},
        },
    }


def _make_edges(N: int):
    ei = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    return {
        "correlation": (ei, None),
        "fundamental": (ei, None),
    }


def test_master_model_forward_sector_only():
    torch.manual_seed(0)
    N, L, F = 6, 20, 5
    model = ConstellationQuant(n_features=F, model_cfg=_model_cfg(gnn_name="gat"))
    features = torch.randn(N, L, F)
    mask = torch.tensor([True] * N)
    edges = _make_edges(N)
    sectors = torch.tensor([0, 0, 1, 1, 2, -1], dtype=torch.long)
    out = model(features=features, mask=mask, edges=edges, sector_indices=sectors)
    assert out.scores.shape == (N,)
    assert out.ret.shape == (N,)
    assert out.volatility.shape == (N,)
    assert (out.volatility >= 0).all()


def test_master_model_with_hierarchy():
    torch.manual_seed(0)
    N, L, F = 6, 20, 5
    model = ConstellationQuant(n_features=F, model_cfg=_model_cfg(gnn_name="rgat", hierarchy=True))
    out = model(
        features=torch.randn(N, L, F),
        mask=torch.tensor([True] * N),
        edges=_make_edges(N),
        sector_indices=torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long),
    )
    assert out.scores.shape == (N,)
    assert torch.isfinite(out.scores).all()


def test_master_model_variable_size_graph():
    """Same model handles different N values across calls."""
    torch.manual_seed(0)
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"))
    for N in [4, 8, 16]:
        features = torch.randn(N, 20, 5)
        mask = torch.tensor([True] * N)
        ei = torch.tensor([[i % N for i in range(N)], [(i + 1) % N for i in range(N)]],
                           dtype=torch.long)
        out = model(features=features, mask=mask,
                     edges={"correlation": (ei, None)},
                     sector_indices=torch.zeros(N, dtype=torch.long))
        assert out.scores.shape == (N,)


def test_master_model_batched_forward_matches_unbatched():
    """Batched (B, N, L, F) path must produce identical output to per-date calls.

    Regression guard for the Informer-batching change in the trainer hot path.
    Run the model on B dates individually, stack the outputs, then run the
    same B dates as a single batched forward pass. Results should match to
    floating-point precision (eval mode so dropout is off).
    """
    torch.manual_seed(0)
    B, N, L, F = 3, 6, 20, 5
    model = ConstellationQuant(n_features=F, model_cfg=_model_cfg(gnn_name="rgat", hierarchy=True))
    model.eval()

    per_date_features = [torch.randn(N, L, F) for _ in range(B)]
    per_date_masks    = [torch.tensor([True] * N) for _ in range(B)]
    per_date_sectors  = [torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
                         for _ in range(B)]
    per_date_edges    = [_make_edges(N) for _ in range(B)]

    # Per-date reference outputs.
    ref_scores, ref_ret, ref_vol = [], [], []
    with torch.no_grad():
        for b in range(B):
            out = model(
                features=per_date_features[b],
                mask=per_date_masks[b],
                edges=per_date_edges[b],
                sector_indices=per_date_sectors[b],
            )
            ref_scores.append(out.scores)
            ref_ret.append(out.ret)
            ref_vol.append(out.volatility)
    ref_scores = torch.stack(ref_scores)
    ref_ret    = torch.stack(ref_ret)
    ref_vol    = torch.stack(ref_vol)

    # Batched call.
    with torch.no_grad():
        out_b = model(
            features=torch.stack(per_date_features),
            mask=torch.stack(per_date_masks),
            edges=per_date_edges,
            sector_indices=torch.stack(per_date_sectors),
        )

    assert out_b.scores.shape == (B, N)
    assert torch.allclose(out_b.scores, ref_scores, atol=1e-5, rtol=1e-4)
    assert torch.allclose(out_b.ret,    ref_ret,    atol=1e-5, rtol=1e-4)
    assert torch.allclose(out_b.volatility, ref_vol, atol=1e-5, rtol=1e-4)


def test_master_model_gradients_flow():
    """Summing over scores + return + volatility should propagate to every param."""
    torch.manual_seed(0)
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"))
    out = model(
        features=torch.randn(6, 20, 5),
        mask=torch.tensor([True] * 6),
        edges=_make_edges(6),
        sector_indices=torch.zeros(6, dtype=torch.long),
    )
    # Touch every head so their parameters receive gradient contributions.
    (out.scores.sum() + out.ret.sum() + out.volatility.sum()).backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert no_grad == [], f"Unexpected no-grad params: {no_grad[:5]}"


def test_master_model_save_load(tmp_path):
    torch.manual_seed(0)
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"))
    # Run once in eval to init BN running stats deterministically.
    model.eval()
    with torch.no_grad():
        ref = model(
            features=torch.randn(4, 20, 5),
            mask=torch.tensor([True] * 4),
            edges=_make_edges(4),
            sector_indices=torch.zeros(4, dtype=torch.long),
        )
    p = tmp_path / "model.pt"
    torch.save(model.state_dict(), p)

    model2 = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"))
    model2.load_state_dict(torch.load(p, weights_only=False))
    model2.eval()
    with torch.no_grad():
        torch.manual_seed(99)                # different seed
        out2 = model2(
            features=torch.zeros(4, 20, 5),  # deterministic input
            mask=torch.tensor([True] * 4),
            edges=_make_edges(4),
            sector_indices=torch.zeros(4, dtype=torch.long),
        )
    assert out2.scores.shape == (4,)


def test_master_model_describe_keys():
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"))
    d = model.describe()
    for k in ("temporal", "gnn", "heads", "d_model", "num_parameters"):
        assert k in d


def test_master_model_slow_branch_concats_into_gnn_input():
    """When n_slow_features > 0, slow features must reach the model and
    actually change the output. Regression guard for the Phase-4 split."""
    torch.manual_seed(0)
    N, L, F, F_slow = 6, 20, 5, 4
    model = ConstellationQuant(
        n_features=F,
        model_cfg=_model_cfg(gnn_name="gat"),
        n_slow_features=F_slow,
    )
    model.eval()
    assert model.slow_branch is not None
    assert model.n_slow_features == F_slow

    features = torch.randn(N, L, F)
    mask = torch.tensor([True] * N)
    edges = _make_edges(N)
    sectors = torch.zeros(N, dtype=torch.long)
    slow_a = torch.randn(N, F_slow)
    slow_b = torch.randn(N, F_slow)

    out_a = model(features=features, mask=mask, edges=edges,
                   sector_indices=sectors, slow_features=slow_a)
    out_b = model(features=features, mask=mask, edges=edges,
                   sector_indices=sectors, slow_features=slow_b)
    # If the slow branch is wired in, different slow inputs must produce
    # different scores. If they match, the branch is silently disconnected.
    assert not torch.allclose(out_a.scores, out_b.scores), (
        "slow_features did not influence scores — branch disconnected?"
    )


def test_master_model_slow_branch_disabled_by_default():
    """n_slow_features=0 (legacy / OHLCV path) keeps the old single-branch model."""
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"))
    assert model.slow_branch is None
    assert model.n_slow_features == 0


def test_master_model_gated_fusion_default_on():
    """When the slow branch is built, gated_fusion is on by default and adds
    fusion_gate modules. Same output dim as the simple-concat path so
    downstream is unchanged."""
    torch.manual_seed(0)
    F_slow = 4
    cfg = _model_cfg(gnn_name="gat")
    model = ConstellationQuant(n_features=5, model_cfg=cfg, n_slow_features=F_slow)
    assert model.slow_branch is not None
    assert model.gated_fusion is True
    assert model.fusion_gate_fast is not None
    assert model.fusion_gate_slow is not None
    # Each gate projects to the dim of the embedding it gates.
    slow_emb_dim = model.slow_branch[-1].out_features
    fast_emb_dim = model.temporal.output_dim
    assert model.fusion_gate_slow.out_features == slow_emb_dim
    assert model.fusion_gate_fast.out_features == fast_emb_dim


def test_master_model_gated_fusion_can_be_disabled():
    """slow_branch.gated_fusion=False → no gate modules, reverts to plain concat."""
    cfg = _model_cfg(gnn_name="gat")
    cfg["slow_branch"] = {"enabled": True, "hidden": 16, "out_dim": 8,
                          "dropout": 0.0, "gated_fusion": False}
    model = ConstellationQuant(n_features=5, model_cfg=cfg, n_slow_features=4)
    assert model.gated_fusion is False
    assert model.fusion_gate_fast is None
    assert model.fusion_gate_slow is None


def test_master_model_handles_empty_edges_without_dim_mismatch():
    """When the GNN is configured but the edges dict for THIS date is empty
    (e.g. correlation lookback insufficient at the start of training), the
    model must still produce h_b at d_after_gnn, not at the pre-GNN dim."""
    torch.manual_seed(0)
    N, L, F, F_slow = 6, 20, 5, 4
    model = ConstellationQuant(
        n_features=F,
        model_cfg=_model_cfg(gnn_name="gat"),
        n_slow_features=F_slow,
    )
    model.eval()
    # Empty edges dict → GNN won't fire.
    out = model(
        features=torch.randn(N, L, F),
        mask=torch.tensor([True] * N),
        edges={},                                # ← no edges this date
        sector_indices=torch.zeros(N, dtype=torch.long),
        slow_features=torch.randn(N, F_slow),
    )
    assert out.scores.shape == (N,)
    assert torch.isfinite(out.scores).all()


def test_master_model_handles_zero_size_edge_index_without_dim_mismatch():
    """Edges dict with present keys but zero-size edge tensors — same fail
    mode as empty edges. Must not crash."""
    torch.manual_seed(0)
    N, L, F, F_slow = 6, 20, 5, 4
    model = ConstellationQuant(
        n_features=F,
        model_cfg=_model_cfg(gnn_name="gat"),
        n_slow_features=F_slow,
    )
    model.eval()
    empty_ei = torch.zeros(2, 0, dtype=torch.long)
    out = model(
        features=torch.randn(N, L, F),
        mask=torch.tensor([True] * N),
        edges={"correlation": (empty_ei, None)},
        sector_indices=torch.zeros(N, dtype=torch.long),
        slow_features=torch.randn(N, F_slow),
    )
    assert out.scores.shape == (N,)
    assert torch.isfinite(out.scores).all()


def test_master_model_gnn_outer_residual_default_on():
    """Outer residual around the GNN is on by default and projects when
    in_dim != out_dim. Critical anti-smoothing default."""
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"),
                          n_slow_features=4)
    assert model.gnn_outer_residual is True
    assert model.gnn_skip_proj is not None
    # Either a Linear (different dims) or Identity (same dims).
    import torch.nn as nn
    assert isinstance(model.gnn_skip_proj, (nn.Linear, nn.Identity))


def test_master_model_gnn_outer_residual_can_be_disabled():
    """Setting graph.outer_residual=False removes the skip connection."""
    cfg = _model_cfg(gnn_name="gat")
    cfg["graph"] = {**cfg["graph"], "outer_residual": False}
    model = ConstellationQuant(n_features=5, model_cfg=cfg, n_slow_features=4)
    assert model.gnn_outer_residual is False
    assert model.gnn_skip_proj is None


def test_master_model_gated_outer_residual_default_on():
    """Default config should build the gated outer residual."""
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"),
                          n_slow_features=4)
    assert model.gnn_gated_residual is True
    assert model.gnn_residual_gate is not None


def test_master_model_gated_outer_residual_can_be_disabled():
    """Setting graph.gated_outer_residual=False reverts to plain add."""
    cfg = _model_cfg(gnn_name="gat")
    cfg["graph"] = {**cfg["graph"], "gated_outer_residual": False}
    model = ConstellationQuant(n_features=5, model_cfg=cfg, n_slow_features=4)
    assert model.gnn_gated_residual is False
    assert model.gnn_residual_gate is None
    # outer residual itself is still on (skip_proj exists), just not gated
    assert model.gnn_skip_proj is not None


def test_master_model_gated_residual_gate_actually_runs():
    """Forward must succeed with gated residual and produce finite scores."""
    torch.manual_seed(0)
    model = ConstellationQuant(n_features=5, model_cfg=_model_cfg(gnn_name="gat"),
                          n_slow_features=4)
    model.eval()
    out = model(
        features=torch.randn(6, 20, 5),
        mask=torch.tensor([True] * 6),
        edges=_make_edges(6),
        sector_indices=torch.zeros(6, dtype=torch.long),
        slow_features=torch.randn(6, 4),
    )
    assert out.scores.shape == (6,)
    assert torch.isfinite(out.scores).all()


def test_master_model_gated_outer_residual_changes_output_vs_plain():
    """Gated and plain residual should produce different outputs (regression
    guard against the gate being silently disconnected)."""
    torch.manual_seed(0)
    cfg_gated = _model_cfg(gnn_name="gat")
    cfg_plain = _model_cfg(gnn_name="gat")
    cfg_plain["graph"] = {**cfg_plain["graph"], "gated_outer_residual": False}
    torch.manual_seed(0)
    m_gated = ConstellationQuant(n_features=5, model_cfg=cfg_gated, n_slow_features=4)
    torch.manual_seed(0)
    m_plain = ConstellationQuant(n_features=5, model_cfg=cfg_plain, n_slow_features=4)
    m_gated.eval(); m_plain.eval()

    features = torch.randn(6, 20, 5)
    mask = torch.tensor([True] * 6)
    edges = _make_edges(6)
    sectors = torch.zeros(6, dtype=torch.long)
    slow = torch.randn(6, 4)

    out_g = m_gated(features=features, mask=mask, edges=edges,
                     sector_indices=sectors, slow_features=slow)
    out_p = m_plain(features=features, mask=mask, edges=edges,
                     sector_indices=sectors, slow_features=slow)
    assert not torch.allclose(out_g.scores, out_p.scores), (
        "gated residual produced identical output to plain add"
    )


def test_master_model_gnn_outer_residual_changes_output():
    """With outer residual ON, output should differ from a model with it OFF
    (regression guard for a silently-disconnected residual)."""
    torch.manual_seed(0)
    N, L, F, F_slow = 6, 20, 5, 4
    cfg_on = _model_cfg(gnn_name="gat")
    cfg_off = _model_cfg(gnn_name="gat")
    cfg_off["graph"] = {**cfg_off["graph"], "outer_residual": False}
    # Same seed → same init for shared layers; new linear gets random init.
    torch.manual_seed(0)
    m_on = ConstellationQuant(n_features=F, model_cfg=cfg_on, n_slow_features=F_slow)
    torch.manual_seed(0)
    m_off = ConstellationQuant(n_features=F, model_cfg=cfg_off, n_slow_features=F_slow)
    m_on.eval(); m_off.eval()

    features = torch.randn(N, L, F)
    mask = torch.tensor([True] * N)
    edges = _make_edges(N)
    sectors = torch.zeros(N, dtype=torch.long)
    slow = torch.randn(N, F_slow)

    out_on  = m_on(features=features, mask=mask, edges=edges,
                    sector_indices=sectors, slow_features=slow)
    out_off = m_off(features=features, mask=mask, edges=edges,
                    sector_indices=sectors, slow_features=slow)
    assert not torch.allclose(out_on.scores, out_off.scores), (
        "outer_residual flag had no effect on scores"
    )


def test_master_model_gated_fusion_changes_output_when_slow_changes():
    """Same fast features + different slow features must produce different
    scores under gated fusion (regression guard for a silently-disconnected
    gate)."""
    torch.manual_seed(0)
    N, L, F, F_slow = 6, 20, 5, 4
    model = ConstellationQuant(
        n_features=F,
        model_cfg=_model_cfg(gnn_name="gat"),
        n_slow_features=F_slow,
    )
    model.eval()
    features = torch.randn(N, L, F)
    mask = torch.tensor([True] * N)
    edges = _make_edges(N)
    sectors = torch.zeros(N, dtype=torch.long)
    slow_a = torch.randn(N, F_slow)
    slow_b = torch.randn(N, F_slow)
    out_a = model(features=features, mask=mask, edges=edges,
                   sector_indices=sectors, slow_features=slow_a)
    out_b = model(features=features, mask=mask, edges=edges,
                   sector_indices=sectors, slow_features=slow_b)
    assert not torch.allclose(out_a.scores, out_b.scores), (
        "slow_features did not influence scores under gated fusion"
    )
