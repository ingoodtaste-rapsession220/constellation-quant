"""Master model class — assembles any ablation variant from config.

Forward-pass signatures (both supported):

  Single date (used by validator / evaluate.py):
    outputs = model(
        features:   (N, L, F)  float,
        mask:       (N,)       bool,
        edges:      Dict[str, (edge_index, edge_weight)],
        sector_indices: (N,)   int64,
    )

  Batched (used by trainer for throughput — ~5-10× faster on H100):
    outputs = model(
        features:   (B, N, L, F)  float,
        mask:       (B, N)        bool,
        edges:      List[Dict[str, (edge_index, edge_weight)]],   # length B
        sector_indices: (B, N)   int64,
    )

The Informer runs once on the full (B*N) pile of stock-windows which is
where almost all the FLOPs live. GNN / hierarchy / head still iterate per
date — their compute is negligible and batching edges requires node-index
offsets that don't compose cleanly with multi-relation R-GAT.

Output shapes mirror the input rank (unsqueezed B dim removed in the
single-date path). Every ablation variant A..I runs the same code path;
config picks the components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from constellation_quant.models.graph_nn.factory import get_gnn_layer
from constellation_quant.models.graph_nn.hierarchical_mp import HierarchicalMessagePassing
from constellation_quant.models.output_heads import get_output_head
from constellation_quant.models.temporal.factory import get_temporal_encoder


EdgeData = Tuple[torch.Tensor, Optional[torch.Tensor]]


@dataclass
class ModelOutputs:
    scores:     torch.Tensor
    ret:        Optional[torch.Tensor] = None
    volatility: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None


class ConstellationQuant(nn.Module):
    """The master model. One config → one variant."""

    def __init__(
        self,
        n_features: int,
        model_cfg: Mapping[str, Any],
        n_slow_features: int = 0,
    ):
        super().__init__()
        self.cfg = dict(model_cfg)

        # ── 1. Temporal encoder ─────────────────────────────────────────
        # In split mode `n_features` is just the FAST count (e.g. 7) — the
        # dataset routes the slow features through `slow_branch` below.
        temporal_cfg = dict(self.cfg.get("temporal", {}) or {})
        self.temporal_name = temporal_cfg.get("name", "informer")
        self.temporal = get_temporal_encoder(
            self.temporal_name, n_features=n_features, config=temporal_cfg,
        )
        d_temporal = self.temporal.output_dim

        # ── 2. Optional multi-scale (two encoders on two windows) ───────
        self.multi_scale = bool(self.cfg.get("multi_scale", False))
        self.multi_scale_windows = list(self.cfg.get("multi_scale_windows", [20, 120]))
        if self.multi_scale:
            self.secondary_temporal = get_temporal_encoder(
                self.temporal_name, n_features=n_features, config=temporal_cfg,
            )
            d_temporal = d_temporal + self.secondary_temporal.output_dim
        else:
            self.secondary_temporal = None

        # ── 2b. Slow static branch ─────────────────────────────────────
        # Tiny MLP for the per-stock slow-feature snapshot (8-d in technical
        # mode). Output is fused with the temporal embedding before the GNN
        # sees it. When n_slow_features=0 (OHLCV / legacy) this branch is
        # skipped entirely.
        #
        # Two fusion modes are supported, selected by `slow_branch.gated_fusion`
        # in model_config.yaml:
        #   - False: simple `torch.cat([fast, slow])` — fixed equal influence.
        #   - True (default): per-channel sigmoid gates on BOTH branches let
        #     the model decide per-stock-per-day how much to trust each path.
        #     Output dimensionality is unchanged (fast_dim + slow_dim) so the
        #     downstream GNN doesn't need to know which mode was used.
        slow_cfg = dict(self.cfg.get("slow_branch", {}) or {})
        slow_enabled = bool(slow_cfg.get("enabled", True)) and n_slow_features > 0
        self.n_slow_features = int(n_slow_features) if slow_enabled else 0
        if slow_enabled:
            slow_hidden = int(slow_cfg.get("hidden", 32))
            slow_out    = int(slow_cfg.get("out_dim", 16))
            slow_dropout = float(slow_cfg.get("dropout", 0.2))
            self.slow_branch = nn.Sequential(
                nn.Linear(self.n_slow_features, slow_hidden),
                nn.GELU(),
                nn.Dropout(slow_dropout),
                nn.Linear(slow_hidden, slow_out),
            )
            self.gated_fusion = bool(slow_cfg.get("gated_fusion", True))
            if self.gated_fusion:
                fusion_in = d_temporal + slow_out
                self.fusion_gate_fast = nn.Linear(fusion_in, d_temporal)
                self.fusion_gate_slow = nn.Linear(fusion_in, slow_out)
            else:
                self.fusion_gate_fast = None
                self.fusion_gate_slow = None
            d_temporal = d_temporal + slow_out
        else:
            self.slow_branch = None
            self.gated_fusion = False
            self.fusion_gate_fast = None
            self.fusion_gate_slow = None

        # ── 3. GNN (optional) ──────────────────────────────────────────
        # `outer_residual` adds a skip connection AROUND the entire GNN block
        # (h_post = GNN(h_pre) + project(h_pre)). Mitigates GAT/RGAT
        # over-smoothing — the temporal embedding survives even after the
        # graph layers mix it with neighbours. Defaults to True. The
        # `residual` field already used by some GNN factories controls
        # per-LAYER residuals INSIDE the GNN, which is independent.
        graph_cfg = dict(self.cfg.get("graph", {}) or {})
        self.graph_enabled = bool(graph_cfg.get("enabled", True))
        self.gnn_name = str(graph_cfg.get("gnn_name", "rgat"))
        if self.graph_enabled and self.gnn_name != "none":
            self.gnn = get_gnn_layer(self.gnn_name, in_dim=d_temporal, config=graph_cfg)
            d_after_gnn = self.gnn.output_dim
            self.gnn_outer_residual = bool(graph_cfg.get("outer_residual", True))
            # gated_outer_residual: when True, mix post-GNN and pre-GNN with a
            # learned per-channel sigmoid gate instead of a fixed 50/50 add.
            # Lets the model decide per-stock-day how much to trust the
            # cross-stock graph mix vs. the pure temporal embedding. Same
            # mechanic as the slow/fast fusion gate. Default True.
            self.gnn_gated_residual = bool(graph_cfg.get("gated_outer_residual", True))
            if self.gnn_outer_residual:
                self.gnn_skip_proj = (
                    nn.Identity() if d_temporal == d_after_gnn
                    else nn.Linear(d_temporal, d_after_gnn)
                )
                if self.gnn_gated_residual:
                    # Gate input: concat of post-GNN output and projected
                    # pre-GNN input — both already at d_after_gnn.
                    self.gnn_residual_gate = nn.Linear(d_after_gnn * 2, d_after_gnn)
                else:
                    self.gnn_residual_gate = None
            else:
                self.gnn_skip_proj = None
                self.gnn_residual_gate = None
        else:
            self.gnn = None
            d_after_gnn = d_temporal
            self.gnn_outer_residual = False
            self.gnn_gated_residual = False
            self.gnn_skip_proj = None
            self.gnn_residual_gate = None

        # ── 4. Hierarchical MP (optional) ──────────────────────────────
        hier_cfg = dict(self.cfg.get("hierarchy", {}) or {})
        self.hierarchy_enabled = bool(hier_cfg.get("enabled", False))
        if self.hierarchy_enabled:
            self.hierarchical_mp = HierarchicalMessagePassing(
                d_model=d_after_gnn,
                n_sectors=int(hier_cfg.get("sector_nodes", 11)),
            )
        else:
            self.hierarchical_mp = None

        # ── 5. Output heads ────────────────────────────────────────────
        heads_cfg = dict(self.cfg.get("heads", {}) or {})
        self.heads: Dict[str, nn.Module] = {}
        for name in ("ranking", "return", "volatility"):
            sub = dict(heads_cfg.get(name, {}) or {})
            if not sub.get("enabled", name == "ranking"):
                continue
            self.heads[name] = get_output_head(name, in_dim=d_after_gnn, config=sub)
        self.heads_m = nn.ModuleDict(self.heads)

        self.d_model = d_after_gnn
        self._n_features = n_features

    # ── Forward ────────────────────────────────────────────────────────

    def forward(
        self,
        features: torch.Tensor,                           # (N, L, F) or (B, N, L, F)
        mask:     Optional[torch.Tensor] = None,          # (N,) or (B, N) bool
        edges:    Optional[Union[Mapping[str, EdgeData],
                                 Sequence[Optional[Mapping[str, EdgeData]]]]] = None,
        sector_indices: Optional[torch.Tensor] = None,    # (N,) or (B, N) int64
        secondary_features: Optional[torch.Tensor] = None,
        slow_features: Optional[torch.Tensor] = None,     # (N, F_slow) or (B, N, F_slow)
    ) -> ModelOutputs:
        if features.dim() not in (3, 4):
            raise ValueError(
                f"features expected 3D (N, L, F) or 4D (B, N, L, F); got {features.shape}"
            )
        is_batched = features.dim() == 4
        if not is_batched:
            features = features.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
            if sector_indices is not None:
                sector_indices = sector_indices.unsqueeze(0)
            if secondary_features is not None:
                secondary_features = secondary_features.unsqueeze(0)
            if slow_features is not None:
                slow_features = slow_features.unsqueeze(0)
            # Single edges dict wraps into a 1-element list for the unified path.
            edges_list: List[Optional[Mapping[str, EdgeData]]] = [edges]
        else:
            if edges is None:
                edges_list = [None] * features.shape[0]
            elif isinstance(edges, Mapping):
                # Same graph applied to every date — unusual but valid.
                edges_list = [edges] * features.shape[0]
            else:
                edges_list = list(edges)

        B, N, L, F_dim = features.shape
        device = features.device
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        if sector_indices is None:
            sector_indices = torch.zeros(B, N, dtype=torch.long, device=device)

        # 1. Temporal encoding — batched across all B*N stock-windows so the
        #    Informer (the compute bottleneck) runs once per mini-batch
        #    rather than once per date.
        h_primary = self.temporal(features.reshape(B * N, L, F_dim)).reshape(B, N, -1)

        if self.multi_scale and self.secondary_temporal is not None:
            if secondary_features is None:
                short_L = min(self.multi_scale_windows[0], L)
                secondary_features = features[:, :, -short_L:]
            sec_L = secondary_features.shape[2]
            h_secondary = self.secondary_temporal(
                secondary_features.reshape(B * N, sec_L, F_dim)
            ).reshape(B, N, -1)
            h = torch.cat([h_primary, h_secondary], dim=-1)
        else:
            h = h_primary

        # Slow-static branch: fuse the per-stock slow embedding with the
        # temporal embedding so the GNN sees both streams. When the branch
        # is configured but no slow_features tensor was provided, fall back
        # to a zero vector so we never silently drop the channel.
        if self.slow_branch is not None:
            if slow_features is None:
                slow_emb = features.new_zeros(B, N, self.slow_branch[-1].out_features)
            else:
                if slow_features.shape[-1] != self.n_slow_features:
                    raise ValueError(
                        f"slow_features last dim {slow_features.shape[-1]} "
                        f"!= configured n_slow_features {self.n_slow_features}"
                    )
                slow_emb = self.slow_branch(slow_features.reshape(B * N, -1)).reshape(B, N, -1)
            if self.gated_fusion:
                # Gates computed from the joint state of both branches so the
                # fusion can suppress whichever branch is uninformative for
                # this particular stock-day. sigmoid gives independent [0, 1]
                # weights per channel — equivalent to a learned soft mask.
                combined = torch.cat([h, slow_emb], dim=-1)
                gate_fast = torch.sigmoid(self.fusion_gate_fast(combined))
                gate_slow = torch.sigmoid(self.fusion_gate_slow(combined))
                h = torch.cat([h * gate_fast, slow_emb * gate_slow], dim=-1)
            else:
                h = torch.cat([h, slow_emb], dim=-1)

        # 2. GNN + hierarchy + heads — per-date loop. Each date has its own
        #    edges so batching them requires node-index offset tricks that
        #    add complexity without a meaningful speed-up (GNN is <10% of
        #    per-step FLOPs at our scale).
        scores_list: List[torch.Tensor] = []
        ret_list:    List[Optional[torch.Tensor]] = []
        vol_list:    List[Optional[torch.Tensor]] = []
        emb_list:    List[torch.Tensor] = []

        for b in range(B):
            h_b = h[b]
            mask_b = mask[b]
            edges_b = edges_list[b]

            h_b_pre_gnn = h_b
            gnn_applied = False
            if self.gnn is not None and edges_b is not None and len(edges_b) > 0:
                if getattr(self.gnn, "is_multi_relation", False):
                    h_b = self.gnn(h_b, edges_b, node_mask=mask_b)
                    gnn_applied = True
                else:
                    for _, (edge_index, edge_weight) in edges_b.items():
                        if edge_index.numel() == 0:
                            continue
                        h_b = self.gnn(h_b, edge_index, edge_weight=edge_weight)
                        gnn_applied = True
                        break
            # Reconcile h_b's dim to d_after_gnn regardless of whether the
            # GNN fired. This matters in two cases:
            #   1. GNN never enabled → no projection needed (d_after_gnn = d_temporal).
            #   2. GNN enabled but no edges this date (e.g. correlation needs
            #      90d history but pred_date is too early) → must project the
            #      pre-GNN representation to d_after_gnn so heads still work.
            # Outer residual: when the GNN DID fire, mix pre + post; when it
            # didn't fire, we just use the projected pre-GNN.
            if self.gnn is not None and self.gnn_skip_proj is not None:
                pre = self.gnn_skip_proj(h_b_pre_gnn)
                if gnn_applied:
                    if self.gnn_residual_gate is not None:
                        gate = torch.sigmoid(
                            self.gnn_residual_gate(torch.cat([h_b, pre], dim=-1))
                        )
                        h_b = gate * h_b + (1.0 - gate) * pre
                    else:
                        h_b = h_b + pre
                else:
                    # GNN couldn't run — fall back to the projected temporal
                    # embedding so downstream heads see the expected dim.
                    h_b = pre

            if self.hierarchical_mp is not None and sector_indices is not None:
                h_b = self.hierarchical_mp(h_b, sector_indices[b], node_mask=mask_b)

            head_out_b: Dict[str, torch.Tensor] = {}
            for head_name, head in self.heads_m.items():
                raw = head(h_b)
                raw = raw.masked_fill(~mask_b, 0.0)
                head_out_b[head_name] = raw

            scores_list.append(
                head_out_b.get("ranking", torch.zeros(N, device=device))
            )
            ret_list.append(head_out_b.get("return"))
            vol_list.append(head_out_b.get("volatility"))
            emb_list.append(h_b)

        scores = torch.stack(scores_list, dim=0)                               # (B, N)
        ret = (torch.stack([r for r in ret_list], dim=0)
               if ret_list and ret_list[0] is not None else None)
        vol = (torch.stack([v for v in vol_list], dim=0)
               if vol_list and vol_list[0] is not None else None)
        embeddings = torch.stack(emb_list, dim=0)                              # (B, N, d)

        if not is_batched:
            scores = scores.squeeze(0)
            if ret is not None:
                ret = ret.squeeze(0)
            if vol is not None:
                vol = vol.squeeze(0)
            embeddings = embeddings.squeeze(0)

        return ModelOutputs(
            scores=scores,
            ret=ret,
            volatility=vol,
            embeddings=embeddings,
        )

    # ── Metadata helpers ──────────────────────────────────────────────

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if p.requires_grad or not trainable_only)
        return sum(p.numel() for p in params)

    def describe(self) -> Dict[str, Any]:
        return {
            "n_features": self._n_features,
            "n_slow_features": self.n_slow_features,
            "slow_branch": self.slow_branch is not None,
            "gated_fusion": self.gated_fusion,
            "temporal": self.temporal_name,
            "multi_scale": self.multi_scale,
            "gnn": self.gnn_name if self.gnn is not None else "none",
            "gnn_outer_residual": self.gnn_outer_residual,
            "gnn_gated_residual": self.gnn_gated_residual,
            "hierarchy": self.hierarchy_enabled,
            "heads": list(self.heads_m.keys()),
            "d_model": self.d_model,
            "num_parameters": self.num_parameters(),
        }
