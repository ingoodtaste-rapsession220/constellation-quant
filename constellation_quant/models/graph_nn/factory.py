"""GNN-layer factory.

Two interfaces produced depending on name:

* Single-relation (`gcn`, `gat`, `graphsage`):
    forward(x, edge_index, edge_weight=None) -> (N, hidden)

* Multi-relation (`rgat`):
    forward(x, edges_by_rel, node_mask=None) -> (N, hidden)

The master model inspects `module.is_multi_relation` to dispatch correctly.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch.nn as nn

from constellation_quant.models.graph_nn.gat import GATStack
from constellation_quant.models.graph_nn.gcn import GCNStack
from constellation_quant.models.graph_nn.graphsage import GraphSAGEStack
from constellation_quant.models.graph_nn.rgat import RGATStack


# Flag multi-relational stacks so the master model knows to pass the full
# edges-by-relation dict instead of a flat edge_index.
GCNStack.is_multi_relation       = False
GATStack.is_multi_relation       = False
GraphSAGEStack.is_multi_relation = False
RGATStack.is_multi_relation      = True


_REGISTRY: Dict[str, Callable[[int, Mapping[str, Any]], nn.Module]] = {
    "gcn":       lambda d, c: GCNStack(d, **_filter(c)),
    "gat":       lambda d, c: GATStack(d, **_filter(c)),
    "graphsage": lambda d, c: GraphSAGEStack(d, **_filter(c)),
    "rgat":      lambda d, c: RGATStack(d, **_filter(c)),
}


def get_gnn_layer(
    name: str,
    in_dim: int,
    config: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    key = name.lower().strip()
    if key == "none":
        raise ValueError("GNN factory called with name='none' — caller should skip GNN entirely.")
    if key not in _REGISTRY:
        raise ValueError(f"Unknown GNN {name!r}. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[key](in_dim, dict(config or {}))


def list_gnn_layers() -> list[str]:
    return sorted(_REGISTRY)


def _filter(config: Mapping[str, Any]) -> Dict[str, Any]:
    keep = {
        "hidden_dim", "num_layers", "attention_heads", "dropout",
        "residual", "aggregator", "edge_types", "learned_attention",
        "top_k_attention",
    }
    return {k: v for k, v in config.items() if k in keep}
