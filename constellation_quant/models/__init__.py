"""Neural network architectures: temporal encoders, GNNs, output heads, master model."""

from constellation_quant.models.model import ConstellationQuant, ModelOutputs
from constellation_quant.models.graph_nn import (
    GATStack,
    GCNStack,
    GraphSAGEStack,
    HierarchicalMessagePassing,
    RGATStack,
    get_gnn_layer,
    list_gnn_layers,
)
from constellation_quant.models.output_heads import (
    RankingHead,
    ReturnHead,
    VolatilityHead,
    get_output_head,
)
from constellation_quant.models.temporal import (
    InformerConfig,
    InformerEncoder,
    LSTMEncoder,
    MambaEncoder,
    TCNEncoder,
    TransformerEncoder,
    get_temporal_encoder,
    list_temporal_encoders,
)

__all__ = [
    "ConstellationQuant",
    "ModelOutputs",
    # Factories
    "get_temporal_encoder",
    "get_gnn_layer",
    "get_output_head",
    "list_temporal_encoders",
    "list_gnn_layers",
    # Encoders
    "InformerEncoder",
    "InformerConfig",
    "LSTMEncoder",
    "TransformerEncoder",
    "TCNEncoder",
    "MambaEncoder",
    # GNNs
    "GCNStack",
    "GATStack",
    "RGATStack",
    "GraphSAGEStack",
    "HierarchicalMessagePassing",
    # Heads
    "RankingHead",
    "ReturnHead",
    "VolatilityHead",
]
