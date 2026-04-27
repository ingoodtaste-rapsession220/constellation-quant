"""Temporal-encoder factory.

    encoder = get_temporal_encoder("informer", n_features=F, config={...})

Every encoder obeys the same interface:

    forward(x: (B, L, F)) -> (B, d_model)
    output_dim : int property

This lets the ablation runner swap encoders via `temporal.name` in the
model config without any other downstream change.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

import torch.nn as nn

from constellation_quant.models.temporal.informer import InformerConfig, InformerEncoder
from constellation_quant.models.temporal.lstm import LSTMEncoder
from constellation_quant.models.temporal.mamba import MambaEncoder
from constellation_quant.models.temporal.tcn import TCNEncoder
from constellation_quant.models.temporal.transformer import TransformerEncoder


def _build_informer(n_features: int, config: Mapping[str, Any]) -> nn.Module:
    cfg = InformerConfig(
        d_model=int(config.get("d_model", 256)),
        n_heads=int(config.get("n_heads", 8)),
        e_layers=int(config.get("e_layers", 3)),
        d_ff=int(config.get("d_ff", 512)),
        dropout=float(config.get("dropout", 0.1)),
        probsparse_factor=int(config.get("probsparse_factor", 5)),
        distil=bool(config.get("distil", True)),
        use_learnable_pe=bool(config.get("use_learnable_pe", True)),
        pooling=str(config.get("pooling", "attention_weighted_mean")),
    )
    return InformerEncoder(n_features=n_features, config=cfg)


_REGISTRY: Dict[str, Callable[[int, Mapping[str, Any]], nn.Module]] = {
    "informer":    _build_informer,
    "lstm":        lambda F, c: LSTMEncoder(F, **_filter(c, LSTMEncoder)),
    "transformer": lambda F, c: TransformerEncoder(F, **_filter(c, TransformerEncoder)),
    "tcn":         lambda F, c: TCNEncoder(F, **_filter(c, TCNEncoder)),
    "mamba":       lambda F, c: MambaEncoder(F, **_filter(c, MambaEncoder)),
}


def get_temporal_encoder(
    name: str,
    n_features: int,
    config: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    """Instantiate a temporal encoder by registry name."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown temporal encoder {name!r}. "
            f"Known: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[key](n_features, dict(config or {}))


def list_temporal_encoders() -> list[str]:
    return sorted(_REGISTRY)


def _filter(config: Mapping[str, Any], cls) -> Dict[str, Any]:
    """Drop config keys that aren't accepted by the target class."""
    # All encoders accept a **_ catch-all, but we prune noisy keys anyway so
    # the encoder's actual hyperparameters land cleanly.
    keep = {
        "d_model", "n_heads", "e_layers", "d_ff", "dropout",
        "num_layers", "num_blocks", "bidirectional", "pooling",
        "kernel_size", "d_state", "n_layers",
    }
    return {k: v for k, v in config.items() if k in keep}
