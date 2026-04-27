"""Output heads: ranking (primary), return, volatility."""

from typing import Any, Dict, Mapping, Optional

import torch.nn as nn

from constellation_quant.models.output_heads.ranking_head import RankingHead
from constellation_quant.models.output_heads.return_head import ReturnHead
from constellation_quant.models.output_heads.volatility_head import VolatilityHead


def get_output_head(
    name: str,
    in_dim: int,
    config: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    cfg = dict(config or {})
    mlp = cfg.get("mlp", [128, 64, 1])
    # Drop trailing 1 (the output layer handled inside the head).
    hidden = [int(x) for x in mlp[:-1]] if mlp and mlp[-1] == 1 else [int(x) for x in mlp]
    dropout = float(cfg.get("dropout", 0.1))

    key = name.lower().strip()
    if key == "ranking":
        return RankingHead(
            in_dim=in_dim,
            hidden=hidden or [128, 64],
            dropout=dropout,
            temperature_scaling=bool(cfg.get("temperature_scaling", True)),
        )
    if key == "return":
        return ReturnHead(in_dim=in_dim, hidden=hidden or [128], dropout=dropout)
    if key == "volatility":
        return VolatilityHead(in_dim=in_dim, hidden=hidden or [128], dropout=dropout)
    raise ValueError(f"Unknown output head {name!r}")


__all__ = ["RankingHead", "ReturnHead", "VolatilityHead", "get_output_head"]
