"""Learning-rate schedule builders.

Supports the four schedulers referenced by `training_config.yaml > scheduler`:
    - cosine_annealing     : torch.optim.lr_scheduler.CosineAnnealingLR
    - warm_restarts        : CosineAnnealingWarmRestarts
    - reduce_on_plateau    : ReduceLROnPlateau  (stepped on val_ic)
    - one_cycle            : OneCycleLR

Plus an optional linear warmup prefix — when `warmup_epochs > 0`, the chosen
schedule is chained after a linear ramp from `warmup_start_factor·lr` to `lr`.
"""

from __future__ import annotations

from typing import Any, Mapping

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: Optimizer,
    config: Mapping[str, Any],
    steps_per_epoch: int | None = None,
):
    """Instantiate an LR scheduler from the parsed training_config block.

    Returns either a `_LRScheduler` or a `ReduceLROnPlateau` (the latter has
    a different step signature; callers should inspect and dispatch).
    """
    cfg = dict(config)
    name = str(cfg.get("name", "cosine_annealing")).lower()
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    warmup_factor = float(cfg.get("warmup_start_factor", 0.01))

    main: Any
    if name == "cosine_annealing":
        main = CosineAnnealingLR(
            optimizer,
            T_max=int(cfg.get("T_max", 100)),
            eta_min=float(cfg.get("min_lr", 0.0)),
        )
    elif name == "warm_restarts":
        main = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(cfg.get("T_0", 50)),
            T_mult=int(cfg.get("T_mult", 2)),
            eta_min=float(cfg.get("min_lr", 0.0)),
        )
    elif name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",                  # tracking val_ic (higher is better)
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 10)),
            min_lr=float(cfg.get("min_lr", 0.0)),
        )
    elif name == "one_cycle":
        if steps_per_epoch is None:
            raise ValueError("one_cycle scheduler requires steps_per_epoch")
        main = OneCycleLR(
            optimizer,
            max_lr=float(cfg.get("max_lr", optimizer.param_groups[0]["lr"])),
            steps_per_epoch=steps_per_epoch,
            epochs=int(cfg.get("epochs", 100)),
        )
    else:
        raise ValueError(f"Unknown scheduler {name!r}")

    if warmup_epochs > 0 and name != "one_cycle":
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        return SequentialLR(optimizer, schedulers=[warmup, main], milestones=[warmup_epochs])
    return main
