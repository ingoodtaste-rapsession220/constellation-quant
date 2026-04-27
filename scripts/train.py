"""Train a single model variant.

    python scripts/train.py \\
        --model-config    configs/model_config.yaml \\
        --training-config configs/training_config.yaml \\
        --data-config     configs/data_config.yaml \\
        --feature-config  configs/feature_config.yaml \\
        --paths-config    configs/paths.yaml

Default-safe for single-GPU / CPU runs. Pass `--distributed` when launched
via torchrun (DDP). `--resume` is the default in all SLURM templates.

Training requires data on disk — if `$DATA_DIR/raw/prices/*.parquet` and
the membership roster don't exist, the script aborts early with a clear
message. Run `scripts/download_data.py` first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from constellation_quant.data import DataPaths, MacroFeatures, MembershipRoster
from constellation_quant.data.dataset import DynaGraphDataset
from constellation_quant.graph import GraphBuilder, build_returns_wide
from constellation_quant.models import ConstellationQuant
from constellation_quant.training import Trainer, TrainerConfig
from constellation_quant.utils import (
    get_device,
    get_logger,
    init_distributed,
    is_main_process,
    load_config,
    log_environment,
    set_seed,
)

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-config",    type=Path, default=Path("configs/model_config.yaml"))
    p.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    p.add_argument("--data-config",     type=Path, default=Path("configs/data_config.yaml"))
    p.add_argument("--feature-config",  type=Path, default=Path("configs/feature_config.yaml"))
    p.add_argument("--paths-config",    type=Path, default=Path("configs/paths.yaml"))
    p.add_argument("--variant-name",    type=str,  default="default")
    p.add_argument("--distributed",     action="store_true",
                   help="Initialise DDP (set by torchrun-launched runs).")
    p.add_argument("--resume",          action="store_true",
                   help="Resume from last checkpoint if one exists.")
    return p.parse_args()


# ── Env helpers ────────────────────────────────────────────────────────────


def _ensure_path_env_vars() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd().resolve()))
    os.environ.setdefault("SCRATCH",      os.environ["PROJECT_ROOT"] + "/.scratch")
    os.environ.setdefault("DATA_DIR",     os.environ["SCRATCH"] + "/constellation_quant")


# ── Sample adapter ─────────────────────────────────────────────────────────


def make_sample_adapter(
    graph_builder: Optional[GraphBuilder],
):
    """Adapter that turns a Dataset sample into full model inputs.

    Handles graph construction (when enabled) and moves tensors to the
    target device. Returns the dict the Trainer expects.
    """

    def adapter(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        features = sample["features"].to(device)                    # (N, L, F_fast)
        mask     = sample["mask"].to(device)
        targets  = sample["targets"].to(device)
        sectors  = sample.get("sectors", torch.zeros_like(mask, dtype=torch.long)).to(device)

        out: Dict[str, Any] = {
            "features": features,
            "mask":     mask,
            "targets":  targets,
            "sector_indices": sectors,
            "return":   targets,                # return-head shares target with ranking
        }
        if "slow_features" in sample:
            out["slow_features"] = sample["slow_features"].to(device)
        if "volatility" in sample:
            out["volatility"] = sample["volatility"].to(device)

        if graph_builder is None:
            out["edges"] = {}
            return out

        # Build edges using an all-zero placeholder for per-stock embeddings
        # (the trainer doesn't need edges conditional on embeddings here —
        # R-GAT learns its own attention edges online, correlation/sector
        # edges are feature-independent).
        N = features.shape[0]
        built = graph_builder.build(
            pred_date=sample["date"],
            universe_tickers=sample["tickers"],
            node_features=np.zeros((N, 1), dtype=np.float32),
        )
        edges: Dict[str, Any] = {}
        for rel, spec in built.edges.items():
            ei = torch.from_numpy(spec.edge_index).to(device).long()
            ew = torch.from_numpy(spec.edge_weight).to(device).float() if spec.edge_weight.size else None
            edges[rel] = (ei, ew)
        out["edges"] = edges
        out["sector_indices"] = torch.from_numpy(built.sector_indices).to(device).long()
        return out

    return adapter


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    _ensure_path_env_vars()

    local_rank = init_distributed() if args.distributed else 0
    device = get_device(local_rank=local_rank if args.distributed else None)

    if is_main_process():
        log.info("Environment: {}", log_environment())

    # Configs
    model_cfg    = load_config(args.model_config)
    training_cfg = load_config(args.training_config)
    data_cfg     = load_config(args.data_config)
    paths_cfg    = load_config(args.paths_config)

    set_seed(int(training_cfg.get("seed", 42)),
             deterministic=bool(training_cfg.get("deterministic_cuda", True)))

    paths = DataPaths.from_config(paths_cfg)
    paths.ensure_dirs()

    # Data preconditions
    if not paths.membership_file.exists():
        log.error("No membership roster at {}. Run scripts/download_data.py first.",
                  paths.membership_file)
        return 2
    if not any(paths.raw_prices.glob("*.parquet")):
        log.error("No price parquet files in {}. Run scripts/download_data.py first.",
                  paths.raw_prices)
        return 2

    roster = MembershipRoster.load_json(paths.membership_file)

    # Optional macro / market-wide features (VIX / TNX / DXY / SPY 5d-changes).
    # Silent no-op when no macro parquets exist on disk.
    macro_loader = MacroFeatures.from_paths(paths)
    if not macro_loader.is_empty() and is_main_process():
        log.info("Macro features active: {}", list(macro_loader.series.keys()))

    # Datasets — train and val split from data_config.
    splits = data_cfg["splits"]
    lookback = int(model_cfg.get("lookback", 60))
    horizon  = int(model_cfg.get("horizon", 5))

    # stride = horizon → non-overlapping targets between consecutive samples.
    # purge_end = horizon → last target window in each split can't spill into
    # the next split's date range.
    train_ds = DynaGraphDataset(
        paths=paths, membership=roster,
        start_date=splits["train"]["start"], end_date=splits["train"]["end"],
        lookback=lookback, horizon=horizon, stride=horizon,
        purge_end=horizon, preload=True,
        macro_features=macro_loader,
    )
    val_ds = DynaGraphDataset(
        paths=paths, membership=roster,
        start_date=splits["val"]["start"], end_date=splits["val"]["end"],
        lookback=lookback, horizon=horizon, stride=horizon,
        purge_end=horizon, preload=True,
        macro_features=macro_loader,
    )

    if is_main_process():
        log.info("Datasets | train={} val={} | N_max={} F={}",
                 len(train_ds), len(val_ds),
                 train_ds.n_max, train_ds.shapes().n_features)

    # Graph builder — reuses train returns as the correlation input.
    returns_wide = _build_returns_wide_from_dataset(train_ds)
    graph_cfg = dict(model_cfg.get("graph", {}) or {})

    # Sector map — scraped from Wikipedia at data-download time, saved to
    # $DATA_DIR/sector_map.json. Required for Models B, C, H, I (sector
    # edges + hierarchy). Falls back to empty dict if missing (old checkpoints).
    sector_map_path = paths.data_dir / "sector_map.json"
    sector_map: Dict[str, str] = {}
    if sector_map_path.exists():
        with sector_map_path.open("r") as f:
            sector_map = {k.upper(): v for k, v in json.load(f).items()}
        if is_main_process():
            log.info("Loaded sector_map: {} tickers across {} sectors",
                     len(sector_map), len(set(sector_map.values())))
    else:
        if is_main_process():
            log.warning("No sector_map.json at {} — Models B/C/H/I will be impaired.",
                        sector_map_path)

    graph_builder: Optional[GraphBuilder] = None
    if graph_cfg.get("enabled", True) and graph_cfg.get("gnn_name", "rgat") != "none":
        graph_builder = GraphBuilder(
            model_cfg=model_cfg,
            sector_map=sector_map,
            returns_wide=returns_wide,
        )

    # Model
    train_shapes = train_ds.shapes()
    model = ConstellationQuant(
        n_features=train_shapes.n_features,
        model_cfg=model_cfg,
        n_slow_features=train_shapes.n_slow_features,
    ).to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=bool(training_cfg.get("distributed", {}).get(
                "find_unused_parameters", False,
            )),
        )

    if is_main_process():
        log.info("Model: {}", getattr(model, "module", model).describe())

    # Trainer
    tcfg = TrainerConfig(
        optimizer_cfg       = training_cfg.get("optimizer", {}),
        scheduler_cfg       = training_cfg.get("scheduler", {}),
        loop_cfg            = training_cfg.get("loop", {}),
        loss_cfg            = training_cfg.get("losses", {}),
        regularization_cfg  = training_cfg.get("regularization", {}),
        mixed_precision_cfg = training_cfg.get("mixed_precision", {}),
        early_stopping_cfg  = training_cfg.get("early_stopping", {}),
        checkpoint_cfg      = training_cfg.get("checkpoint", {}),
        wandb_cfg           = training_cfg.get("wandb", {}),
        variant_name        = args.variant_name,
        checkpoint_dir      = paths.checkpoint_dir,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        sample_adapter=make_sample_adapter(graph_builder),
        config=tcfg,
        device=device,
        rank=local_rank,
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    trainer.fit(resume=args.resume)
    return 0


def _build_returns_wide_from_dataset(dataset: DynaGraphDataset) -> pd.DataFrame:
    """Construct the wide log-returns frame directly from the Dataset's cache."""
    frames = {
        t: df.reset_index() for t, df in dataset._frames.items() if not df.empty   # noqa: SLF001
    }
    return build_returns_wide(frames)


if __name__ == "__main__":
    sys.exit(main())
