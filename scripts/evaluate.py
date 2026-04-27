"""Evaluate a trained checkpoint on the test period.

    python scripts/evaluate.py \\
        --checkpoint $SCRATCH/constellation_quant/checkpoints/default_best.pt \\
        --model-config    configs/model_config.yaml \\
        --data-config     configs/data_config.yaml \\
        --portfolio equal_weight

Loads a ConstellationQuant checkpoint, runs inference over the test-split date
range, feeds the per-day scores through the backtester + portfolio
constructor, and prints / logs the Sharpe, annual return, max drawdown,
turnover, total cost, and per-regime breakdown.

If `--checkpoint` is omitted or missing, falls back to a random-initialised
model — useful for smoke-testing the pipeline end-to-end. Results land under
`--output` (default `outputs/eval/<variant>/`).
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
from constellation_quant.evaluation import (
    Backtester,
    DailyPrediction,
    RegimeAnalyzer,
    aggregate_metrics,
    build_portfolio_constructor,
    daily_metrics,
    regime_stats_to_dataframe,
)
from constellation_quant.graph import GraphBuilder, build_returns_wide
from constellation_quant.models import ConstellationQuant
from constellation_quant.training.checkpoint import CheckpointManager
from constellation_quant.utils import get_device, get_logger, load_config, log_environment, set_seed

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",      type=Path, default=None)
    p.add_argument("--model-config",    type=Path, default=Path("configs/model_config.yaml"))
    p.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    p.add_argument("--data-config",     type=Path, default=Path("configs/data_config.yaml"))
    p.add_argument("--feature-config",  type=Path, default=Path("configs/feature_config.yaml"))
    p.add_argument("--paths-config",    type=Path, default=Path("configs/paths.yaml"))
    p.add_argument("--test-period",     type=str,  default=None,
                   help="YYYY-MM-DD:YYYY-MM-DD — overrides data_config test split.")
    p.add_argument("--portfolio",       type=str, default="equal_weight",
                   choices=["equal_weight", "risk_parity", "sector_neutral"])
    p.add_argument("--top-n",           type=int, default=50)
    p.add_argument("--transaction-cost-bps", type=float, default=5.0)
    p.add_argument("--rebalance",       type=str, default="W", choices=["D", "W", "M"])
    p.add_argument("--output",          type=Path, default=Path("outputs/eval"))
    return p.parse_args()


# ── Env helpers ────────────────────────────────────────────────────────────


def _ensure_path_env_vars() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd().resolve()))
    os.environ.setdefault("SCRATCH",      os.environ["PROJECT_ROOT"] + "/.scratch")
    os.environ.setdefault("DATA_DIR",     os.environ["SCRATCH"] + "/constellation_quant")


def _parse_test_period(spec: Optional[str], data_cfg: Mapping[str, Any]) -> tuple[str, str]:
    if spec is not None:
        start, end = spec.split(":")
        return start.strip(), end.strip()
    return data_cfg["splits"]["test"]["start"], data_cfg["splits"]["test"]["end"]


# ── Inference ──────────────────────────────────────────────────────────────


@torch.no_grad()
def generate_predictions(
    model: ConstellationQuant,
    dataset: DynaGraphDataset,
    graph_builder: Optional[GraphBuilder],
    device: torch.device,
) -> List[DailyPrediction]:
    """Score every date in `dataset` and emit DailyPrediction records."""
    model.eval()
    predictions: List[DailyPrediction] = []
    for i in range(len(dataset)):
        sample = dataset[i]
        features = sample["features"].to(device)
        mask     = sample["mask"].to(device)
        sectors  = sample.get("sectors", torch.zeros_like(mask, dtype=torch.long)).to(device)
        slow_features = (
            sample["slow_features"].to(device) if "slow_features" in sample else None
        )

        edges: Dict[str, Any] = {}
        if graph_builder is not None:
            built = graph_builder.build(
                pred_date=sample["date"],
                universe_tickers=sample["tickers"],
                node_features=np.zeros((features.shape[0], 1), dtype=np.float32),
            )
            for rel, spec in built.edges.items():
                ei = torch.from_numpy(spec.edge_index).to(device).long()
                ew = (torch.from_numpy(spec.edge_weight).to(device).float()
                      if spec.edge_weight.size else None)
                edges[rel] = (ei, ew)
            sectors = torch.from_numpy(built.sector_indices).to(device).long()

        out = model(features=features, mask=mask, edges=edges, sector_indices=sectors,
                    slow_features=slow_features)

        scores = out.scores.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        vol_np = (out.volatility.detach().cpu().numpy()
                  if out.volatility is not None else None)
        tickers = sample["tickers"]

        predictions.append(DailyPrediction(
            date=pd.Timestamp(sample["date"]).normalize(),
            tickers=list(tickers),
            scores=scores,
            mask=mask_np,
            volatility=vol_np,
        ))
    return predictions


# ── Reporting ──────────────────────────────────────────────────────────────


def write_report(
    output_dir: Path,
    result,
    predictions: List[DailyPrediction],
    regime_stats: Dict,
    config: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Summary JSON
    summary = {
        "backtest": result.summary(),
        "regimes":  {k: {
            "n_days":        s.n_days,
            "annual_return": s.annual_return,
            "annual_vol":    s.annual_vol,
            "sharpe":        s.sharpe,
            "max_drawdown":  s.max_drawdown,
            "hit_rate":      s.hit_rate,
        } for k, s in regime_stats.items()},
        "config": dict(config),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    # 2. Time series CSVs
    result.daily_returns.to_csv(output_dir / "daily_returns.csv", header=["return"])
    result.equity_curve.to_csv(output_dir / "equity_curve.csv", header=["equity"])
    result.drawdown.to_csv(output_dir / "drawdown.csv", header=["drawdown"])
    result.turnover.to_csv(output_dir / "turnover.csv", header=["turnover"])

    # 3. Per-day IC series (from raw predictions + realised targets omitted here —
    #    the backtest engine doesn't see forward returns directly).
    log.info("Wrote evaluation report to {}", output_dir)


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    _ensure_path_env_vars()
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

    if not paths.membership_file.exists():
        log.error("No membership roster at {}. Run scripts/download_data.py first.",
                  paths.membership_file)
        return 2
    if not any(paths.raw_prices.glob("*.parquet")):
        log.error("No price parquet files in {}. Run scripts/download_data.py first.",
                  paths.raw_prices)
        return 2

    roster = MembershipRoster.load_json(paths.membership_file)
    start, end = _parse_test_period(args.test_period, data_cfg)
    log.info("Test period: {} → {}", start, end)

    lookback = int(model_cfg.get("lookback", 60))
    horizon  = int(model_cfg.get("horizon",  5))

    # Evaluation uses stride=horizon to match training, and no purge —
    # the test split is the final split and has no successor to leak into.
    macro_loader = MacroFeatures.from_paths(paths)
    if not macro_loader.is_empty():
        log.info("Macro features active: {}", list(macro_loader.series.keys()))
    test_ds = DynaGraphDataset(
        paths=paths, membership=roster,
        start_date=start, end_date=end,
        lookback=lookback, horizon=horizon, stride=horizon,
        purge_end=0, preload=True,
        macro_features=macro_loader,
    )
    log.info("Dataset | test={} | N_max={} F={}",
             len(test_ds), test_ds.n_max, test_ds.shapes().n_features)

    # Device + model
    device = get_device()
    test_shapes = test_ds.shapes()
    model = ConstellationQuant(
        n_features=test_shapes.n_features,
        model_cfg=model_cfg,
        n_slow_features=test_shapes.n_slow_features,
    ).to(device)

    if args.checkpoint and args.checkpoint.exists():
        mgr = CheckpointManager(ckpt_dir=args.checkpoint.parent, variant_name="_eval")
        ckpt = mgr.load_into(args.checkpoint, model, strict=False)
        log.info("Loaded checkpoint from {} (epoch {}, best_metric {:.5f})",
                 args.checkpoint.name, ckpt.epoch, ckpt.best_metric)
    else:
        log.warning("No checkpoint provided — evaluating RANDOM-INITIALISED model "
                    "(pipeline smoke test only).")

    # Graph builder
    graph_builder: Optional[GraphBuilder] = None
    if model_cfg.get("graph", {}).get("enabled", True):
        returns_wide = build_returns_wide({
            t: df.reset_index() for t, df in test_ds._frames.items()      # noqa: SLF001
            if not df.empty
        })
        graph_builder = GraphBuilder(
            model_cfg=model_cfg, sector_map={}, returns_wide=returns_wide,
        )

    # 1. Predictions
    log.info("Generating predictions...")
    predictions = generate_predictions(model, test_ds, graph_builder, device)

    # 2. Backtest
    price_frames = {
        t: df.reset_index() for t, df in test_ds._frames.items()          # noqa: SLF001
    }
    constructor = build_portfolio_constructor(
        args.portfolio,
        config={"top_n": args.top_n},
    )
    backtester = Backtester(
        constructor=constructor,
        transaction_cost_bps=args.transaction_cost_bps,
        rebalance_frequency=args.rebalance,
        members_fn=lambda d: roster.tickers_on(pd.Timestamp(d).date()),
    )
    log.info("Running backtest ({} dates, portfolio={}, rebalance={})...",
             len(predictions), args.portfolio, args.rebalance)
    result = backtester.run(predictions, price_frames)
    log.info("Backtest: {}", result.summary())

    # 3. Regime analysis
    regime_stats = RegimeAnalyzer().analyze(result)

    # 4. Write report
    variant = args.checkpoint.stem if args.checkpoint else "random_init"
    output_dir = args.output / variant
    write_report(output_dir, result, predictions, regime_stats,
                  config={"portfolio": args.portfolio,
                          "rebalance": args.rebalance,
                          "transaction_cost_bps": args.transaction_cost_bps,
                          "top_n": args.top_n})
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
