"""Daily forward-testing driver.

Typical schedule (cron / SLURM):

    # Nightly, after market close:
    python scripts/forward_test.py predict --checkpoint $SCRATCH/.../best.pt

    # Later (e.g. once a week), score accumulated predictions:
    python scripts/forward_test.py rescore

    # Anytime: print live IC stats.
    python scripts/forward_test.py summary

Every subcommand is idempotent. `predict` refuses to overwrite an existing
record for the same date; `rescore` only writes result rows that aren't
already in the log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd
import torch

from constellation_quant.data import DataPaths, MembershipRoster
from constellation_quant.data.dataset import DynaGraphDataset
from constellation_quant.forward_testing import ForwardTestConfig, ForwardTestPipeline
from constellation_quant.graph import GraphBuilder, build_returns_wide
from constellation_quant.models import ConstellationQuant
from constellation_quant.training.checkpoint import CheckpointManager
from constellation_quant.utils import get_device, get_logger, load_config, log_environment, set_seed

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["predict", "rescore", "summary"])
    p.add_argument("--checkpoint",     type=Path, default=None)
    p.add_argument("--model-config",   type=Path, default=Path("configs/model_config.yaml"))
    p.add_argument("--data-config",    type=Path, default=Path("configs/data_config.yaml"))
    p.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    p.add_argument("--paths-config",   type=Path, default=Path("configs/paths.yaml"))
    p.add_argument("--horizon",        type=int,  default=None,
                   help="Override model horizon. Default: from model_config.")
    p.add_argument("--top-n",          type=int,  default=50)
    p.add_argument("--log-dir",        type=Path, default=Path("outputs/forward_test"))
    p.add_argument("--as-of",          type=str, default=None,
                   help="Prediction date (YYYY-MM-DD). Default: most recent "
                        "trading day on disk.")
    return p.parse_args()


def _ensure_path_env_vars() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd().resolve()))
    os.environ.setdefault("SCRATCH",      os.environ["PROJECT_ROOT"] + "/.scratch")
    os.environ.setdefault("DATA_DIR",     os.environ["SCRATCH"] + "/constellation_quant")


def _load_datasets(paths: DataPaths, roster: MembershipRoster, model_cfg: Mapping,
                    as_of: Optional[str]):
    """Build a small dataset covering the most recent rolling window for scoring."""
    lookback = int(model_cfg.get("lookback", 60))
    horizon  = int(model_cfg.get("horizon",  5))
    if as_of is not None:
        end_date = pd.Timestamp(as_of).normalize()
    else:
        # Find the most recent date across all price parquets.
        dates = []
        for p in paths.raw_prices.glob("*.parquet"):
            try:
                last = pd.read_parquet(p, columns=["date"])["date"].max()
                if pd.notna(last):
                    dates.append(pd.Timestamp(last).normalize())
            except Exception:
                continue
        if not dates:
            raise SystemExit("No price data on disk. Run scripts/download_data.py first.")
        end_date = max(dates)

    start_date = end_date - pd.Timedelta(days=max(lookback * 2, 180))
    ds = DynaGraphDataset(
        paths=paths, membership=roster,
        start_date=start_date, end_date=end_date,
        lookback=lookback, horizon=horizon, preload=True,
    )
    return ds, end_date


# ── Inference ──────────────────────────────────────────────────────────────


def _build_scorer(model: ConstellationQuant, graph_builder: Optional[GraphBuilder],
                   device: torch.device):
    """Closure that scores a single (date, tickers, feature_frames) triple."""
    @torch.no_grad()
    def _scorer(pred_date, tickers, frames):
        # Pull the latest feature window per ticker.
        windows: list[np.ndarray] = []
        for t in tickers:
            df = frames.get(t)
            if df is None or df.empty:
                raise ValueError(f"No data for {t} at {pred_date}")
            if "date" in df.columns:
                df = df.set_index("date")
            window = df.loc[df.index <= pred_date].tail(model.cfg.get("lookback", 60))
            arr = window[["open", "high", "low", "close", "adj_close", "volume"]] \
                .to_numpy(dtype=np.float32)
            windows.append(arr)
        features = torch.from_numpy(np.stack(windows, axis=0)).to(device)
        mask = torch.ones(features.shape[0], dtype=torch.bool, device=device)
        edges: dict = {}
        sectors = torch.zeros(features.shape[0], dtype=torch.long, device=device)
        if graph_builder is not None:
            built = graph_builder.build(
                pred_date=pd.Timestamp(pred_date),
                universe_tickers=list(tickers),
                node_features=np.zeros((features.shape[0], 1), dtype=np.float32),
            )
            for rel, spec in built.edges.items():
                ei = torch.from_numpy(spec.edge_index).to(device).long()
                ew = (torch.from_numpy(spec.edge_weight).to(device).float()
                      if spec.edge_weight.size else None)
                edges[rel] = (ei, ew)
            sectors = torch.from_numpy(built.sector_indices).to(device).long()
        out = model(features=features, mask=mask, edges=edges,
                     sector_indices=sectors)
        return out.scores.detach().cpu().numpy()
    return _scorer


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    _ensure_path_env_vars()
    log.info("Environment: {}", log_environment())

    # Configs (summary command still wants paths)
    paths_cfg = load_config(args.paths_config)
    paths = DataPaths.from_config(paths_cfg)
    paths.ensure_dirs()

    if args.command == "summary":
        pipeline = ForwardTestPipeline(ForwardTestConfig(
            log_dir=args.log_dir, horizon=args.horizon or 5, top_n=args.top_n,
        ))
        print(json.dumps(pipeline.summary().to_dict(), indent=2))
        return 0

    if not paths.membership_file.exists():
        log.error("Run scripts/download_data.py first.")
        return 2
    roster = MembershipRoster.load_json(paths.membership_file)

    if args.command == "rescore":
        frames = _load_price_frames(paths, roster)
        pipeline = ForwardTestPipeline(ForwardTestConfig(
            log_dir=args.log_dir, horizon=args.horizon or 5, top_n=args.top_n,
        ))
        n = pipeline.rescore(frames)
        log.info("Rescored {} prediction row(s).", n)
        return 0

    # args.command == "predict"
    model_cfg = load_config(args.model_config)
    training_cfg = load_config(args.training_config)
    set_seed(int(training_cfg.get("seed", 42)),
             deterministic=bool(training_cfg.get("deterministic_cuda", True)))

    ds, end_date = _load_datasets(paths, roster, model_cfg, args.as_of)

    device = get_device()
    model = ConstellationQuant(n_features=ds.shapes().n_features, model_cfg=model_cfg).to(device)

    if args.checkpoint and args.checkpoint.exists():
        mgr = CheckpointManager(ckpt_dir=args.checkpoint.parent, variant_name="_fwd")
        mgr.load_into(args.checkpoint, model, strict=False)
        log.info("Loaded checkpoint: {}", args.checkpoint.name)
    else:
        log.warning("Scoring with RANDOM-INITIALISED weights — dev run only.")

    graph_builder: Optional[GraphBuilder] = None
    if model_cfg.get("graph", {}).get("enabled", True):
        returns_wide = build_returns_wide({
            t: df.reset_index() for t, df in ds._frames.items()      # noqa: SLF001
            if not df.empty
        })
        graph_builder = GraphBuilder(
            model_cfg=model_cfg, sector_map={}, returns_wide=returns_wide,
        )

    # Universe = roster members on `end_date` that also have data on disk.
    members = sorted(roster.tickers_on(end_date.date()))
    members = [t for t in members if t in ds._frames and not ds._frames[t].empty]
    if not members:
        log.error("No tradeable members on {} with data available.", end_date.date())
        return 3

    pipeline = ForwardTestPipeline(ForwardTestConfig(
        log_dir=args.log_dir,
        horizon=int(args.horizon or model_cfg.get("horizon", 5)),
        top_n=args.top_n,
    ))
    scorer = _build_scorer(model, graph_builder, device)
    frames = {t: ds._frames[t] for t in members}
    pipeline.predict(end_date, members, frames, scorer)
    print(json.dumps({"date": end_date.date().isoformat(),
                       "n_predictions": len(members)}, indent=2))
    return 0


def _load_price_frames(paths: DataPaths, roster: MembershipRoster) -> dict:
    frames = {}
    for ticker in roster.all_tickers_ever():
        path = paths.price_file(ticker)
        if path.exists():
            try:
                frames[ticker] = pd.read_parquet(path)
            except Exception:
                continue
    return frames


if __name__ == "__main__":
    sys.exit(main())
