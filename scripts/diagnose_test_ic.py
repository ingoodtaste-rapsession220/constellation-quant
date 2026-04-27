"""Test-period IC + score-distribution diagnostic.

Mirrors `evaluate.py`'s checkpoint-loading and dataset/graph-builder
construction, but skips the backtester entirely. Two questions:

  1. What is the actual cross-sectional Spearman IC of the trained
     model on the held-out test period?

       IC ≈ 0   → overfitting (model has no signal)
       IC < 0   → regime inversion (model has REAL inverted signal)
       IC > 0   → backtest issue (signal exists but pipeline destroys it)

  2. Are the predicted scores collapsed to near-uniform values, or do
     they actually spread the universe? Distinguishes ListMLE-collapse
     from "model ranks fine, just wrong".

Usage on HPC:

    python scripts/diagnose_test_ic.py \\
        --checkpoint     data/checkpoints/D_best.pt \\
        --model-config   configs/ablation/model_D.yaml \\
        --feature-config configs/ablation/features_D.yaml

Optionally use `--limit N` to score only the first N test dates while
sanity-checking the script. Default is the full test split.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np
import pandas as pd
import torch

from constellation_quant.data import DataPaths, MacroFeatures, MembershipRoster
from constellation_quant.data.dataset import DynaGraphDataset
from constellation_quant.graph import GraphBuilder, build_returns_wide
from constellation_quant.models import ConstellationQuant
from constellation_quant.training.checkpoint import CheckpointManager
from constellation_quant.utils import get_device, get_logger, load_config, set_seed

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",      type=Path, required=True)
    p.add_argument("--model-config",    type=Path, default=Path("configs/model_config.yaml"))
    p.add_argument("--training-config", type=Path, default=Path("configs/training_config.yaml"))
    p.add_argument("--data-config",     type=Path, default=Path("configs/data_config.yaml"))
    p.add_argument("--feature-config",  type=Path, default=Path("configs/feature_config.yaml"))
    p.add_argument("--paths-config",    type=Path, default=Path("configs/paths.yaml"))
    p.add_argument("--test-period",     type=str,  default=None,
                   help="YYYY-MM-DD:YYYY-MM-DD — overrides data_config test split.")
    p.add_argument("--limit",           type=int,  default=None,
                   help="Score only the first N test dates (smoke run).")
    p.add_argument("--csv-out",         type=Path, default=None,
                   help="Append a one-row summary to this CSV (creates header if absent). "
                        "Use across phases to track headline metrics in one place.")
    p.add_argument("--phase-tag",       type=str,  default="",
                   help="Free-text tag (e.g. 'phase3-smaller') written to the CSV row.")
    return p.parse_args()


def _ensure_path_env_vars() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd().resolve()))
    os.environ.setdefault("SCRATCH",      os.environ["PROJECT_ROOT"] + "/.scratch")
    os.environ.setdefault("DATA_DIR",     os.environ["SCRATCH"] + "/constellation_quant")


def _parse_test_period(spec: Optional[str], data_cfg: Mapping[str, Any]) -> tuple[str, str]:
    if spec is not None:
        start, end = spec.split(":")
        return start.strip(), end.strip()
    return data_cfg["splits"]["test"]["start"], data_cfg["splits"]["test"]["end"]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. Pure numpy, ties handled via average rank."""
    if a.size < 3 or b.size < 3:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy()
    br = pd.Series(b).rank(method="average").to_numpy()
    if ar.std() == 0 or br.std() == 0:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")


def _print_score_distribution(label: str, date: pd.Timestamp,
                              scores: np.ndarray, targets: np.ndarray,
                              ic: float) -> None:
    print(f"\n[{label}] date={date.date()}  n_valid={scores.size}  IC={ic:+.4f}")
    print(f"  scores : min={scores.min():+.4f}  p10={_pct(scores,10):+.4f}  "
          f"p50={_pct(scores,50):+.4f}  p90={_pct(scores,90):+.4f}  max={scores.max():+.4f}")
    print(f"           mean={scores.mean():+.4f}  std={scores.std():.4f}  "
          f"IQR={_pct(scores,75)-_pct(scores,25):.4f}  "
          f"range/std={(scores.max()-scores.min())/(scores.std()+1e-12):.2f}")
    print(f"  targets: min={targets.min():+.4f}  p10={_pct(targets,10):+.4f}  "
          f"p50={_pct(targets,50):+.4f}  p90={_pct(targets,90):+.4f}  max={targets.max():+.4f}")
    print(f"           mean={targets.mean():+.4f}  std={targets.std():.4f}")


@torch.no_grad()
def main() -> int:
    args = parse_args()
    _ensure_path_env_vars()

    model_cfg    = load_config(args.model_config)
    training_cfg = load_config(args.training_config)
    data_cfg     = load_config(args.data_config)
    paths_cfg    = load_config(args.paths_config)

    set_seed(int(training_cfg.get("seed", 42)),
             deterministic=bool(training_cfg.get("deterministic_cuda", True)))

    paths = DataPaths.from_config(paths_cfg)
    if not paths.membership_file.exists():
        log.error("No membership roster at {}.", paths.membership_file)
        return 2
    if not any(paths.raw_prices.glob("*.parquet")):
        log.error("No price parquets in {}.", paths.raw_prices)
        return 2

    roster = MembershipRoster.load_json(paths.membership_file)
    start, end = _parse_test_period(args.test_period, data_cfg)
    log.info("Test period: {} → {}", start, end)

    lookback = int(model_cfg.get("lookback", 60))
    horizon  = int(model_cfg.get("horizon",  5))

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
    log.info("Dataset: {} dates, N_max={}, F={}",
             len(test_ds), test_ds.n_max, test_ds.shapes().n_features)

    device = get_device()
    test_shapes = test_ds.shapes()
    model = ConstellationQuant(
        n_features=test_shapes.n_features,
        model_cfg=model_cfg,
        n_slow_features=test_shapes.n_slow_features,
    ).to(device)

    if not args.checkpoint.exists():
        log.error("Checkpoint not found: {}", args.checkpoint)
        return 2
    mgr = CheckpointManager(ckpt_dir=args.checkpoint.parent, variant_name="_diag")
    ckpt = mgr.load_into(args.checkpoint, model, strict=False)
    log.info("Loaded {} (epoch={} best_metric={:.5f})",
             args.checkpoint.name, ckpt.epoch, ckpt.best_metric)

    graph_builder: Optional[GraphBuilder] = None
    if model_cfg.get("graph", {}).get("enabled", True):
        returns_wide = build_returns_wide({
            t: df.reset_index() for t, df in test_ds._frames.items()      # noqa: SLF001
            if not df.empty
        })
        graph_builder = GraphBuilder(
            model_cfg=model_cfg, sector_map={}, returns_wide=returns_wide,
        )

    model.eval()
    n_dates = len(test_ds) if args.limit is None else min(args.limit, len(test_ds))
    if n_dates == 0:
        log.error("No test dates to score.")
        return 1

    daily_ic: List[float] = []
    daily_dates: List[pd.Timestamp] = []
    daily_n: List[int] = []
    snapshot_indices = sorted({0, n_dates // 2, n_dates - 1})
    snapshots: List[tuple] = []     # (date, scores, targets, ic)

    log.info("Scoring {} test dates...", n_dates)
    for i in range(n_dates):
        sample = test_ds[i]
        features = sample["features"].to(device)
        mask     = sample["mask"].to(device)
        sectors  = sample.get("sectors", torch.zeros_like(mask, dtype=torch.long)).to(device)
        slow_features = (
            sample["slow_features"].to(device) if "slow_features" in sample else None
        )

        edges: dict = {}
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

        scores  = out.scores.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy().astype(bool)
        targets = sample["targets"].numpy()

        s_valid = scores[mask_np]
        t_valid = targets[mask_np]
        ic = _spearman(s_valid, t_valid)

        daily_ic.append(ic)
        daily_dates.append(sample["date"])
        daily_n.append(int(mask_np.sum()))
        if i in snapshot_indices:
            snapshots.append((sample["date"], s_valid.copy(), t_valid.copy(), ic))

        if (i + 1) % 50 == 0:
            log.info("  scored {} / {} dates", i + 1, n_dates)

    ic_arr = np.array(daily_ic, dtype=float)
    finite = ic_arr[np.isfinite(ic_arr)]

    print("\n" + "=" * 72)
    print("TEST-PERIOD IC SUMMARY  (variant: {})".format(args.checkpoint.stem))
    print("=" * 72)
    print(f"checkpoint:        {args.checkpoint}")
    print(f"period:            {start}  →  {end}")
    print(f"dates scored:      {len(ic_arr)}  (finite IC: {finite.size})")
    print(f"avg n_stocks/day:  {np.mean(daily_n):.1f}")
    print()
    print(f"mean IC:           {finite.mean():+.5f}")
    print(f"median IC:         {np.median(finite):+.5f}")
    print(f"std IC:            {finite.std():.5f}")
    se = finite.std() / np.sqrt(max(finite.size, 1))
    print(f"std error of mean: {se:.5f}")
    print(f"t-stat (mean/se):  {finite.mean() / max(se, 1e-12):+.2f}")
    print(f"frac days IC > 0:  {(finite > 0).mean():.3f}")
    print(f"frac days IC < 0:  {(finite < 0).mean():.3f}")
    print()
    print("IC percentiles:")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:02d}: {_pct(finite, q):+.4f}")
    print()
    print("IC by half-year bucket:")
    df = pd.DataFrame({"date": daily_dates, "ic": ic_arr, "n": daily_n}).dropna()
    df["bucket"] = df["date"].apply(
        lambda d: f"{d.year}-{'H1' if d.month <= 6 else 'H2'}"
    )
    grp = df.groupby("bucket")["ic"].agg(["count", "mean", "std"])
    for bucket, row in grp.iterrows():
        print(f"  {bucket}: n={int(row['count']):3d}  mean={row['mean']:+.4f}  std={row['std']:.4f}")

    print("\n" + "=" * 72)
    print("SCORE-DISTRIBUTION SNAPSHOTS  (3 dates: first / mid / last)")
    print("=" * 72)
    for date, s, t, ic in snapshots:
        idx = daily_dates.index(date)
        _print_score_distribution(f"snapshot[{idx}]", date, s, t, ic)

    print("\n" + "=" * 72)
    print("INTERPRETATION GUIDE")
    print("=" * 72)
    print("• mean IC ≈ 0 (|t-stat| < 2)        → overfitting (no test signal)")
    print("• mean IC < 0 with |t-stat| > 2     → regime inversion")
    print("• mean IC > 0 with negative Sharpe  → backtest pipeline bug")
    print("• score range/std < ~3 across dates → ListMLE collapse likely")
    print("• score std comparable across dates → scores spread fine")

    mean_ic = float(finite.mean()) if finite.size else float("nan")
    t_stat = float(finite.mean() / max(se, 1e-12)) if finite.size else float("nan")
    avg_range_over_std = float(np.mean([
        (s.max() - s.min()) / (s.std() + 1e-12) for _, s, _, _ in snapshots
    ]))
    if not np.isfinite(mean_ic):
        verdict = "verdict: NO DATA"
    elif abs(t_stat) < 2:
        verdict = "verdict: OVERFITTING (test signal indistinguishable from zero)"
    elif t_stat <= -2:
        verdict = "verdict: REGIME INVERSION (model has real but flipped signal)"
    elif t_stat >= 3 and mean_ic >= 0.05:
        verdict = "verdict: WORKING MODEL (signal beats threshold)"
    elif t_stat >= 2 and mean_ic >= 0.02:
        verdict = "verdict: WEAK SIGNAL (positive IC, below target threshold)"
    else:
        verdict = "verdict: AMBIGUOUS"
    if avg_range_over_std < 3.0:
        verdict += "  [WARN: score range/std < 3 — possible ListMLE collapse]"
    print("\n" + "=" * 72)
    print(verdict)
    print("=" * 72)

    if args.csv_out is not None:
        import csv
        from datetime import datetime
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "timestamp", "phase_tag", "checkpoint", "period_start", "period_end",
            "n_dates", "mean_ic", "median_ic", "std_ic", "t_stat",
            "frac_pos", "frac_neg", "avg_range_over_std", "verdict",
        ]
        row = [
            datetime.now().isoformat(timespec="seconds"),
            args.phase_tag,
            str(args.checkpoint),
            start, end,
            len(ic_arr),
            f"{mean_ic:+.6f}",
            f"{float(np.median(finite)):+.6f}" if finite.size else "nan",
            f"{float(finite.std()):.6f}" if finite.size else "nan",
            f"{t_stat:+.3f}",
            f"{float((finite > 0).mean()):.3f}" if finite.size else "nan",
            f"{float((finite < 0).mean()):.3f}" if finite.size else "nan",
            f"{avg_range_over_std:.2f}",
            verdict,
        ]
        existed = args.csv_out.exists()
        with args.csv_out.open("a", newline="") as f:
            writer = csv.writer(f)
            if not existed:
                writer.writerow(header)
            writer.writerow(row)
        log.info("Appended summary row to {}", args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
