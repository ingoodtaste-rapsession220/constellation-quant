"""Plan / launch the full ablation sweep.

Three modes:

    # 1. Plan only — emit per-variant config files under configs/ablation/
    #    and a plan.json under outputs/ablation/. No training happens.
    python scripts/run_ablation.py --mode plan

    # 2. Local execution — run scripts/train.py per variant sequentially.
    #    Use --only A,B,C to restrict; --dry-run prints commands without
    #    executing. Respects --resume (skip variants with an existing summary).
    python scripts/run_ablation.py --mode local --only A,B,C

    # 3. SLURM emit — write an sbatch array-job script. You submit it yourself.
    python scripts/run_ablation.py --mode slurm
    sbatch outputs/ablation/run_array.sh

In all modes the runner writes:
    - configs/ablation/model_<name>.yaml
    - configs/ablation/features_<name>.yaml
    - outputs/ablation/plan.json
    - outputs/ablation/<name>.log   (local mode only)
    - outputs/ablation/run_array.sh (slurm mode only)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from constellation_quant.ablation import (
    AblationRunner,
    RunnerPaths,
    plan_all_sweeps,
)
from constellation_quant.utils import get_logger, load_config, log_environment

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",            type=Path, default=Path("configs/ablation_config.yaml"))
    p.add_argument("--model-config",      type=Path, default=Path("configs/model_config.yaml"))
    p.add_argument("--feature-config",    type=Path, default=Path("configs/feature_config.yaml"))
    p.add_argument("--training-config",   type=Path, default=Path("configs/training_config.yaml"))
    p.add_argument("--paths-config",      type=Path, default=Path("configs/paths.yaml"))
    p.add_argument("--mode", choices=["plan", "local", "slurm"], default="plan")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated variant names to include (e.g. A,B,C).")
    p.add_argument("--include-sweeps", action="store_true",
                   help="Include secondary sweeps in addition to A..I.")
    p.add_argument("--dry-run", action="store_true",
                   help="Local mode: print commands without executing.")
    p.add_argument("--no-resume", action="store_true",
                   help="Local mode: re-run variants even if summaries exist.")
    p.add_argument("--slurm-header", type=Path, default=None,
                   help="Path to a custom sbatch header file (overrides default).")
    return p.parse_args()


def _ensure_path_env_vars() -> None:
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd().resolve()))
    os.environ.setdefault("SCRATCH",      os.environ["PROJECT_ROOT"] + "/.scratch")
    os.environ.setdefault("DATA_DIR",     os.environ["SCRATCH"] + "/constellation_quant")


def _split_only(spec: Optional[str]) -> Optional[List[str]]:
    if spec is None:
        return None
    return [s.strip() for s in spec.split(",") if s.strip()]


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    _ensure_path_env_vars()
    log.info("Environment: {}", log_environment())

    ablation_cfg = load_config(args.config)
    model_cfg    = load_config(args.model_config)
    feature_cfg  = load_config(args.feature_config)

    project_root = Path.cwd().resolve()
    runner = AblationRunner(
        ablation_config=ablation_cfg,
        base_model_config=model_cfg,
        base_feature_config=feature_cfg,
        training_config_path=args.training_config,
        paths_config_path=args.paths_config,
        runner_paths=RunnerPaths.under(project_root),
    )

    plan = runner.plan(
        only=_split_only(args.only),
        include_sweeps=args.include_sweeps,
    )
    log.info("Planned {} variant(s).", len(plan.variants))

    if args.mode == "plan":
        print(json.dumps(plan.summary(), indent=2))
        return 0

    if args.mode == "local":
        codes = runner.run_local(plan, resume=not args.no_resume, dry_run=args.dry_run)
        failures = {n: c for n, c in codes.items() if c != 0}
        if failures:
            log.error("Failed variants: {}", failures)
            return 1
        log.info("All local runs completed.")
        return 0

    if args.mode == "slurm":
        header = args.slurm_header.read_text() if args.slurm_header else None
        script_path = runner.emit_slurm_array(plan, sbatch_header=header)
        print(f"Wrote SLURM script: {script_path}")
        print("Submit with:  sbatch", script_path)
        return 0

    log.error("Unknown mode: {}", args.mode)
    return 2


if __name__ == "__main__":
    sys.exit(main())
