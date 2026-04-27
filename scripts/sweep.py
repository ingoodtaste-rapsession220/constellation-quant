"""Wandb hyperparameter sweep launcher.

Two modes:

    # Register the sweep and print the ID:
    python scripts/sweep.py register --config configs/sweep_config.yaml

    # Launch N agents (reads sweep_id from stdin or --sweep-id):
    python scripts/sweep.py agent --sweep-id abc123 --count 10

On HPC the typical flow is:
    1. Register locally once, get the sweep ID.
    2. `sbatch scripts/slurm/sweep_agent.sh ABC123 COUNT=50` — each SLURM
       task runs one agent pulling trials from the shared wandb sweep queue.

Setting `WANDB_MODE=offline` is respected — in offline mode agents run but
logs land on local disk; sync them from the login node afterwards.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Register a new sweep and print the ID.")
    reg.add_argument("--config",  type=Path, default=Path("configs/sweep_config.yaml"))
    reg.add_argument("--project", type=str, default="constellation_quant_sweep")
    reg.add_argument("--entity",  type=str, default=None)

    agent = sub.add_parser("agent", help="Run a wandb agent against an existing sweep.")
    agent.add_argument("--sweep-id", type=str, required=True)
    agent.add_argument("--count",    type=int, default=1,
                        help="Number of trials this agent executes before exiting.")
    agent.add_argument("--project",  type=str, default="constellation_quant_sweep")
    agent.add_argument("--entity",   type=str, default=None)

    return p.parse_args()


def _qualified_sweep_id(sweep_id: str, project: str, entity: str | None) -> str:
    if "/" in sweep_id:
        return sweep_id          # already qualified
    if entity:
        return f"{entity}/{project}/{sweep_id}"
    return f"{project}/{sweep_id}"


def main() -> int:
    args = parse_args()
    try:
        import wandb
    except ImportError as exc:
        print(f"wandb is required: {exc}", file=sys.stderr)
        return 2

    if args.command == "register":
        import yaml
        with args.config.open("r") as f:
            sweep_cfg = yaml.safe_load(f)
        sweep_id = wandb.sweep(
            sweep=sweep_cfg,
            project=args.project,
            entity=args.entity,
        )
        print(sweep_id)
        return 0

    if args.command == "agent":
        qualified = _qualified_sweep_id(args.sweep_id, args.project, args.entity)
        wandb.agent(qualified, count=args.count)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
