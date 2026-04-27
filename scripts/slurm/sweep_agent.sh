#!/bin/bash
#SBATCH --job-name=cq-sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pilot_gpu
#SBATCH --array=0-49                      # 50 sweep agents; override on submit

# Run one wandb-sweep agent per array task. Each agent pulls trials from the
# shared sweep queue, runs scripts/train.py with sampled hyperparameters, and
# exits when the sweep terminates it.
#
#   Usage:  sbatch scripts/slurm/sweep_agent.sh <SWEEP_ID> [COUNT=1]

set -euo pipefail

SWEEP_ID="${1:-}"
COUNT="${2:-1}"
if [[ -z "$SWEEP_ID" ]]; then
    echo "usage: sbatch $0 <SWEEP_ID> [COUNT=1]" >&2
    exit 2
fi

echo "[$(date)] Sweep agent for $SWEEP_ID on $(hostname)"
nvidia-smi || true

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export WANDB_MODE=offline
export WANDB_DIR="$DATA_DIR/wandb"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"
python scripts/sweep.py agent --sweep-id "$SWEEP_ID" --count "$COUNT"

echo "[$(date)] Sweep agent finished"
