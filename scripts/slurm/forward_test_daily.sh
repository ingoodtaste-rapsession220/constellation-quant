#!/bin/bash
#SBATCH --job-name=cq-fwd
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00                  # fits in gpushort
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --account=pilot_gpu

# Daily paper-trading forward-test. Intended to run as a cron-driven
# `sbatch` after market close. Pulls latest data, scores today's universe,
# back-scores rows that now have `horizon` days of realised data.

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"
nvidia-smi || true

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export WANDB_MODE=offline

CHECKPOINT="${CHECKPOINT:-$DATA_DIR/checkpoints/default_best.pt}"

source "$PROJECT_ROOT/.venv/bin/activate"
cd "$PROJECT_ROOT"

# 1. Incremental data update (recent window only).
python scripts/download_data.py --resume --skip-membership --skip-fundamentals \
    --skip-sentiment \
    --start-date "$(date -d '30 days ago' '+%Y-%m-%d')" \
    --end-date   "$(date '+%Y-%m-%d')" || true

# 2. Score today's universe.
python scripts/forward_test.py predict --checkpoint "$CHECKPOINT"

# 3. Back-score everything that's knowable now.
python scripts/forward_test.py rescore

# 4. Print the live summary for the job log.
python scripts/forward_test.py summary

echo "[$(date)] Forward test complete."
