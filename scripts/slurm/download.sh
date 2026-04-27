#!/bin/bash
#SBATCH --job-name=cq-download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=cpu                  # TODO: confirm with `sinfo` on your cluster
#SBATCH --account=pilot_gpu              # same account; adjust if CPU jobs use a different one

# CPU-only data download. Pulls yfinance prices + fundamentals + membership
# roster. Safe to re-run — every stage is resume-aware.
#
# If you've already downloaded locally, just rsync the data to $DATA_DIR
# instead of running this.

set -euo pipefail

echo "[$(date)] Job started on $(hostname)"

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
mkdir -p "$DATA_DIR/raw/prices" "$DATA_DIR/raw/fundamentals" "$DATA_DIR/raw/sentiment"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"
python scripts/download_data.py \
    --data-config  configs/data_config.yaml \
    --paths-config configs/paths.yaml \
    --resume

echo "[$(date)] Download complete."
