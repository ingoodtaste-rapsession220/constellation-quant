#!/bin/bash
#SBATCH --job-name=cq-short
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00                   # gpushort 1 h hard cap
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --account=pilot_gpu
#SBATCH --signal=B:SIGTERM@120            # SIGTERM 2 min before kill

# Chained short-queue training. Use `chain_short.sh` to submit N linked jobs.
# Each run loads the latest checkpoint for the variant, trains until either
# val_IC converges OR SLURM kills it at 60 min, and saves every epoch so the
# next run picks up with no lost work.
#
#   Usage:  sbatch scripts/slurm/train_short.sh <VARIANT>
#   Chain:  bash   scripts/slurm/chain_short.sh <VARIANT> [N=6]

set -euo pipefail

VARIANT="${1:-default}"

echo "[$(date)] Job started on $(hostname)"
nvidia-smi || true

# ── Environment ─────────────────────────────────────────────────────────
export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export WANDB_MODE=offline
export WANDB_DIR="$DATA_DIR/wandb"

# ── Activate venv ───────────────────────────────────────────────────────
source "$PROJECT_ROOT/.venv/bin/activate"

# ── Pick per-variant configs when present (from run_ablation.py) ────────
MODEL_CONFIG="$PROJECT_ROOT/configs/model_config.yaml"
FEATURE_CONFIG="$PROJECT_ROOT/configs/feature_config.yaml"
if [[ -f "$PROJECT_ROOT/configs/ablation/model_${VARIANT}.yaml" ]]; then
    MODEL_CONFIG="$PROJECT_ROOT/configs/ablation/model_${VARIANT}.yaml"
fi
if [[ -f "$PROJECT_ROOT/configs/ablation/features_${VARIANT}.yaml" ]]; then
    FEATURE_CONFIG="$PROJECT_ROOT/configs/ablation/features_${VARIANT}.yaml"
fi

echo "  variant         = $VARIANT"
echo "  model_config    = $MODEL_CONFIG"
echo "  feature_config  = $FEATURE_CONFIG"
echo "  PROJECT_ROOT    = $PROJECT_ROOT"
echo "  DATA_DIR        = $DATA_DIR"

# ── Train ───────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"
srun python scripts/train.py \
    --model-config    "$MODEL_CONFIG" \
    --training-config configs/training_config.yaml \
    --data-config     configs/data_config.yaml \
    --feature-config  "$FEATURE_CONFIG" \
    --paths-config    configs/paths.yaml \
    --variant-name    "$VARIANT" \
    --resume

echo "[$(date)] Job finished (exit $?)"
