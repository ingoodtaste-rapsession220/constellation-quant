#!/bin/bash
#SBATCH --job-name=cq-train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00                   # long queue — full convergence
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pilot_gpu

# Single-GPU training on the long queue. Expects PROJECT_ROOT to be exported
# or defaults to the pilot_gpu cluster layout. Resumes from the latest
# checkpoint if one exists.
#
# Submit with an optional variant:   sbatch scripts/slurm/train_single.sh I
# (Defaults to the "default" variant using configs/model_config.yaml as-is.)

set -euo pipefail

VARIANT="${1:-default}"

echo "[$(date)] Job started on $(hostname)"
nvidia-smi || true

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export WANDB_MODE=offline
export WANDB_DIR="$DATA_DIR/wandb"

source "$PROJECT_ROOT/.venv/bin/activate"

MODEL_CONFIG="$PROJECT_ROOT/configs/model_config.yaml"
FEATURE_CONFIG="$PROJECT_ROOT/configs/feature_config.yaml"
if [[ -f "$PROJECT_ROOT/configs/ablation/model_${VARIANT}.yaml" ]]; then
    MODEL_CONFIG="$PROJECT_ROOT/configs/ablation/model_${VARIANT}.yaml"
fi
if [[ -f "$PROJECT_ROOT/configs/ablation/features_${VARIANT}.yaml" ]]; then
    FEATURE_CONFIG="$PROJECT_ROOT/configs/ablation/features_${VARIANT}.yaml"
fi

cd "$PROJECT_ROOT"
srun python scripts/train.py \
    --model-config    "$MODEL_CONFIG" \
    --training-config configs/training_config.yaml \
    --data-config     configs/data_config.yaml \
    --feature-config  "$FEATURE_CONFIG" \
    --paths-config    configs/paths.yaml \
    --variant-name    "$VARIANT" \
    --resume

echo "[$(date)] Job finished"
