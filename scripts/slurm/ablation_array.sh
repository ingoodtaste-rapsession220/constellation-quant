#!/bin/bash
#SBATCH --job-name=cq-ablation
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=pilot_gpu
#SBATCH --array=0-8                       # variants A..I

# Array job: one task per ablation variant. Each trains its variant
# independently and writes summary.json under outputs/ablation/summaries/.
# The report generator consumes those afterwards.
#
# For gpushort (1 h cap) consider chain_short.sh per variant instead.

set -euo pipefail

echo "[$(date)] Array task $SLURM_ARRAY_TASK_ID on $(hostname)"
nvidia-smi || true

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export WANDB_MODE=offline
export WANDB_DIR="$DATA_DIR/wandb"

source "$PROJECT_ROOT/.venv/bin/activate"

VARIANTS=(A B C D E F G H I)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}

echo "Launching ablation variant: $VARIANT"
cd "$PROJECT_ROOT"

srun python scripts/train.py \
    --model-config    "configs/ablation/model_${VARIANT}.yaml" \
    --training-config configs/training_config.yaml \
    --data-config     configs/data_config.yaml \
    --feature-config  "configs/ablation/features_${VARIANT}.yaml" \
    --paths-config    configs/paths.yaml \
    --variant-name    "$VARIANT" \
    --resume

echo "[$(date)] Variant $VARIANT finished"
