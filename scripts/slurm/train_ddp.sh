#!/bin/bash
#SBATCH --job-name=cq-ddp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4                      # request 4 GPUs on one node
#SBATCH --account=pilot_gpu

# Multi-GPU DDP training (single node). For multi-node, bump --nodes and
# propagate MASTER_ADDR via srun. Each GPU process handles a subset of
# training dates; the full graph fits on one GPU so we don't split nodes
# of the graph.

set -euo pipefail

VARIANT="${1:-default}"

echo "[$(date)] Job started on $(hostname)"
nvidia-smi || true

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
export WANDB_MODE=offline
export WANDB_DIR="$DATA_DIR/wandb"

source "$PROJECT_ROOT/.venv/bin/activate"

# DDP rendezvous on a single node
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=${MASTER_PORT:-29500}
NPROC=${SLURM_GPUS_ON_NODE:-4}

MODEL_CONFIG="$PROJECT_ROOT/configs/model_config.yaml"
FEATURE_CONFIG="$PROJECT_ROOT/configs/feature_config.yaml"
if [[ -f "$PROJECT_ROOT/configs/ablation/model_${VARIANT}.yaml" ]]; then
    MODEL_CONFIG="$PROJECT_ROOT/configs/ablation/model_${VARIANT}.yaml"
fi
if [[ -f "$PROJECT_ROOT/configs/ablation/features_${VARIANT}.yaml" ]]; then
    FEATURE_CONFIG="$PROJECT_ROOT/configs/ablation/features_${VARIANT}.yaml"
fi

cd "$PROJECT_ROOT"
torchrun --standalone --nproc_per_node="$NPROC" scripts/train.py \
    --model-config    "$MODEL_CONFIG" \
    --training-config configs/training_config.yaml \
    --data-config     configs/data_config.yaml \
    --feature-config  "$FEATURE_CONFIG" \
    --paths-config    configs/paths.yaml \
    --variant-name    "$VARIANT" \
    --distributed \
    --resume

echo "[$(date)] Job finished"
