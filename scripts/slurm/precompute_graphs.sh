#!/bin/bash
#SBATCH --job-name=cq-graphs
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu                  # TODO: confirm with `sinfo`
#SBATCH --account=pilot_gpu              # adjust if CPU jobs use a different account
#SBATCH --array=0-14                     # 15 chunks across the full date range

# One-time preprocessing: compute all daily correlation graphs and save as
# sparse tensors. Shifts the O(N²) correlation cost out of the training hot
# path. Each array task handles a chunk of the date range.

set -euo pipefail

echo "[$(date)] Chunk $SLURM_ARRAY_TASK_ID on $(hostname)"

export PROJECT_ROOT="${PROJECT_ROOT:-/data/EECS-Theory/Zahir_DAYNGRAPH500}"
export DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"

source "$PROJECT_ROOT/.venv/bin/activate"

cd "$PROJECT_ROOT"
python -m constellation_quant.graph.precompute \
    --data-config  configs/data_config.yaml \
    --model-config configs/model_config.yaml \
    --chunk-id     "$SLURM_ARRAY_TASK_ID" \
    --num-chunks   "$SLURM_ARRAY_TASK_COUNT"

echo "[$(date)] Chunk finished"
