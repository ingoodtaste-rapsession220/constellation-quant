# Shared environment for data_pipeline SLURM jobs. Source this from each
# slurm script's body — it sets the project paths and activates the venv.
#
# Usage in an sbatch:
#     source slurm/_common.sh

set -euo pipefail

# Project paths (matches the existing constellation_quant setup on Apocrita)
export PROJECT_ROOT=/data/EECS-Theory/Zahir_DAYNGRAPH500
export DATA_DIR=$PROJECT_ROOT/data
export SCRATCH=$PROJECT_ROOT
export HF_HOME=$DATA_DIR/hf_cache                # huggingface cache root
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME

mkdir -p "$DATA_DIR" "$HF_HOME" "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

# Activate the project venv (same one used for training)
source "$PROJECT_ROOT/.venv/bin/activate"

# Install / upgrade the data_pipeline package in editable mode if not present.
# Idempotent — checks first.
if ! python -c "import data_pipeline" 2>/dev/null; then
    echo "Installing data_pipeline (editable)..."
    pip install -e "$PROJECT_ROOT/data_pipeline" --quiet
fi

echo "[$(date)] env ready — node=$SLURMD_NODENAME partition=$SLURM_JOB_PARTITION"
