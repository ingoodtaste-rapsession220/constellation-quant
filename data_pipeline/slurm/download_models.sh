#!/bin/bash
#SBATCH --job-name=dp-models
#SBATCH --partition=computeshort
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/dp_models_%j.out
#SBATCH --error=logs/dp_models_%j.err

# Pre-download the NLP models (bge-m3, FinBERT) into the project's local
# HuggingFace cache so the GPU job doesn't need outbound internet to fetch
# them. CPU-only — the download itself is just network I/O.
#
# Run with:
#     sbatch data_pipeline/slurm/download_models.sh

source data_pipeline/slurm/_common.sh

python -m data_pipeline.scripts.download_models \
    --cache-dir "$HF_HOME"

echo "[$(date)] done."
