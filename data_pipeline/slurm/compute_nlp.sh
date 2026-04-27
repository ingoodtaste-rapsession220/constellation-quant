#!/bin/bash
#SBATCH --job-name=dp-nlp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/dp_nlp_%j.out
#SBATCH --error=logs/dp_nlp_%j.err

# Run bge-m3 embeddings + FinBERT sentiment + Q-over-Q drift over the
# parsed filings. This is the GPU-bound stage; fits on a single A100 40GB
# at fp16.
#
# Run with:
#     sbatch data_pipeline/slurm/compute_nlp.sh

source data_pipeline/slurm/_common.sh

python -m data_pipeline.scripts.compute_nlp_features \
    --filings-parquet  "$DATA_DIR/processed/edgar/_combined.parquet" \
    --out-dir          "$DATA_DIR/processed/nlp_features" \
    --hf-cache         "$HF_HOME" \
    --device           cuda \
    --embedding-batch  16 \
    --sentiment-batch  64

echo "[$(date)] done."
