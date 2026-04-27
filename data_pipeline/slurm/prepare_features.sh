#!/bin/bash
#SBATCH --job-name=dp-features
#SBATCH --partition=computeshort
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/dp_features_%j.out
#SBATCH --error=logs/dp_features_%j.err

# Build the per-(ticker, date) wide feature parquet that constellation_quant
# can ingest as new slow features. CPU-only.
#
# Run with:
#     sbatch data_pipeline/slurm/prepare_features.sh

source data_pipeline/slurm/_common.sh

python -m data_pipeline.scripts.prepare_features \
    --nlp-features-dir   "$DATA_DIR/processed/nlp_features" \
    --cache-dir          "$DATA_DIR/cache/edgar" \
    --trading-days-from  1995-01-01 \
    --trading-days-to    2024-12-31 \
    --out-path           "$DATA_DIR/processed/nlp_features/per_ticker_per_date.parquet"

echo "[$(date)] done."
