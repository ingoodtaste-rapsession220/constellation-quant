#!/bin/bash
#SBATCH --job-name=dp-edgar-parse
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/dp_edgar_parse_%j.out
#SBATCH --error=logs/dp_edgar_parse_%j.err

# Parse the raw EDGAR filings into structured per-section parquet files.
# Pure CPU work, ~1 hour for the full S&P 500 universe.
#
# Run with:
#     sbatch data_pipeline/slurm/parse_filings.sh

source data_pipeline/slurm/_common.sh

python -m data_pipeline.scripts.parse_filings \
    --out-root      "$DATA_DIR" \
    --combined-out  "$DATA_DIR/processed/edgar/_combined.parquet"

echo "[$(date)] done."
