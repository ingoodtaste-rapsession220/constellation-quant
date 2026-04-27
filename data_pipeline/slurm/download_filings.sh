#!/bin/bash
#SBATCH --job-name=dp-edgar-dl
#SBATCH --partition=compute
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/dp_edgar_dl_%j.out
#SBATCH --error=logs/dp_edgar_dl_%j.err

# Download all 10-K and 10-Q filings for the S&P 500 universe.
#
# Rate-limited at SEC's 10 req/s cap, so the full universe takes ~6-8 hours
# wall-clock (one filing per second worst case, three filings per second
# average with cached metadata).
#
# Resume-safe — re-running skips filings already in the manifest.
#
# Run with:
#     sbatch data_pipeline/slurm/download_filings.sh

source data_pipeline/slurm/_common.sh

# Build the ticker list from the project's existing membership roster.
# This is the union of all tickers ever in the S&P 500 between 1976 and
# 2026 — ~848 tickers total, of which ~665 have usable yfinance history,
# but EDGAR has data for all of them.
TICKERS_FILE=$DATA_DIR/cache/sp500_tickers.txt
mkdir -p "$DATA_DIR/cache"

if [ ! -f "$TICKERS_FILE" ]; then
    echo "Building ticker list from membership_roster.json ..."
    python - <<PY
import json
from pathlib import Path

src = Path("$DATA_DIR/membership_roster.json")
out = Path("$TICKERS_FILE")
data = json.loads(src.read_text())
# membership roster is { ticker: [date_added, date_removed] } or similar.
# Be liberal — accept dict-of-anything and dump unique tickers.
if isinstance(data, dict):
    tickers = sorted(data.keys())
else:
    # Fallback: accept list of strings or list of dicts with 'ticker'.
    tickers = sorted({
        (entry["ticker"] if isinstance(entry, dict) else entry)
        for entry in data
    })
out.write_text("\n".join(tickers) + "\n")
print(f"Wrote {len(tickers)} tickers to {out}")
PY
fi

USER_AGENT="constellation-quant research nikraftarz@gmail.com"

python -m data_pipeline.scripts.download_filings \
    --tickers-file  "$TICKERS_FILE" \
    --out-root      "$DATA_DIR" \
    --user-agent    "$USER_AGENT" \
    --forms         10-K 10-Q \
    --date-from     1995-01-01 \
    --cache-dir     "$DATA_DIR/cache/edgar"

echo "[$(date)] done."
