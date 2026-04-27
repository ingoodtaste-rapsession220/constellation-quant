# Runbook — running data_pipeline on Apocrita

Step-by-step from a fresh login. Each step is a SLURM job; you submit it
and wait. None of these run on the login node.

## Prerequisites

The constellation_quant venv at `/data/EECS-Theory/Zahir_DAYNGRAPH500/.venv`
is already set up (it's the one used for the existing training jobs).

## 0. One-time setup (login node, ~1 minute — no GPU, no compute)

```bash
cd /data/EECS-Theory/Zahir_DAYNGRAPH500
git pull                                # if the data_pipeline/ folder isn't here yet, sync from local
# OR: rsync from your laptop:
#   rsync -av data_pipeline/ acw720@login.hpc.qmul.ac.uk:/data/EECS-Theory/Zahir_DAYNGRAPH500/data_pipeline/
```

## 1. Smoke test — confirm everything imports (computeshort, ~2 min)

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=dp-smoke
#SBATCH --partition=computeshort
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/dp_smoke_%j.out
#SBATCH --error=logs/dp_smoke_%j.err
source data_pipeline/slurm/_common.sh
python -c "
import data_pipeline
from data_pipeline.edgar import EdgarClient, FilingsParser, FilingsStorage
print('ok — data_pipeline imports cleanly')
print('version:', data_pipeline.__version__)
"
EOF
```

Expected: log shows `ok — data_pipeline imports cleanly`.

## 2. Download the NLP models (computeshort, ~20 min, network only)

```bash
sbatch data_pipeline/slurm/download_models.sh
```

Watch the log:
```bash
tail -f logs/dp_models_*.out
```

When complete: ~3 GB in `$DATA_DIR/hf_cache/` (bge-m3 + FinBERT).

## 3. Mini-test the EDGAR client end-to-end (computeshort, ~5 min)

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=dp-mini
#SBATCH --partition=computeshort
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/dp_mini_%j.out
#SBATCH --error=logs/dp_mini_%j.err
source data_pipeline/slurm/_common.sh
USER_AGENT="constellation-quant research nikraftarz@gmail.com"
python -m data_pipeline.scripts.download_filings \
    --tickers       AAPL MSFT NVDA \
    --out-root      "$DATA_DIR" \
    --user-agent    "$USER_AGENT" \
    --forms         10-K \
    --date-from     2020-01-01 \
    --limit         9 \
    --cache-dir     "$DATA_DIR/cache/edgar"
python -m data_pipeline.scripts.parse_filings \
    --out-root      "$DATA_DIR"
ls -la "$DATA_DIR/raw/edgar/"
ls -la "$DATA_DIR/processed/edgar/"
EOF
```

Expected: 9 raw filings under `data/raw/edgar/<CIK>/`, 3 parsed parquets
under `data/processed/edgar/<CIK>/filings.parquet`.

## 4. Full S&P 500 download (compute, ~6–8 hours)

```bash
sbatch data_pipeline/slurm/download_filings.sh
```

This is the big one. Rate-limited at 10 req/s. Resume-safe — if it gets
interrupted, just re-submit.

Monitor:
```bash
squeue -u $USER
tail -f logs/dp_edgar_dl_*.out
wc -l data/raw/edgar/_manifest.csv      # number of filings on disk so far
```

## 5. Parse the full universe (compute, ~1 hour)

```bash
sbatch data_pipeline/slurm/parse_filings.sh
```

Produces `data/processed/edgar/_combined.parquet` — one row per filing,
columns: `cik, accession, form, filing_date, period, risk_factors, mda,
market_risk, ...`.

## 6. Compute NLP features (gpu, ~1–2 hours on A100)

```bash
sbatch data_pipeline/slurm/compute_nlp.sh
```

Produces three parquets in `data/processed/nlp_features/`:
- `embeddings.parquet` — bge-m3 vectors per (filing × section)
- `sentiment.parquet`  — FinBERT sentiment per (filing × section)
- `drift.parquet`      — Q-over-Q / Y-over-Y drift per filing

## 7. Prepare per-(ticker, date) wide features (computeshort, ~5 min)

```bash
sbatch data_pipeline/slurm/prepare_features.sh
```

Produces `data/processed/nlp_features/per_ticker_per_date.parquet` — ready
to merge into the existing `constellation_quant.data.dataset` pipeline as
new slow features.

## 8. Wire into constellation_quant (manual, separate session)

This is the integration step that goes back into the main package, not in
data_pipeline. It's a small change to `constellation_quant/data/dataset.py`
that loads the new parquet and concatenates it onto the existing slow-feature
tensor. Plan to do this in a focused session after the upstream pipeline
has produced its outputs.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `partition=short not found` | Wrong partition name | Use `compute` or `computeshort` |
| `urllib.error.HTTPError: 403` | Bad User-Agent | Use the format `name email@domain` (must contain `@`) |
| `HTTPError: 429` from EDGAR | Rate-limit hit | The client backs off automatically, re-run will resume |
| `No module named 'data_pipeline'` | Editable install missing | `_common.sh` installs it idempotently — re-run the SLURM script |
| `CUDA error: no kernel image` | Got assigned a V100 (CC 7.0) when PyTorch wheels need CC ≥ 7.5 | The compute_nlp.sh uses `--partition=gpu` which has A100s; if you get a V100, force CPU with `--device cpu` (much slower) or constrain via `--exclude=sbg2` |

## Keeping it tidy

`_common.sh` writes everything under `$DATA_DIR` (the project's data
directory) and `$PROJECT_ROOT/logs`. Nothing is written to `~` — your home
quota stays untouched.

Storage budget after a full pipeline run: ~30 GB for raw filings + ~5 GB
for parsed + ~2 GB for embeddings + ~3 GB for HF model cache = ~40 GB
total. Comfortably within the 436 GB project headroom.
