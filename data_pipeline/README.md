# data_pipeline

Self-contained subsystem for **acquiring, processing, and integrating
external free data sources** into the constellation-quant project.

This is where Phase A onward of [`NEXT_STEPS.md`](../NEXT_STEPS.md) gets
built. It lives inside the constellation-quant repo but is structured as
its own Python subproject so it can be developed, tested, and (eventually)
shipped to EC2 independently.

## What's in here

| Module | Status | What |
|---|---|---|
| `data_pipeline.edgar` | **implemented** | SEC EDGAR API client + 10-K/10-Q parser |
| `data_pipeline.nlp` | **implemented** | bge-m3 embeddings + FinBERT sentiment + Q-over-Q drift |
| `data_pipeline.integration` | **implemented** | merge filings-derived features into the constellation_quant dataset |
| `data_pipeline.wikidata` | stub | Wikidata SPARQL → multi-relational corporate edges (Phase F) |
| `data_pipeline.news` | stub | Free news RSS ingestion + event extraction (Phase G) |
| `data_pipeline.form4` | stub | SEC Form 4 insider trading (Phase H) |
| `data_pipeline.fred` | stub | Broader FRED macro series (Phase I) |
| `data_pipeline.transcripts` | stub | Seeking Alpha transcripts + earnings call audio (Phase J) |
| `data_pipeline.wikipedia` | stub | Wikipedia page views / edit volumes (Phase K) |

## Pipeline overview

```
S&P 500 ticker list
        │
        ▼
[edgar.client]   →  raw filings    (HTML/txt)            in  data/raw/edgar/
        │
[edgar.parser]   →  structured     (parquet, sections)   in  data/processed/edgar/
        │
        ├──► [nlp.embeddings]  →  per-filing embeddings              (parquet)
        ├──► [nlp.sentiment]   →  section-level sentiment scores     (parquet)
        └──► [nlp.drift]       →  Q-over-Q cosine drift features     (parquet)
                                      │
                                      ▼
                        [integration.prepare_features]
                                      │
                                      ▼
              joined per-stock-per-date slow features
                                      │
                                      ▼
              dropped into constellation_quant.data.dataset
              (no architecture changes; this is a feature-set extension)
```

## Quickstart on Apocrita

These commands run on the HPC, in this order. Each step submits a SLURM job;
you don't run anything on the login node.

```bash
cd /data/EECS-Theory/Zahir_DAYNGRAPH500/data_pipeline

# 1. one-time: install the data_pipeline package into the project venv
pip install -e .

# 2. one-time: download the NLP models (bge-m3, FinBERT) into the
#    HuggingFace cache so they're available offline. ~20 min on CPU.
sbatch slurm/download_models.sh

# 3. download all 10-K and 10-Q filings for the S&P 500 universe.
#    Uses computeshort partition with chained jobs — the rate-limited
#    pull at 10 req/s puts the full universe at ~6-8 hours wall-clock.
sbatch slurm/download_filings.sh

# 4. parse the downloaded filings into structured per-section text.
#    CPU job, ~1 hour for the full universe.
sbatch slurm/parse_filings.sh

# 5. compute embeddings + sentiment over the parsed filings.
#    GPU job (A100), ~1-2 hours.
sbatch slurm/compute_nlp.sh

# 6. prepare features for integration with the existing dataset.
sbatch slurm/prepare_features.sh
```

Each SLURM script writes its outputs deterministically to a path the next
script reads from, so they form a clean pipeline. Each is resume-safe —
re-running skips already-completed work.

## Storage layout (HPC)

Everything goes under `$DATA_DIR` (typically `$PROJECT_ROOT/data/`):

```
data/
├── raw/
│   ├── edgar/                     # raw 10-K/10-Q HTML
│   │   └── <CIK>/<accession>.txt
│   └── ...
├── processed/
│   ├── edgar/                     # parsed structured filings (parquet)
│   │   └── <CIK>/filings.parquet
│   └── nlp_features/
│       ├── embeddings.parquet
│       ├── sentiment.parquet
│       └── drift.parquet
└── ...                            # existing constellation_quant data
```

## Local development

The package is installable locally for development and unit tests; only the
SLURM submissions are HPC-specific.

```bash
cd data_pipeline
pip install -e ".[dev]"
pytest                              # run unit tests
```

## Next phases

The stub modules each contain a `TODO.md` with the implementation
checklist. A new chat can pick up any of those phases by reading the
[`NEXT_STEPS.md`](../NEXT_STEPS.md) at repo root for context, then opening
the corresponding `TODO.md` for the specific phase.
