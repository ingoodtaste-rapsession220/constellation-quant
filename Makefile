# constellation-quant — top-level commands.
# All targets are thin wrappers over scripts/*.py; they pick up PROJECT_ROOT
# and $DATA_DIR / $SCRATCH from the environment (set by HPC launch scripts).

PYTHON   ?= python
PIP      ?= pip
PROJECT  := constellation_quant

# Config paths (override on the command line: make train MODEL_CONFIG=...)
MODEL_CONFIG    ?= configs/model_config.yaml
TRAINING_CONFIG ?= configs/training_config.yaml
DATA_CONFIG     ?= configs/data_config.yaml
FEATURE_CONFIG  ?= configs/feature_config.yaml
PATHS_CONFIG    ?= configs/paths.yaml
ABLATION_CONFIG ?= configs/ablation_config.yaml
SWEEP_CONFIG    ?= configs/sweep_config.yaml

# Defaults for evaluate + forward-test; override to point at a real checkpoint.
CHECKPOINT   ?= $(SCRATCH)/constellation_quant/checkpoints/default_best.pt
VARIANT_NAME ?= default

.PHONY: help install dev data train ablation ablation-slurm evaluate report \
        sweep-register sweep-agent forward-predict forward-rescore forward-summary \
        test lint format clean

help:
	@echo "constellation-quant — available targets:"
	@echo ""
	@echo "  Setup"
	@echo "    install            Install package (editable)"
	@echo "    dev                Install + dev deps (pytest, ruff, mypy)"
	@echo ""
	@echo "  Pipeline"
	@echo "    data               Download + cache all raw data"
	@echo "    train              Train one model variant"
	@echo "    evaluate           Evaluate CHECKPOINT=... on the test period"
	@echo "    ablation           Plan the full 9-variant ablation sweep"
	@echo "    ablation-slurm     Emit an sbatch array-job script"
	@echo "    report             Generate the HTML (+ optional PDF) report"
	@echo ""
	@echo "  Hyperparameter sweep (wandb)"
	@echo "    sweep-register     Register a new wandb sweep (prints ID)"
	@echo "    sweep-agent        Run a sweep agent (SWEEP_ID=... required)"
	@echo ""
	@echo "  Forward testing / paper trading"
	@echo "    forward-predict    Score today's universe (appends to log)"
	@echo "    forward-rescore    Back-score knowable predictions"
	@echo "    forward-summary    Print rolling live-IC summary"
	@echo ""
	@echo "  Quality"
	@echo "    test               Run the pytest suite"
	@echo "    lint               Run ruff + mypy"
	@echo "    format             Auto-format with ruff"
	@echo "    clean              Remove caches and build artefacts"

# ── Setup ─────────────────────────────────────────────────────────────────

install:
	$(PIP) install -e .

dev: install
	$(PIP) install pytest pytest-cov ruff mypy

# ── Pipeline ──────────────────────────────────────────────────────────────

data:
	$(PYTHON) scripts/download_data.py \
		--data-config  $(DATA_CONFIG) \
		--paths-config $(PATHS_CONFIG) \
		--resume

train:
	$(PYTHON) scripts/train.py \
		--model-config    $(MODEL_CONFIG) \
		--training-config $(TRAINING_CONFIG) \
		--data-config     $(DATA_CONFIG) \
		--feature-config  $(FEATURE_CONFIG) \
		--paths-config    $(PATHS_CONFIG) \
		--variant-name    $(VARIANT_NAME) \
		--resume

evaluate:
	$(PYTHON) scripts/evaluate.py \
		--checkpoint    $(CHECKPOINT) \
		--model-config  $(MODEL_CONFIG) \
		--data-config   $(DATA_CONFIG) \
		--paths-config  $(PATHS_CONFIG)

ablation:
	$(PYTHON) scripts/run_ablation.py --config $(ABLATION_CONFIG) --mode plan

ablation-slurm:
	$(PYTHON) scripts/run_ablation.py --config $(ABLATION_CONFIG) --mode slurm

report:
	$(PYTHON) scripts/generate_report.py \
		--summaries outputs/ablation/summaries \
		--output    outputs/final_report \
		--config-snapshot $(MODEL_CONFIG)

# ── Hyperparameter sweep ──────────────────────────────────────────────────

sweep-register:
	$(PYTHON) scripts/sweep.py register --config $(SWEEP_CONFIG)

sweep-agent:
	@test -n "$(SWEEP_ID)" || (echo "usage: make sweep-agent SWEEP_ID=<id>" && exit 2)
	$(PYTHON) scripts/sweep.py agent --sweep-id $(SWEEP_ID) --count $(or $(COUNT),1)

# ── Forward testing ───────────────────────────────────────────────────────

forward-predict:
	$(PYTHON) scripts/forward_test.py predict \
		--checkpoint    $(CHECKPOINT) \
		--model-config  $(MODEL_CONFIG) \
		--paths-config  $(PATHS_CONFIG)

forward-rescore:
	$(PYTHON) scripts/forward_test.py rescore --paths-config $(PATHS_CONFIG)

forward-summary:
	$(PYTHON) scripts/forward_test.py summary

# ── Quality ───────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v

lint:
	ruff check $(PROJECT) scripts tests
	mypy $(PROJECT)

format:
	ruff check --fix $(PROJECT) scripts tests
	ruff format     $(PROJECT) scripts tests

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc"     -delete
