# constellation-quant

![constellation-quant](assets/constellation-quant.svg)

**Graph + temporal deep learning for cross-sectional S&P 500 ranking.**

`constellation-quant` treats the S&P 500 as a dynamic network of ~503 interconnected
companies. It combines a **temporal Informer encoder** (ProbSparse attention,
O(n log n)) with a **multi-relational R-GAT** over correlation, fundamental, and
learned-attention edges, plus optional **hierarchical super-nodes**
(stock → sector → market). The model emits a daily ranking of stocks by expected
5-day forward return, which a backtester translates into a long/short portfolio.

The dataset uses time-stamped S&P 500 membership (replayed historically) so every
training sample uses the universe that actually existed on that date — **no
survivorship bias** at the universe level.

---

## Headline results

| Metric | Value | Notes |
|---|---|---|
| Best validation IC | **0.0284** | variant I @ lr=3e-4, ep 10 |
| Best operational checkpoint | **I @ lr=3e-4, ep 11** | val_ic 0.0278 · IR 0.213 · hit@50 60.9% · spread +30 bps / 5d |
| Best hit@50 (any variant) | 0.655 | C @ lr=3e-4, ep 22 |
| Best IC consistency (IR) | 0.280 | H (hierarchical) — Phase 1 |
| Train / Val / Test | 1990–2015 / 2016–2019 / 2020–2024 | chronological, no leakage |
| Universe | 503 securities | time-stamped membership |
| Architectural variants tested | 9 (A–I) | + 9-point LR sweep on top 3 |

Detailed methodology, ablation findings, per-epoch trajectories, and comparison
against published research are in [`PROJECT_REPORT.md`](PROJECT_REPORT.md).

---

## Install

```bash
pip install -e .                  # editable install
```

Python 3.10+. Core deps: `torch ≥ 2.1`, `torch-geometric ≥ 2.4`, `pandas`,
`numpy`, `yfinance`, `pyyaml`. See [`requirements.txt`](requirements.txt).

## Quickstart

```bash
# 1. Download S&P 500 prices, fundamentals, macro series, membership roster
cq-download

# 2. Train one architectural variant (A through I)
cq-train --variant A

# 3. Evaluate a checkpoint on the held-out test period
cq-evaluate --checkpoint path/to/best.pt

# 4. Run the full 9-variant ablation
cq-ablation

# 5. Generate the HTML/PDF report
cq-report
```

The `Makefile` wraps the same operations with sensible defaults; run `make` for
the list.

## Architecture

```
60-day window of fast features (60, 6)
        │
        ▼
┌───────────────┐         ┌──────────┐
│   INFORMER    │─────────►          │
│ temporal      │  fast   │  gated   │
│ encoder       │  (64)   │  fusion  │
└───────────────┘         │          │
                          │          │
slow snapshot (8)         │          │
+ macro      (4) ────────►│          │
                          └────┬─────┘
                               │ (80)
              ◄══ 503 stocks ══►
                               │
                          ┌────▼─────┐    ┌──────────────┐
                          │  R-GAT   │ ←→ │  hierarchy   │
                          │ + outer  │    │ (sector,     │
                          │ residual │    │  market)     │
                          └────┬─────┘    └──────────────┘
                               │ (32)
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
              ranking       return        vol
              (used)        (aux)        (aux)
```

### Key components

- **Informer** temporal encoder (d_model=64, 1 layer, 4 heads).
- **Slow / Fast feature split** — fast signals into the 60-day Informer,
  slow technicals (RSI, MACD, ATR …) snapshot-only into a small MLP, fused with
  per-channel sigmoid gates.
- **Macro broadcast** — VIX, 10Y yield, DXY, SPY 5-day changes shared across all
  stocks per date.
- **Robust correlation graph** — multi-window minimum |ρ| across [10, 30, 90]
  days, inverse-volatility weighting on edges.
- **Outer residual around the GNN** with learned per-channel gate to mitigate
  over-smoothing.
- **IC-maximisation loss** (negated Pearson correlation), directly optimising
  the metric the model is evaluated on.

## Repository layout

```
constellation-quant/
├── configs/                    YAML — model, training, data, ablation
├── constellation_quant/        Python package
│   ├── data/                   dataset, downloaders, macro, membership
│   ├── features/               feature engine (fast + slow + macro)
│   ├── graph/                  edge builders (correlation, sector, fund.)
│   ├── models/                 temporal · GNN · output heads · master model
│   ├── training/               trainer · losses · validator · checkpoint
│   ├── evaluation/             backtester · metrics · regime analyser
│   └── ablation/               variant generator
├── scripts/                    CLI entry points (download / train / evaluate / …)
├── tests/                      224 tests, all passing
├── assets/                     logo + report assets
├── PROJECT_REPORT.md           full methodology + results write-up
├── README.md
└── LICENSE
```

## Reproducibility

- Deterministic seeds wired through `torch`, `numpy`, and Python.
- Resume-safe checkpointing for chained training jobs.
- Mixed-precision (fp16) on Ampere/Hopper GPUs.
- 224-test suite covering data, features, graph, model, training, evaluation.

## Limitations

`constellation-quant` reaches the information ceiling of free yfinance data
(val_ic ~0.025–0.028). It is **not** a deployable trading strategy at typical
hedge-fund hurdles. Net Sharpe after realistic transaction costs is in the
0.5–0.7 range — competitive with academic finance ML papers using comparable
data, short of production strategies that use CRSP, Compustat, sentiment, and
options-implied features. See [`PROJECT_REPORT.md`](PROJECT_REPORT.md) §9 for
the full honest framing.

## License

[MIT](LICENSE) — feel free to use, modify, and redistribute.

## Citation

```bibtex
@misc{constellation-quant,
  author = {Nikraftar, Zahir},
  title  = {constellation-quant: Graph and temporal deep learning for cross-sectional S\&P 500 ranking},
  year   = {2026},
  url    = {https://github.com/zahirnik/constellation-quant}
}
```
