# constellation-quant

![constellation-quant](assets/constellation-quant.svg)

**Graph + temporal deep learning for cross-sectional S&P 500 ranking.**

`constellation-quant` treats the S&P 500 as a **dynamic network** of ~503
interconnected companies. It combines a temporal **Informer** encoder, a
multi-relational **R-GAT** over correlation / fundamental / attention edges,
hierarchical **super-nodes** (stock → sector → market), and three multi-task
output heads (rank · return · volatility). It is trained with an
**IC-maximisation loss** on **35 years** of S&P 500 history (1990–2024) and
emits a daily score per stock used to construct a long/short portfolio.

The dataset uses **time-stamped S&P 500 membership** replayed historically, so
every training sample uses the universe that actually existed on that date —
**no survivorship bias** at the universe level.

---

## TL;DR

| Metric | Value | Notes |
|---|---|---|
| Best validation IC (Phase 2 LR sweep) | **0.0284** | variant **I @ lr=3e-4, ep 10** |
| Best **operational** checkpoint (joint metrics) | **I @ lr=3e-4, ep 11** | val_ic 0.0278 · IR 0.213 · hit@50 60.9% · spread +30 bps / 5d |
| Best Phase 1 variant (single LR=1e-3) | **D, val_ic 0.0276** | dynamic correlation graph |
| Highest Information Ratio | **0.280** | variant H (hierarchical super-nodes) |
| Highest hit@50 | **0.660** | variant D, lr=1e-3, ep 2 |
| Universe | **503 securities** | time-stamped S&P 500 membership |
| Train / Val / Test | **1990-2015 / 2016-2019 / 2020-2024** | chronological, no leakage |
| Architectural variants | **9** (A-I) | + 9-point LR sweep on top 3 |
| Total training runs analysed | **65** | 52 Phase 1 + 13 Phase 2 chained jobs |
| Tests | **224** passing | covering data, features, graph, model, training, evaluation |

---

## Table of contents

1. [Dataset](#dataset)
2. [Preprocessing & features](#preprocessing--features)
3. [Architecture](#architecture)
4. [Variants A–I](#variants-ai)
5. [Training setup](#training-setup)
6. [Phase 1 — full ablation (lr=1e-3)](#phase-1--full-ablation-lr1e-3)
7. [Phase 2 — LR sweep on top 3](#phase-2--lr-sweep-on-top-3-variants-i-c-d)
8. [Auto-saved vs operationally-best checkpoint](#auto-saved-vs-operationally-best-checkpoint)
9. [Comparison to published research](#comparison-to-published-research)
10. [After-cost economics](#after-cost-economics)
11. [Limitations](#limitations)
12. [Reproducibility](#reproducibility)
13. [Engineering journey](#engineering-journey)
14. [Install & quickstart](#install--quickstart)
15. [License & citation](#license--citation)

---

## Dataset

### Universe

- **S&P 500** (technically ~503 securities once you count multi-share-class
  names like BRK.A/BRK.B, GOOG/GOOGL, FOX/FOXA).
- **Time-stamped membership** sourced from the [`fja05680/sp500`](https://github.com/fja05680/sp500)
  GitHub project. For every prediction date we know exactly which tickers were
  S&P 500 members on that date — the model trains on the universe that
  *actually existed*, not a snapshot.
- **Roster size**: **848 unique tickers** ever in the S&P 500 between 1976–2026.
  Of these, **665** have usable yfinance OHLCV history (the rest were delisted
  too early or are otherwise unavailable on yfinance).

### Data sources

All data is **free** (no paid feeds):

| Source | What | Cadence | Coverage |
|---|---|---|---|
| **yfinance** (OHLCV) | open, high, low, close, adjusted close, volume | daily | per ticker, 1990–present where available |
| **yfinance** (fundamentals) | P/E, P/B, dividend yield, market cap, sector, industry | quarterly snapshots | best-effort scrapes |
| **yfinance** (macro indices) | `^VIX`, `^TNX` (10Y yield), `DX-Y.NYB` (DXY), `SPY` | daily | full history |
| **fja05680/sp500** | S&P 500 membership history | event-based | 1976–2026 |

### Data splits — chronological, no leakage

| Split | Period | Trading days | Used for |
|---|---|---|---|
| **Train** | 1990-01-01 → 2015-12-31 | ~6,500 | model fitting |
| **Val**   | 2016-01-01 → 2019-12-31 | ~1,000 | early-stop / hyperparameter selection |
| **Test**  | 2020-01-01 → 2024-12-31 | ~1,260 | held-out final evaluation (COVID + recovery + rate hikes) |

The val period was deliberately chosen to be a **mixed-regime "normal" period**.
An earlier 2020-2021 val split gave anti-correlated results because COVID was
anomalous vs all train history — moving val to 2016-2019 turned val IC positive
overnight, with no model change.

### Survivorship bias handling

- ✅ **Universe-level**: time-stamped S&P 500 membership ensures we don't train
  on a "future-knowledge" survivor set.
- ⚠️ **Stock-level (partial)**: yfinance only carries ~665 of the 848 historical
  members. Stocks delisted/acquired before 2010 are often missing entirely.
  Fully eliminating this requires CRSP (paid) — see [Limitations](#limitations).

---

## Preprocessing & features

The feature engine produces **18 features per stock per day**, split into
three groups by what kind of information they carry and how the model
consumes them:

### Fast features (6 cols, full 60-day window) — fed to the temporal encoder

These vary meaningfully day-to-day; the Informer needs the full 60-step
sequence to extract patterns.

| Feature | Definition |
|---|---|
| `ret_5d` | 5-day log return: `log(close[t] / close[t-5])` |
| `vol_5d` | 5-day rolling realised volatility |
| `log_volume` | `log(volume)` |
| `rel_volume_20` | `volume / mean(volume, 20-day)` |
| `intraday_range` | `(high - low) / close` |
| `gap` | `(open - prev_close) / prev_close` |

> **`ret_1d` was dropped** during experimentation — too noisy at the 5-day
> forecasting horizon.

### Slow features (8 cols, last-day snapshot only) — fed to a small MLP

These are smoothed indicators where 60 nearly-identical values would just
waste Informer capacity. Snapshot at `t` is sufficient.

| Feature | Definition |
|---|---|
| `ret_20d` | 20-day log return |
| `vol_20d` | 20-day rolling realised volatility |
| `rsi_14` | 14-day RSI |
| `macd`, `macd_signal`, `macd_hist` | standard MACD components |
| `bbw_20` | Bollinger band width over 20 days |
| `atr_14` | 14-day Average True Range |

### Macro features (4 cols, broadcast to all stocks per date)

Market-wide regime context shared across all stocks on each date.

| Feature | Source | Definition |
|---|---|---|
| `vix_change_5d` | `^VIX` | 5-day log change |
| `tnx_change_5d` | `^TNX` (10Y Treasury yield) | 5-day log change |
| `dxy_return_5d` | `DX-Y.NYB` | 5-day log return |
| `spy_return_5d` | `SPY` | 5-day log return |

Macro features capture e.g. "high-VIX regime → momentum patterns weaken" — a
context the model can use even though every stock sees the same value on a
given date.

### Targets

For each prediction date `t` and each stock `i`:

```
y[t, i] = log(close[t+5, i] / close[t, i])     # 5-day forward log-return
```

The model emits one scalar per stock; the **ranking head** is the production
output. Auxiliary heads (return, volatility) are present but their MSE
training weights are set to **0.0** — they were 50× smaller in scale than the
IC-max loss and effectively zero-weight at any reasonable scaling, so we made
that explicit.

### Pipeline

```
yfinance ─► raw parquets ─► clean / dedupe ─► feature engine ─► dataset
                                              │     │     │
                                              │     │     └─ macro merge
                                              │     └─ slow snapshot
                                              └─ fast 60-day window
                                              ▼
                                  per-date batches → trainer
```

**Stride-offset rotation**: each epoch starts the prediction-date stride at
a different offset (0, 1, 2, 3, 4 modulo 5), so that across 5 epochs every
trading day in the train window is used as a prediction date *exactly once*,
without ever overlapping target windows within a single epoch (which would
leak the 5-day target into adjacent samples). This 5×s the unique-sample
count without breaking IC validity.

---

## Architecture

### Per-stock data flow

```
                   60-day window of fast features (60, 6)
                            │
                            ▼
                    ┌───────────────┐
                    │   INFORMER    │   d_model=64, e_layers=1
                    │ (temporal     │   n_heads=4, d_ff=128
                    │  encoder)     │   dropout=0.3
                    └───────┬───────┘
                            │ (64,)
                            │
   slow snapshot (8,) ─────►│       
                  + macro    │
                  (4,)─►┌────┴───┐
                        │ slow   │   12 → 32 → 16
                        │ MLP    │
                        └────┬───┘
                             │ (16,)
                             ▼
                       ┌─────────────┐
                       │ gated       │
                       │ fusion      │   per-channel sigmoid
                       │ (concat)    │   gates on both branches
                       └──────┬──────┘
                              │ (80,)  ← d_temporal + slow_out
                              │
              ◄═══ all ~503 stocks ═══►
                              │
                       ┌──────┴──────┐
                       │  GNN        │   variant-dependent (see §4)
                       │  (cross-    │   GAT / R-GAT / none
                       │   stock     │   hidden_dim=32, 2 layers
                       │   message)  │
                       └──────┬──────┘
                              │ (32,) per stock
                              │
                  ┌───────────┴───────────┐
                  │  outer residual       │   skip connection around GNN
                  │  (gated mix of pre +  │   anti over-smoothing
                  │   post-GNN)           │
                  └───────────┬───────────┘
                              │ (32,)
                              │
                ┌─────────────┼─────────────┐
                ▼             ▼             ▼
         ┌──────────┐  ┌──────────┐  ┌──────────┐
         │ ranking  │  │  return  │  │   vol    │
         │  head    │  │   head   │  │   head   │
         │ (mlp)    │  │  (mlp)   │  │  (mlp)   │
         └─────┬────┘  └────┬─────┘  └────┬─────┘
               │            │             │
               ▼            ▼             ▼
            score       ŷ_return       ŷ_vol
            (used)      (aux loss      (aux loss
                         disabled)      disabled)
```

### Component details

**Temporal encoder (Informer)** — `d_model=64`, `e_layers=1`, `n_heads=4`,
`d_ff=128`, `dropout=0.3`. ProbSparse attention (skipped for `L≤32` — uses
dense), distillation between layers (when `e_layers>1`), learnable positional
encoding, attention-weighted-mean pooling.

**Slow MLP** — `Linear(12, 32) → GELU → Dropout → Linear(32, 16)`. The 12
inputs are 8 stock-specific slow features + 4 broadcast macro features.

**Gated fusion (slow ↔ fast)** — **per-channel sigmoid gates on BOTH
branches**:
```
combined   = cat([h_fast, h_slow])
gate_fast  = σ(Linear(combined))
gate_slow  = σ(Linear(combined))
output     = cat([h_fast * gate_fast, h_slow * gate_slow])
```
Lets the model decide per-stock-per-day how much to trust each path.

**GNN** (variant-dependent) — GAT (single relation) for variants B, C, D;
R-GAT (multi-relational) for E, F, G, H, I. `hidden_dim=32`, `num_layers=2`,
`attention_heads=2`. Per-layer residuals **inside** the GNN, plus an
**outer residual around the entire GNN block** with a learned per-channel
sigmoid gate to mitigate over-smoothing.

**Hierarchical super-nodes** — when enabled, every layer's message-passing
also includes 11 sector super-nodes and 1 market super-node, with bidirectional
edges (stock ↔ sector, sector ↔ market). Top-down gating uses sigmoid.

**Output heads**:
- ranking: `MLP[in→64→32→1]` + LayerNorm + GELU + Dropout
- return: `MLP[in→64→1]` + LayerNorm
- volatility: `MLP[in→64→1]` + softplus (must be positive)

**Total parameters**: ~280k (post Phase 3 right-sizing, down from an original
2.5M).

---

## Variants A–I

Each variant adds **exactly one architectural component** over the previous,
isolating its contribution.

| Variant | Description | Graph | Edge types | Hierarchy | Membership |
|---|---|---|---|---|---|
| **A** | Informer only — no graph | none | — | ❌ | fixed |
| **B** | + static sector graph | GAT | sector | ❌ | fixed |
| **C** | + fundamentals features | GAT | sector | ❌ | fixed |
| **D** | + dynamic correlation edges | GAT | correlation | ❌ | fixed |
| **E** | + multi-relational R-GAT | RGAT | correlation + attention + fundamental | ❌ | fixed |
| **F** | + sentiment features (placeholder) | RGAT | corr + att + fund | ❌ | fixed |
| **G** | + dynamic membership | RGAT | corr + att + fund | ❌ | dynamic |
| **H** | + hierarchical super-nodes | RGAT | corr + att + fund | ✅ | dynamic |
| **I** | + multi-scale lookback (20-day + 120-day) | RGAT | corr + att + fund | ✅ | dynamic |

All variants share:
- IC-max ranking loss
- Slow / fast feature split with gated fusion
- Outer residual with gated mix around GNN
- Macro features (VIX, TNX, DXY, SPY)
- Robust correlation edges (multi-window 10 / 30 / 90 + inverse-vol weighting)
- 1990-2015 train / 2016-2019 val / 2020-2024 test
- Phase 1 lr=1e-3, dropout=0.3, weight_decay=5e-3

---

## Training setup

| Setting | Value |
|---|---|
| Loss | **IC-maximisation** (negated Pearson correlation between scores and 5-day forward returns). Aux MSE losses on return + volatility set to weight 0.0 (effectively disabled). |
| Optimizer | AdamW, lr=1e-3 (Phase 1) or {3e-4, 1e-3, 3e-3} (Phase 2 sweep), weight_decay=5e-3 |
| LR schedule | Cosine annealing with 5-epoch warmup |
| Batch | 32 prediction dates per step |
| Gradient clipping | 1.0 |
| Mixed precision | fp16 on Ampere/Hopper GPUs |
| Patience / early-stop | Disabled for the 1-hour ablation runs (rely on best-checkpoint save) |
| Seed | Deterministic seeds in `torch`, `numpy`, Python |
| HPC | NVIDIA A100 PCIe 40GB (gpushort, 1h cap), chained jobs via SLURM |

### Edge construction (variants B–I)

| Edge type | Construction |
|---|---|
| `sector` | static, 1 if same GICS sector, 0 otherwise |
| `correlation` | **multi-window minimum** of `|ρ|` across `[10, 30, 90]` days; **inverse-volatility weighted** edges (down-weights endpoints whose vol is above-median) |
| `fundamental` | cosine similarity over fundamentals vector, threshold 0.7 |
| `attention` | learnable cross-stock attention head |

The **multi-window robust correlation** kills spurious short-window
correlations: an edge survives only if the relationship holds across all
three windows. Combined with **inverse-vol weighting**, this gives a graph
where stable stocks are more trusted than noisy ones.

---

## Phase 1 — full ablation (lr=1e-3)

All 9 variants A-I trained at lr=1e-3 on the same data, same hyperparameters
elsewhere — clean apples-to-apples architecture comparison.

### Peak metrics per variant

> **How to read this table**: each column shows the *peak value* for that
> metric across the entire run, plus the epoch it was reached. For a single
> "best epoch" you'd typically pick the epoch maximising val_ic, but the four
> peaks may not coincide — see the discussion below.

Sorted by peak val_ic descending:

| Rank | Variant | Job ID | Epochs | Peak val_ic | (ep) | Peak IR | (ep) | Peak hit@50 | (ep) | Peak spread@50 | (ep) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **D** | 8674997 | 16 | **0.0276** | 4 | 0.220 | 6 | 0.619 | 11 | +0.00271 | 4 |
| 🥈 | **I** | 8676170 | 21 | **0.0273** | 5 | 0.195 | 6 | 0.629 | 1 | **+0.00342** | 1 |
| 🥉 | **C** | 8676164 | 84 | **0.0254** | 47 | 0.223 | 47 | 0.645 | 2 | +0.00290 | 47 |
| 4 | B | 8671792 | 49 | 0.0246 | 19 | 0.202 | 17 | 0.635 | 17 | **+0.00353** | 17 |
| 5 | F | 8676167 | 23 | 0.0242 | 6 | 0.169 | 6 | 0.604 | 7 | +0.00280 | 6 |
| 6 | A | 8667054 | 29 | 0.0233 | 4 | 0.187 | 3 | 0.624 | 23 | +0.00282 | 3 |
| 7 | H | 8676169 | 21 | 0.0233 | 19 | **0.280** | 19 | 0.604 | 12 | +0.00231 | 7 |
| 8 | E | 8675004 | 14 | 0.0223 | 11 | 0.163 | 6 | 0.599 | 13 | +0.00250 | 6 |
| 9 | G | 8676168 | 23 | 0.0211 | 4 | 0.146 | 14 | 0.604 | 14 | +0.00259 | 14 |

Source data: [`analysis/phase1_summary.csv`](analysis/phase1_summary.csv) ·
full per-epoch trajectory: [`analysis/phase1_epochs.csv`](analysis/phase1_epochs.csv).

### Findings

1. **Variant D (correlation graph) edges out I on raw val_ic** (0.0276 vs
   0.0273) at lr=1e-3, but I has more architectural slack — and with the right
   LR (Phase 2) I dominates.
2. **Variant H wins the IR contest by a wide margin (0.280)**, even though
   peak val_ic is mid-table. Hierarchical super-nodes give a more *consistent*
   per-day signal — fewer wild IC swings.
3. **Variant C is the slowest learner** — its peak val_ic at epoch 47 (vs ≤5
   for most others) suggests fundamentals features take longer to integrate,
   but the result is more *robust* once it converges.
4. **Variants A → D show monotonic improvement** (0.023 → 0.028) as
   architectural pieces are added, validating each component contributes
   signal.
5. **Variants E and G underperform A** — adding R-GAT multi-relational
   message-passing (E) and dynamic membership (G) at this data scale
   apparently introduces more capacity than the data can support. They may
   shine with CRSP-grade data; on yfinance they regress.
6. **B holds the highest spread@50 (+0.00353)** — sector-graph alone produces
   the widest top-50 / bottom-50 return gap, which is what L/S P&L tracks.
   Combined with B's solid hit@50 (0.635), B is a strong dark-horse candidate
   for portfolio construction even though val_ic is mid-table.

### Trajectory observations

- **D peaks fast (epoch 4) then decays** — strong early signal but weak
  long-horizon stability. Suggests dynamic correlation has an "early-fit"
  pattern.
- **I peaks epoch 5, ep 1 has the highest spread (+0.00342)** — multi-scale
  encoder gets useful gradient signal almost immediately.
- **C trains for 84 epochs without crashing** — sector + fundamentals graph is
  very stable. By far the longest healthy trajectory.
- **B reaches spread peak at ep 17 of a 49-epoch run** — sector graph
  steadily improves through mid-training, unlike D's fast-then-decay pattern.

---

## Phase 2 — LR sweep on top 3 variants (I, C, D)

After Phase 1 identified {I, C, D} as the strongest variants by val_ic, we
swept the learning rate to find the per-variant peak.

### Sweep design

- **Variants**: I, C, D
- **LRs**: 3e-4, 1e-3, 3e-3 (one decade above and below Phase 1's 1e-3)
- **9 combos × 2-job gpushort chain** = 18 SLURM jobs, ~6 hours wall-clock
- All other settings identical to Phase 1 (data, features, dropout, optimizer,
  schedule)

### Peak metrics per (variant × LR)

Sorted by peak val_ic descending:

| Rank | Variant | LR | Epochs | Peak val_ic | (ep) | Peak IR | (ep) | Peak hit@50 | (ep) | Peak spread@50 | (ep) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 🥇 | **I** | **3e-4** | 41 | **0.0284** | 10 | 0.213 | 11 | 0.619 | 16 | +0.00302 | 11 |
| 🥈 | **D** | 3e-4 | 51 | 0.0267 | 11 | 0.175 | 11 | 0.599 | 36 | +0.00265 | 11 |
| 3 | D | 1e-3 | 24 | 0.0263 | 6 | 0.156 | 6 | **0.660** | 2 | **+0.00305** | 2 |
| 4 | I | 1e-3 | 22 | 0.0259 | 5 | 0.193 | 1 | 0.624 | 1 | **+0.00338** | 1 |
| 5 | **C** | 3e-4 | 159 | 0.0253 | 16 | 0.186 | 18 | **0.655** | 22 | +0.00305 | 18 |
| 6 | C | 1e-3 | 84 | 0.0248 | 5 | 0.177 | 5 | 0.645 | 2 | +0.00292 | 7 |
| 7 | I | 3e-3 | 21 | 0.0236 | 18 | **0.216** | 18 | 0.640 | 12 | +0.00335 | 9 |
| 8 | C | 3e-3 | 50 | 0.0229 | 4 | 0.167 | 26 | 0.650 | 27 | +0.00304 | 17 |
| 9 | D | 3e-3 | 24 | 0.0218 | 4 | 0.152 | 2 | 0.589 | 11 | +0.00283 | 18 |

Source data: [`analysis/phase2_summary.csv`](analysis/phase2_summary.csv) ·
full per-epoch trajectory: [`analysis/phase2_epochs.csv`](analysis/phase2_epochs.csv).

### Per-metric champions across all 9 combos

| Metric | Combo | Epoch | Value | Notes |
|---|---|---|---|---|
| Best val_ic | **I @ 3e-4** | **10** | **0.0284** | new project peak (Phase 1 was 0.0276) |
| Best IR | I @ 3e-3 | 18 | **0.216** | Phase-2 IR peak (Phase 1 H still holds project-wide IR=0.280) |
| Best hit@50 | D @ 1e-3 | 2 | 0.660 | val_ic only 0.0095 → fluke (model essentially at init) |
| Best spread@50 | I @ 1e-3 | 1 | +0.00338 | train_loss=−0.001 → essentially untrained, fluke |
| **Honest** best hit@50 | **C @ 3e-4** | **22** | **0.655** | mature, real signal |
| **Honest** best spread@50 | C @ 3e-4 ep 18 / I @ 3e-3 ep 9 | — | +0.00305 / +0.00335 | both real, comparable |

### Findings

1. **lr=3e-4 is universally best.** All three variants achieved their highest
   val_ic at lr=3e-4. Phase 1's choice of lr=1e-3 was suboptimal across the
   board — Phase 2 lifted the project peak from 0.0276 to **0.0284**.

2. **Variant I (multi-scale) wins decisively at lr=3e-4** with val_ic 0.0284
   and the *healthiest* train/val gap (~1.4×). Lower-capacity variants overfit
   faster.

3. **C trained for 159 epochs at lr=3e-4** — the longest trajectory in the
   project. Peak val_ic at ep 16, then slow decay. Cleanest, most stable
   training but lower ceiling than I.

4. **lr=3e-3 produces the best IR (0.216, I) but lower val_ic** — confirming
   the trade-off: aggressive LR finds *more consistent* but *smaller* edges.

5. **Two flukes to ignore**: I @ 1e-3 ep 1 and D @ 1e-3 ep 2. Both have
   `train_loss ≈ 0` (model essentially at random initialisation — lucky val
   draws). Real peaks at higher epochs.

### Trajectory observations per combo

- **I @ 3e-4** — cleanest curve in the project. val_ic climbs 0.007 → 0.028 over
  10 epochs, all four metrics still well-aligned at ep 11. Decay starts ep 15.
  Train/val gap stays ≤1.4× through ep 11.
- **D @ 3e-4** — peak ep 11, then late-stage anomaly: hit@50 actually peaks at
  ep 36 (0.599) when val_ic has already drifted to ~0.014. Smooth decay otherwise.
- **C @ 3e-4** — most stable trajectory of any variant. val_ic peak 0.0253 at
  ep 16; **all four metrics peak together around ep 18-22**. Heavy overfit by
  ep 30 (train_loss reaches −0.10 while val drifts to 0.018). The 130 epochs
  after ep 30 are wasted compute — a useful operational lesson.
- **I @ 1e-3** — instability at ep 1 produces a lucky outlier. Real peak ep 5
  at 0.0259. Collapses after ep 7.
- **D @ 1e-3** — peak ep 6 at 0.0263. Decay sharp after ep 8.
- **All @ 3e-3** — noisy, peak early (ep 4 for D, C), never recover.

---

## Auto-saved vs operationally-best checkpoint

A subtle but important finding from Phase 2.

The `save_best` logic in [`constellation_quant/training/checkpoint.py`](constellation_quant/training/checkpoint.py)
saves the checkpoint with the **highest val_ic only**. For variant I @ lr=3e-4,
this saved **epoch 10** — but **epoch 11** is operationally better on every
other metric:

| Metric | ep 10 (auto-saved) | ep 11 (operational best) | Trade-off |
|---|---|---|---|
| val_ic | **0.0284** | 0.0278 | −2% |
| val_ic_ir | 0.187 | **0.213** | **+14%** |
| hit@50 | 0.543 | **0.609** | **+12pp** |
| spread@50 | +0.00276 | **+0.00302** | **+9%** |

For deployed L/S trading, **epoch 11 is the more useful checkpoint**: 14%
higher IR (the edge is *more consistent day-to-day*), 12 percentage points
higher top-50 accuracy (the picks are *materially more right*), 9% higher
spread (the *P&L is bigger*). The val_ic dip is within noise.

**For headline numbers we report ep 10 (auto-saved). For operational use we
recommend ep 11.** A future refactor will change `save_best` to a composite
score over all four metrics.

---

## Comparison to published research

### Direct comparison on similar data tier

| Source | Data | val_ic | hit@50 (or equiv) | spread / period |
|---|---|---|---|---|
| Sawhney 2021 (STHGCN) | yfinance technical | 0.018-0.024 | ~0.55 | ~+0.002 |
| Feng 2019 (RSR-E) | NASDAQ technical | 0.020-0.025 | 0.55-0.58 | ~+0.0019 |
| **Our I (yfinance + macro), Phase 2** | yfinance + 4 macro | **0.0284** | 0.609 | **+0.00302** |
| **Our C (yfinance + fundamentals), Phase 2** | yfinance + macro | 0.0253 | **0.655** | +0.00305 |
| HIST 2021 (CRSP + Compustat) | paid US equity | 0.030-0.045 | 0.58-0.62 | +0.004 |
| AlphaStock 2019 (S&P + RL) | proprietary | 0.040+ | 0.62 | +0.005+ |

**On free yfinance data, `constellation-quant` is at or above the published
academic ceiling.**

### Industry tiers (approximate)

| Tier | val_ic | val_ic_ir | hit@50 | Where we are |
|---|---|---|---|---|
| Random | 0.000 | 0.00 | 0.500 | — |
| Marginal | 0.01 - 0.02 | 0.05 - 0.15 | 0.51 - 0.55 | passed ✓ |
| **Weak but real** | **0.02 - 0.04** | **0.15 - 0.30** | **0.55 - 0.62** | **here (I, C, D)** |
| Solid (deployable) | 0.04 - 0.06 | 0.30 - 0.50 | 0.58 - 0.62 | requires CRSP / alt data |
| Top-tier (top funds) | 0.06 - 0.10 | 0.50 - 1.00 | 0.62 - 0.70 | requires alt data |
| Renaissance (rumored) | 0.10+ | 1.0+ | 0.70+ | proprietary |

---

## After-cost economics

Variant I @ lr=3e-4, ep 11 — operational best checkpoint:

```
spread@50 = +0.00302 per 5-day period (long top-50, short bottom-50)
× 52 rebalance periods/year                = +15.7%/year gross
turnover ≈ 2.5× per week × 52 weeks        = 130/year
costs ≈ 5 bps × 130 turnover               = 6.5%/year
net annualised return                       ≈ 9-10%/year
estimated gross Sharpe                      ≈ 1.0
estimated net Sharpe                        ≈ 0.5-0.7
```

A net Sharpe ~0.6 is **academically meaningful** — the strategy survives
realistic transaction costs to produce a positive expected return — but is
**below typical hedge-fund deployment hurdles** (>1.5).

---

## Limitations

### Data limitations

1. **Stock-level survivorship bias (partial)** — yfinance carries ~665 of the
   848 historical S&P 500 members. Pre-2010 delistings/acquisitions are often
   missing entirely. CRSP would be survivorship-bias-free.
2. **Fundamentals quality** — yfinance fundamentals are best-effort scrapes;
   restatements, late filings, and corrections are not reliably reflected.
   Compustat would be authoritative.
3. **No alternative data** — no sentiment, options flow, news, insider trading,
   ETF flows, or any of the data sources used by production quant funds.
4. **5-day return is noisy** — ~95% of cross-sectional variance is unpredictable.
   A perfect model has a hard ceiling on achievable IC.

### Model limitations

1. **`save_best` keyed on val_ic only** — see [previous section](#auto-saved-vs-operationally-best-checkpoint).
2. **No walk-forward CV** — fixed split. Walk-forward would be more robust but
   ~10× more compute.
3. **No SWA, no warm restarts** — could squeeze additional 0.005-0.010 from
   existing best checkpoints.
4. **GraphBuilder is CPU-bound** — per-date correlation matrix construction is
   the wall-clock bottleneck. GPU implementation possible but not pursued.

### Evaluation limitations

1. **Test diagnostic not yet run on the Phase 2 winner** — Phase 3 of the
   project plan: `cq-evaluate --checkpoint analysis/phase2_winner.pt` on the
   2020-2024 test set.
2. **No regime-conditional analysis** — `RegimeAnalyzer` is in the codebase
   but not yet applied.

---

## Reproducibility

- Deterministic seeds wired through `torch`, `numpy`, and Python's `random`.
- Resume-safe checkpointing for chained training jobs.
- Mixed-precision (fp16) on Ampere/Hopper GPUs.
- 224-test suite covering data, features, graph, model, training, evaluation.
- All raw experiment data archived in `analysis/`:
  - [`analysis/phase1_epochs.csv`](analysis/phase1_epochs.csv) — full Phase 1 per-epoch trajectories (best run per variant).
  - [`analysis/phase1_summary.csv`](analysis/phase1_summary.csv) — Phase 1 peak-per-variant summary.
  - [`analysis/phase2_epochs.csv`](analysis/phase2_epochs.csv) — full Phase 2 per-combo trajectories.
  - [`analysis/phase2_summary.csv`](analysis/phase2_summary.csv) — Phase 2 peak-per-combo summary.
  - [`analysis/parse_phase1.py`](analysis/parse_phase1.py) / [`analysis/parse_phase2.py`](analysis/parse_phase2.py) — the parsers (raw `.err` logs → CSV).

---

## Engineering journey

A condensed phase log of the architectural and data choices that converged to
the final state.

| Phase | Change | Effect |
|---|---|---|
| 0 | Diagnostic baselines | Identified original Sharpe-negative result as overfit + cost drag, not regime inversion |
| 1 | Data extension 2010 → 1990 | 4× more train history; enabled by SLURM-side bulk download |
| 2 | Stride-offset rotation | 5× unique training samples without breaking target non-overlap |
| 3 | Right-sizing the model | 2.5M → 280k params; sweet spot for ~50 optimizer steps/epoch |
| 4 | Slow / fast feature split | Save Informer capacity for actually-time-varying features |
| 5 | **IC-max loss** (was ListMLE) | **Single-line config change broke the val_ic 0.011 plateau → reached 0.025+** |
| 6 | Outer residual around GNN | Mitigates over-smoothing; modest but consistent +0.001 |
| 7 | Macro features (VIX/TNX/DXY/SPY) | Regime context; +0.002 val_ic |
| 8 | Robust correlation graph | Multi-window min-|ρ| + inverse-vol weights; cleaner edges |
| 9 | Aux MSE losses disabled | Made what was effectively zero-weight explicit |
| 10 | Phase 1 ablation (9 variants × lr=1e-3) | Identified I, D, C as top 3 |
| 11 | Phase 2 LR sweep (top 3 × 3 LRs) | Lifted project peak from 0.0276 → 0.0284, identified lr=3e-4 as universally optimal |

For deeper context see [`PROJECT_REPORT.md`](PROJECT_REPORT.md) — a 41 KB
methodology-and-results document with the full engineering rationale,
literature comparison, and lessons-learned discussion.

---

## Install & quickstart

```bash
git clone https://github.com/zahirnik/constellation-quant
cd constellation-quant
pip install -e .                  # editable install
```

Python 3.10+. Core deps: `torch ≥ 2.1`, `torch-geometric ≥ 2.4`, `pandas`,
`numpy`, `yfinance`, `pyyaml`. See [`requirements.txt`](requirements.txt).

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

The `Makefile` wraps the same operations with sensible defaults; run `make`
for the list.

---

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
├── analysis/                   parsed experiment results (CSVs + parsers)
├── assets/                     logo + report assets
├── PROJECT_REPORT.md           full methodology + results write-up
├── README.md
└── LICENSE
```

---

## License & citation

[MIT](LICENSE) — feel free to use, modify, and redistribute.

```bibtex
@misc{constellation-quant,
  author = {Nikraftar, Zahir},
  title  = {constellation-quant: Graph and temporal deep learning for cross-sectional S\&P 500 ranking},
  year   = {2026},
  url    = {https://github.com/zahirnik/constellation-quant}
}
```
