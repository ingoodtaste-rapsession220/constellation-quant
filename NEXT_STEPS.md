# constellation-quant — Next Steps Plan

A self-contained planning document for the next phase of work on
constellation-quant. Read this with [`README.md`](README.md) and
[`PROJECT_REPORT.md`](PROJECT_REPORT.md) for the complete picture of where
the project is and what comes next.

---

## 1. Where the project stands today

The methodology side is solid. The data side is the binding constraint.

- **Validation result (2016–2019)**: Best variant `I @ lr=3e-4` reaches val
  IC = **0.0284** with hit@50 = 60.9% and spread@50 = +30 bps / 5d.
  Competitive with academic finance ML on free yfinance data.
- **Held-out test result (2020–2024)**: All three top variants (I, D, C at
  lr=3e-4) come back with **test IC ≈ 0** (t-stats below 0.5). The val
  signal does not survive the regime shift.
- **Diagnosis**: This is honest overfitting, not a pipeline bug. Score
  distributions are healthy (range/std 7.9–16, well above the ListMLE
  collapse threshold of ~3). The architecture is sound; the data is the
  binding constraint.
- **Implication**: To break out of this regime, the project needs richer
  data. Architectural tweaks alone won't move the test IC.

The current free-data ceiling (yfinance OHLCV + scraped fundamentals + 4
macro series + technical indicators) appears to be **~0.025–0.028 in-sample,
≈ 0 out-of-sample** for cross-sectional 5-day return prediction on the
S&P 500.

---

## 2. The publication objective

The goal of the next phase is to produce results strong enough to publish in
a venue that hedge funds and quant-finance industry actually read. This
means:

- **Out-of-sample test IC must be statistically significant** (t-stat > 2,
  ideally > 3) — not just positive.
- **Results must hold across regime shifts** (the 2020–2024 test window
  contains COVID, recovery, rate hikes, tech boom/bust). A model that only
  works in trending markets is not publishable.
- **Methodology must be defensible** — proper walk-forward CV is preferred
  over fixed splits; ablations must isolate component contributions cleanly;
  baselines must include the obvious comparators (LSTM, static GNN, TGN,
  TGAT, MLP).
- **After-cost economics must be reported honestly** — not gross returns
  alone.

### Target venues (ranked)

The target venues are the same ones the comparison papers in §3 published
in. Anything in this list is read by quant funds and industry research
teams.

| Venue | Tier | Read by |
|---|---|---|
| **NeurIPS / ICML / ICLR** | Top ML | Industry research labs, quant teams |
| **AAAI / KDD / WWW / IJCAI** | Top AI/data mining | Quant funds, fintech research |
| **ACM TOIS / TKDD / TKDE** | Top journals | Industry, academia |
| **Quantitative Finance** (Taylor & Francis) | Top finance journal | Hedge funds, asset managers |
| **Journal of Financial Data Science** | Specialised | Industry quant teams |
| **European Journal of Operational Research** | Q1 finance/OR | Quant industry |

### What needs to change to clear that bar

The current val IC of 0.0284 is competitive with published academic finance
ML on free data; the OOS-test IC of 0 is not. To clear the publication bar,
realistic targets after the data upgrade:

| Metric | Current (val) | Current (test) | Target (val) | Target (test) |
|---|---|---|---|---|
| Mean IC | 0.0284 | 0.003 | 0.045–0.060 | 0.025–0.040 |
| t-stat | — | 0.3 | — | > 3 |
| Hit@50 | 0.609 | 0.482 | 0.62–0.65 | 0.55–0.58 |
| Realistic Sharpe (after costs) | ~1.0 estimated | ~0 realised | 1.5–2.0 | 0.8–1.2 |

Hitting these requires data the project doesn't currently have. The rest of
this document explains which data sources, why, and in what order.

---

## 3. Comparison papers — what they did, what they got, with what data

The six papers most directly relevant for benchmarking constellation-quant.
All used **free / public data** (yfinance, Yahoo Finance, public OHLCV) and
were published in venues that quant industry takes seriously.

| # | Paper | Venue | Data | Architecture | Headline result |
|---|---|---|---|---|---|
| 1 | **Feng, Chen, He, Ding, Sun, Chua. 2019. "Temporal Relational Ranking for Stock Prediction."** | ACM TOIS | NASDAQ + NYSE Yahoo Finance, 2013–2017 | LSTM + Relational Stock Ranking (RSR-E / RSR-I), graph from sector + Wikidata | NDCG@5 ≈ 0.35–0.40 · IRR ≈ +0.20 cumulative |
| 2 | **Sawhney, Agarwal, Wadhwa, Shah. 2021. "STHAN-SR — Stock Selection via Spatiotemporal Hypergraph Attention Network."** | AAAI 2021 | NASDAQ + NYSE Yahoo Finance | Hypergraph attention + GRU, learning-to-rank | Sharpe ≈ 1.0–1.5 · IRR ≈ +0.25–0.40 · NDCG@5 ≈ 0.40 |
| 3 | **Yoo, Soun, Park, Kang. 2021. "DTML — Accurate Multivariate Stock Movement Prediction via Data-Axis Transformer."** | KDD 2021 | S&P 500 + NASDAQ + NIKKEI from Yahoo Finance | Data-axis Transformer with cross-stock attention | Accuracy ≈ 58–60% · MCC ≈ 0.10–0.13 |
| 4 | **Hou, Wang, Zheng, Yang, Wang, Wang. 2021. "REST — Relational Event-driven Stock Trend Forecasting."** | WWW 2021 | NASDAQ + NYSE Yahoo Finance + financial news | Relational graph + event extraction from news | F1, IRR, Sharpe; news events lift baseline by ~3–5 pp |
| 5 | **Kim, Cho, Choi, Lee, Park, Lee. 2019. "HATS — Hierarchical Graph Attention Network."** | NeurIPS 2019 RAFS workshop | S&P 500 + KOSPI from Yahoo Finance + Wikidata | Hierarchical GAT on price + relational graph | Accuracy ≈ 57% · F1 ≈ 0.55 |
| 6 | **Krauss, Do, Huck. 2017. "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage on the S&P 500."** | European Journal of Operational Research | S&P 500 OHLCV (Yahoo Finance), 1992–2015 | Deep MLP / GBT / RF ensemble | Top-decile spread ≈ 0.46% / day gross; net negative after costs |

### What this comparison reveals

- **In-sample on free data, the bar is around NDCG@5 ≈ 0.35–0.40 (val IC ≈
  0.020–0.030) and Sharpe ≈ 1.0–1.5.** constellation-quant sits cleanly in
  this band.
- **Most of these papers do not run a strict OOS regime-shift test**, or
  they report results on a held-out window that's similar to the train
  period (no COVID-grade regime change).
- **The papers that bring in extra data sources beyond OHLCV** — Wikidata
  relations (Feng, Sawhney, Kim), news (Hou) — get noticeable lifts in
  in-sample metrics. None of them solve the OOS regime shift problem on
  yfinance alone.
- **Krauss 2017 is the most honest** — they explicitly show that gross
  returns disappear after realistic transaction costs. constellation-quant's
  Phase 3 result is the same finding in a different frame.

### Implication

Cross-sectional ML on free OHLCV-only data hits a wall at val IC ≈ 0.03
and OOS Sharpe ≈ 0–0.5 *regardless of architecture*. The path past that
ceiling is more-and-better data, not bigger models. The next sections list
the data sources that the higher-IC papers used and that constellation-quant
doesn't yet incorporate.

---

## 4. Free data sources we should add (ranked by expected lift)

All of these are free; none require paid feeds. Ranked by the realistic lift
each would bring to test IC, given the project's current state.

### 4.1 SEC EDGAR full-text 10-K / 10-Q filings — **highest priority**

**Why**: Cohen, Malloy & Pomorski (2020), "Lazy Prices" (Journal of Finance,
top-3 finance journal), is the most-cited evidence that filing language
drift is one of the strongest single signals for forward stock returns.
Companies that change their 10-K language relative to their own historical
baseline experience predictably lower returns over the following months.
The signal is well-documented, free to use, and not yet in
constellation-quant.

**What it is**: Mandatory annual (10-K) and quarterly (10-Q) filings
submitted to the SEC by every US public company. Full text, machine-readable,
free via the EDGAR API.

**What we'd extract**:
- **Item 1A (Risk Factors)** — explicit risk language, year-over-year
  diff for the language-drift signal.
- **Item 7 (MD&A)** — management discussion + sentiment.
- **Document-level embeddings** for cosine-distance drift across quarters.
- **Topic-level summary** of risk factors (with an LLM, optional).

**Volume**: ~503 companies × 30 years × ~4 filings/year = ~60,000 filings
(roughly 30 GB raw, parses down to ~5 GB structured text).

**Status in literature**: Used by REST 2021 (WWW) for event extraction,
by GraphAlpha-core / FinGraph (sister project) for the language-drift
signal. Foundational for any modern finance ML system that wants to claim
"text-aware".

### 4.2 Public earnings call transcripts — **second priority, complementary to filings**

**Why**: Same family of signal as filings (language drift, sentiment, tone),
but on **spoken** management commentary rather than written, formal
documents. Transcripts capture management hedging, uncertainty, and tone
shifts that the formal filings smooth out. Multiple academic papers have
shown that transcript language adds incremental signal *on top of* filings.

**What it is**: Quarterly earnings call transcripts. Free sources include
Seeking Alpha (the standard), Yahoo Finance (summaries only), and audio
recordings from company IR pages (transcribable with a Whisper-class model).

**Caveat**: Transcripts are **complementary** to 10-K/10-Q, not a
substitute. The Lazy Prices effect is anchored on filings. Add transcripts
*after* filings are integrated, not before.

**Volume**: ~503 companies × ~4 earnings calls/year × 25 years ≈ 50,000
transcripts.

**Status in literature**: Used by GraphAlpha-core for the
quarter-over-quarter management-tone-drift signal, extending the Lazy Prices
idea to spoken text.

### 4.3 Wikidata corporate relations — **third priority**

**Why**: constellation-quant's variants E–I currently have multi-relational
R-GAT capacity but operate on only three thin edge types (correlation,
fundamentals similarity, learnable attention). Adding richer edge types —
supply chain, parent-subsidiary, board interlocks, executive movements,
investor overlap — would let those variants demonstrate the architecture's
true potential. Multiple comparison papers (Feng 2019, Kim 2019,
Sawhney 2021) saw 2–4 pp lifts from Wikidata-derived relations.

**What it is**: Free SPARQL queries against [wikidata.org](https://wikidata.org)
return structured corporate relations:
- Subsidiary / parent (`P749` — parent organisation)
- Board members (`P3320`, `P488`)
- Industry / sector (`P452`)
- Owner of (`P127`)
- Founder (`P112`)
- Headquartered in (`P159`) — for geographic clustering

**Volume**: One query per ticker, results in ~10–50 KB JSON each. Total ~25
MB of relational data.

**Effort**: Medium — Wikidata coverage of US public companies is good but
not perfect; needs entity resolution against the ticker symbols.

### 4.4 Free financial news / RSS — **fourth priority, event-driven**

**Why**: Event extraction from news headlines is a well-studied signal.
Papers like REST 2021 (WWW) showed news events lift in-sample metrics by
3–5 pp over price-only baselines. The signal is strongest around specific
event types — earnings beats/misses, analyst rating changes, M&A, executive
transitions, regulatory actions.

**What it is**: Free news streams:
- **Yahoo Finance RSS** — per-ticker headlines
- **Google News RSS** — query-based
- **FinViz news** — aggregated financial news
- **Reddit / public Bloomberg headlines** — supplementary

**Caveats**: Noisy. Headlines decay fast (within hours/days). Quality
varies. The historical archive going back to 1990 is patchy.

**Volume**: Hard to estimate — depends on coverage and dedup. Modern
period (2010+) is rich; earlier periods are thin.

### 4.5 SEC Form 4 insider trading filings — **fifth priority**

**Why**: Insider buys (especially cluster buys by multiple insiders) have
well-documented predictive power for forward returns. The signal is strong
but **narrow** — it only fires when there's actual insider activity. Many
trading days have no Form 4 filings for a given ticker.

**What it is**: Free via SEC EDGAR. Form 4 filings disclose insider
transactions (purchases, sales, option exercises). Structured data; easy to
parse.

**Volume**: ~10,000–50,000 filings per year across the S&P 500.

**Effort**: Low — same EDGAR infrastructure as 10-K/10-Q. Probably a
~100-line add-on to the SEC EDGAR client we'd build for §4.1.

### 4.6 Broader FRED macro series — **sixth priority**

**Why**: We already have 4 macro series (VIX, 10Y yield, DXY, SPY). FRED
hosts thousands more, free. The marginal signal from adding more macro
series is small relative to filings or news, but the cost is also small.

**What to add**:
- Term-structure spreads (10Y–2Y, 10Y–3M)
- Credit spreads (Moody's Baa-Aaa, BAML high-yield OAS)
- Unemployment, ISM Manufacturing PMI
- M2 money supply, Fed funds rate
- OFR Financial Stress Index
- Commodity prices (WTI crude, gold)

**Volume**: A few hundred KB per series, decades of history.

**Effort**: Low — a few-hundred-line script.

### 4.7 Wikipedia page views and edit volumes — **lowest priority**

**Why**: Attention proxy. Some papers report a weak signal; most find it
marginal once price-volume features are already in the model. Worth trying
only after the higher-priority sources are in place.

**What it is**: Free via the Wikimedia API.

---

## 5. Implementation plan — phased delivery

### Architecture

The data-upgrade work integrates into the existing constellation-quant
codebase as new modules:

```
constellation_quant/
├── data/
│   ├── edgar.py              [NEW] SEC EDGAR client (rate-limited, resume-safe)
│   ├── filings_parser.py     [NEW] HTML → text, extract Item 1A + Item 7
│   ├── transcripts.py        [NEW] Earnings call ingestion
│   └── ...existing modules
├── nlp/                      [NEW package]
│   ├── __init__.py
│   ├── embeddings.py         bge-m3 wrapper (batched, GPU)
│   ├── sentiment.py          FinBERT wrapper (sentence-level financial sentiment)
│   ├── drift.py              Q-over-Q cosine drift features
│   ├── extraction.py         (optional) Qwen 2.5 14B structured extraction
│   └── pipeline.py           orchestrate the above on a batch of filings
├── graph/
│   └── wikidata_edges.py     [NEW] Wikidata SPARQL → multi-relational edges
└── scripts/
    ├── download_filings.py   CLI for SEC EDGAR download
    ├── parse_filings.py      CLI for filing parsing
    ├── compute_nlp_features.py  CLI for embeddings + sentiment + drift
    ├── fetch_wikidata.py     CLI for Wikidata relations
    └── slurm/
        ├── download_filings.sh    CPU job, long queue
        ├── parse_filings.sh       CPU job
        ├── compute_nlp.sh         GPU job (A100)
        └── fetch_wikidata.sh      CPU job
```

### Phases

| Phase | Scope | Lines (est.) | Wall-clock to write & test | HPC time on full universe | Priority |
|---|---|---|---|---|---|
| **A** | SEC EDGAR client + filings parser + storage layer | ~600 | one focused session | 4–8 h (rate-limited at 10 req/s, ~60k filings) | Foundational — do first |
| **B** | bge-m3 embeddings + Q-over-Q drift features | ~300 | ~1 h | ~1 h on A100 (60k docs batched) | High — main signal |
| **C** | FinBERT sentiment scoring + slow-feature integration | ~200 | ~30 min | ~30 min on A100 | High |
| **D** | Integration into `constellation_quant.data.dataset` as new slow features; retrain top variants; Phase-3 OOS test on new model | ~200 | ~1 h | depends on training time (~6 h chained jobs) | High — this is where we measure the lift |
| **E** | (Optional) Qwen 2.5 14B structured extraction on MD&A and Risk Factors only | ~400 | ~1.5 h | **~12+ h on A100** | Optional — defer until A–D show clear lift |
| **F** | Wikidata SPARQL client + multi-relational edge integration into variants E–I | ~300 | ~1 h | ~30 min | Medium — second-wave signal |
| **G** | Free news RSS ingestion + event extraction features | ~400 | ~2 h | ongoing | Lower — noisy data |
| **H** | SEC Form 4 insider trading features | ~200 | ~1 h | ~1 h | Lower |
| **I** | Broader FRED macro series | ~150 | ~30 min | ~10 min | Lowest — marginal lift |
| **J** | Earnings call transcripts (Seeking Alpha + Whisper for audio) | ~500 | ~3 h | varies | Lower — complement, not foundational |
| **K** | Wikipedia views/edits | ~100 | ~30 min | ~10 min | Lowest |

**Recommended initial scope**: Phases A → B → C → D. That's ~1300 lines of
code, around half a day of focused engineering, ~10 h of HPC compute, and
delivers a complete end-to-end pipeline with the highest-priority signal
(filings language-drift + sentiment).

After A–D produces measurable test-IC lift (the success metric for
publication), expand into F (Wikidata edges), then G/H (news, insider
trading), then E (LLM extraction) for nuanced features, then J (transcripts).

### Storage budget

Within the cluster's 436 GB project headroom:

| Data | Approx size |
|---|---|
| Raw 10-K/10-Q filings (60k × 500 KB cleaned) | ~30 GB |
| Embeddings (60k × 1024 × fp32) | ~250 MB |
| Sentiment scores | ~50 MB |
| Wikidata relational data | ~25 MB |
| FRED macro series | ~50 MB |
| News headlines (modern only) | ~5 GB |
| **Total Phase A–I** | **~36 GB** |
| Earnings transcripts (Phase J, raw text) | ~20 GB |
| Audio (Phase J, optional) | ~500 GB — exceeds quota |

**Decision point**: do not download earnings call audio. Use text
transcripts only (Seeking Alpha, where available). Audio + Whisper would
require either a separate scratch filesystem or selective per-call
processing.

### Hardware requirements

| Phase | Hardware |
|---|---|
| A (download) | CPU only, long queue |
| B (embeddings) | A100 40GB; bge-m3 fits comfortably at fp16 |
| C (sentiment) | A100 40GB or even V100; FinBERT is ~440 MB |
| D (training) | A100 40GB on gpushort, chained jobs (existing infrastructure) |
| E (LLM extraction) | A100 40GB at fp16 for Qwen 2.5 14B (~28 GB), or 4-bit quant on V100 16GB (~7 GB) |
| F, G, H, I, K | CPU only |
| J (transcripts text) | CPU; (audio + Whisper would need GPU) |

---

## 6. Model choices for the NLP pipeline

| Job | Model | Size | Why |
|---|---|---|---|
| Document embeddings (drift signal) | `BAAI/bge-m3` | ~600M params, ~2 GB | Top of MTEB leaderboard, multilingual, 8K context, very fast on A100 |
| Financial sentiment (sentence-level) | `ProsusAI/finbert` | ~110M params, ~440 MB | Industry-standard finance NLP baseline, pretrained on Reuters financial news |
| Structured extraction (optional, Phase E) | `Qwen/Qwen2.5-14B-Instruct` | 14B params, ~28 GB at fp16 | Best balance of quality and inference cost in the 7–15B range; 128K context handles long filings |

**Note on cost**: do **not** run a 14B LLM on every filing's full text. The
pipeline should be:

1. bge-m3 → full-document embeddings → drift features (cheap)
2. FinBERT → sentence-level sentiment over selected sections (cheap)
3. (Optional, Phase E) Qwen → structured extraction *only* on
   Item 1A and Item 7 sections, *only* on filings that show high cosine
   drift (selective, not exhaustive)

This keeps total inference cost under 24 hours of A100 time across the
entire universe.

---

## 7. Success criteria

The Phase A–D pipeline succeeds if, after retraining the top variants with
the new filing-derived features:

- **Test IC moves from ~0 to > 0.015 (t-stat > 2)** on the 2020–2024 test
  set for at least one variant.
- **Hit@50 on test set rises by at least 2 pp** above the current ~48% (so,
  > 50% true).
- **Per-half-year regime breakdown shows positive IC in at least 7 of 10
  buckets** (currently 5 of 10 for variant I).

If any of these are met, the project is publishable in a tier-2 venue
(KDD/IJCAI/AAAI workshop or finance journal). All three are needed for a
top-tier conference (NeurIPS/AAAI/KDD main track) or top-1 finance journal
submission.

If Phase A–D does **not** clear these bars, the next move is one of:

1. Add Phase F (Wikidata multi-relational edges) and re-test.
2. Add Phase E (LLM structured extraction on filings) and re-test.
3. Apply for **CRSP / Compustat access via WRDS** (Queen Mary subscription;
   the user has an application in flight) and rebuild on paid data. This
   should comfortably clear the bar based on what HIST 2021 and AlphaStock
   2019 achieved.

---

## 8. How a new chat should pick this up

This document is self-contained. A new conversation can take over with the
following context:

### Read first

1. This file (`NEXT_STEPS.md`).
2. [`README.md`](README.md) — overall project state, headline numbers,
   methodology summary.
3. [`PROJECT_REPORT.md`](PROJECT_REPORT.md) — full results, especially
   §5 (Phase 1 ablation), §5.5 (Phase 2 LR sweep), and the test-results
   subsection (Phase 3, OOS confirms overfitting).
4. [`analysis/test_ic_results.csv`](analysis/test_ic_results.csv) — the
   raw OOS test diagnostic output that motivates this whole plan.

### Decision points the user will need to make

- **Which phase to start with?** Recommendation: A → B → C → D as the
  minimal end-to-end sequence. Stop after D, evaluate, then decide whether
  to keep going.
- **Optional Phase E (LLM extraction with Qwen 2.5 14B)?** Defer — only do
  this if A–D produces measurable test-IC lift and there's a clear gap that
  structured extraction would fill.
- **Should we ship intermediate phases to GitHub?** Yes — each phase
  produces a self-contained subsystem (download script, parser, NLP
  pipeline). Push each phase as a PR-grade commit so the public repo's
  Engineering Journey continues to grow.

### What the new chat should NOT do

- **Don't try to do everything at once.** The full plan is ~3500 lines of
  code across 11 phases. Doing it in one session produces low-quality
  output. Phase-by-phase is the right pace.
- **Don't skip Phase A.** Without filings downloaded, none of B/C/D/E work.
  Phase A is the foundation.
- **Don't run Qwen 14B on every filing's full text.** The cost is
  prohibitive (would take weeks of A100 time). Use the embedding + sentiment
  + selective-LLM pattern described in §6.
- **Don't add architectural changes to the model in this phase.** The
  goal is to feed *better data* into the existing architecture, not to
  redesign the architecture. Architecture changes should come *after*
  the data upgrade, not before.

### What the new chat should produce

For each phase the new chat tackles:

1. **The code** — well-documented, tested, integrated into
   `constellation_quant/` as new modules.
2. **A SLURM submission script** under `scripts/slurm/` that runs the
   phase's compute cleanly.
3. **A short progress note** appended to this document under §10
   (Progress log).
4. **Updates to `PROJECT_REPORT.md`** when a phase produces new
   measurable results.
5. **Commits and pushes** to the public GitHub repo so the engineering
   journey stays current.

---

## 9. Quick reference — what to do right now

If you just want to get started:

```bash
# 1. SSH to the HPC cluster
ssh acw720@login.hpc.qmul.ac.uk
cd /data/EECS-Theory/Zahir_DAYNGRAPH500

# 2. Pull latest code (the new modules will be added here)
git pull   # if the repo is checked out on the cluster
# (otherwise: rsync from the local repo)

# 3. Begin Phase A by asking the next chat to "implement Phase A from
#    NEXT_STEPS.md — SEC EDGAR client + filings parser, with a SLURM
#    submission for downloading the full S&P 500 universe."
```

That single sentence is enough context for the next chat to get started.
This document holds the rest.

---

## 10. Progress log

*(append entries as phases complete)*

- **2026-04-27** — `NEXT_STEPS.md` created. Project is at end of Phase 3
  (held-out OOS test diagnostic complete; val signal does not transfer).
  Next phase: Phase A — SEC EDGAR filings download.
