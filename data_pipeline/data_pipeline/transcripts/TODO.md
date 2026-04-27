# Phase J — Earnings call transcripts

## Goal

Apply the same language-drift / sentiment pipeline (built in Phase B+C) to
spoken management commentary. Adds tone / hedging / uncertainty signals
that filings smooth out.

**Important**: transcripts are *complementary* to filings, not a substitute.
The Lazy Prices effect is anchored on filings. Add transcripts AFTER the
filings pipeline is working.

## Sources (free, in order of feasibility)

1. **Seeking Alpha transcripts** (text). Hostile to scraping — needs
   careful rate limiting and robust parsing. Coverage is the best of any
   free source.
2. **Yahoo Finance earnings** (summaries only, not full transcripts).
3. **Company IR audio recordings** + Whisper-class transcription. Fully
   reliable for coverage but storage-heavy (audio is ~50–100 MB per call;
   503 companies × 4/yr × 25 yr × 75 MB ≈ 3.7 TB raw audio — exceeds
   project quota). Skip the audio approach unless we get a separate
   scratch filesystem.

## Implementation sketch

```
data_pipeline/transcripts/
├── seeking_alpha_client.py   # rate-limited HTML scraper, with retry / backoff
├── parser.py                 # transcript HTML → structured (speaker, role, text)
└── pipeline.py               # orchestrate fetch + parse + reuse Phase B/C NLP
```

Per-call structured rows: `ticker`, `call_date`, `speaker`, `role`
(executive | analyst | operator), `text_chunk`.

Then run the existing `data_pipeline.nlp` pipeline over executive-role
chunks only (analyst questions don't carry management signal) → produces
the same per-(ticker, date) features as filings, with a `source=transcripts`
column to distinguish.

## Why Phase J (after A–H)

Transcripts add ~0.5–1 pp test IC on top of filings in the published
literature. They're a good *complement* but not a replacement for the
foundational filings signal. Build them after Phase A–D demonstrably moves
the needle.
