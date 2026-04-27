# Phase G — Free financial news / RSS

## Goal

Per-ticker daily news headlines for event extraction. Lift over price-only
baselines: 3–5 pp in REST 2021 (WWW).

## Sources (free)

- **Yahoo Finance RSS**: `https://feeds.finance.yahoo.com/rss/2.0/headline?s=<TICKER>&region=US&lang=en-US`
- **Google News RSS**: `https://news.google.com/rss/search?q=<TICKER>+stock`
- **FinViz news**: scrape `https://finviz.com/quote.ashx?t=<TICKER>` (HTML, no rate limit if polite)
- **SEC filings as news**: 8-K filings are de-facto company press releases (already covered by Phase A's EDGAR client — extend `--forms` to include 8-K).

## Caveats

- Historical archive is patchy — modern (2010+) is rich, earlier sparse.
- Headlines decay fast. Must include timestamp.
- Quality varies — needs deduplication and source-credibility weighting.

## Implementation sketch

```
data_pipeline/news/
├── client.py              # per-source RSS fetchers
├── deduper.py             # cross-source dedup (URL + title hash)
├── event_classifier.py    # FinBERT on headlines for sentiment + event-type classification
└── aggregator.py          # daily per-ticker rollup
```

Output: `data/processed/news/per_ticker_per_date.parquet`
Columns: `ticker`, `date`, `n_headlines`, `mean_sentiment`,
         `frac_negative_events`, `top_event_type`

## Why Phase G (not earlier)

Noisier than filings, not the foundational signal. Add only after
Phases A–D show measurable test-IC lift on filing-based features alone.
