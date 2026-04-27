# Phase K — Wikipedia page views / edits

Lowest priority. Free attention-proxy signal that has shown weak but real
predictive power in some published studies; most papers find it marginal
once price-volume features are already in the model.

## Source

[Wikimedia REST API](https://wikimedia.org/api/rest_v1/) — free, generous
rate limits, no API key needed.

## Endpoints

- Page views: `/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{article}/daily/{start}/{end}`
- Edits: `/metrics/edits/per-page/en.wikipedia/{article}/all-editor-types/daily/{start}/{end}`

## Implementation

A short script that maps ticker → Wikipedia article (S&P 500 companies have
predictable article titles, e.g. "Apple Inc." for AAPL), pulls daily
page views and edit counts, writes to parquet.

Output: `data/processed/wikipedia/per_ticker_per_date.parquet`
Columns: `ticker`, `date`, `wp_views`, `wp_edits`, `wp_views_zscore_30d`

## Why Phase K (last)

Marginal lift. Build only after Phases A–J are in place and there's
specifically a need for an attention proxy.
