# Phase H — SEC Form 4 insider trading

## Goal

Insider buy/sell signal as a slow feature. Strong in event windows (cluster
buys, especially), narrow in scope (only fires when there's actual insider
activity).

## Source

Same EDGAR endpoint we already use in Phase A. Form 4 entries appear in the
submissions JSON alongside 10-K/10-Q. The Phase A `EdgarClient.list_filings()`
already supports an arbitrary `forms` tuple — pass `("4",)`.

## Implementation

The Form 4 XBRL is highly structured (no NLP needed). Add:

```
data_pipeline/form4/
├── parser.py             # Form 4 XBRL → structured rows
└── aggregator.py         # daily per-ticker rollup
```

Per-row fields: `cik`, `insider_name`, `insider_role`,
`transaction_date`, `transaction_code` (P=buy, S=sell), `shares`, `price`,
`total_value_usd`.

Daily rollup features per ticker:
- `n_insider_buys_30d`
- `n_insider_sells_30d`
- `usd_net_insider_30d`
- `cluster_buy_flag` (≥3 distinct insiders buying within 30d)

## Why Phase H (later)

Adds at most a few percentage points on stocks with active insider trading,
zero on the rest. Real signal but narrow.
