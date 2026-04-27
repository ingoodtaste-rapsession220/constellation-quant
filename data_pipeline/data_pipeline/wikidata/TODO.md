# Phase F — Wikidata corporate relations

## Goal

Populate the multi-relational graph for variants E–I with real corporate
relationships (board interlocks, supply chain, parent-subsidiary, owner-of)
extracted from Wikidata. Currently those R-GAT variants only see three
thin edge types (correlation, fundamentals similarity, learnable
attention) — adding richer edges should give them something real to
message-pass over.

## Source

[Wikidata.org](https://www.wikidata.org) — free SPARQL endpoint at
`https://query.wikidata.org/sparql`.

## Useful properties

| Property | Meaning |
|---|---|
| `P749` | parent organization |
| `P127` | owner of |
| `P488` | chairperson |
| `P3320` | board member |
| `P112` | founder |
| `P1830` | owner of (legal) |
| `P159` | headquartered in |
| `P452` | industry |

## Implementation

Build `client.py` exposing:

- `WikidataClient.find_company_qid(ticker, name)` — resolve ticker/name → QID
- `WikidataClient.fetch_relations(qid)` — return JSON of relations
- `extract_edges(relations) -> List[Edge]` — flatten into edge list

And `scripts/fetch_wikidata.py` to run the lookup over the S&P 500 universe.

## Output

Parquet: `data/processed/wikidata/edges.parquet` with columns:
- `src_ticker`, `dst_ticker`, `relation`, `confidence`, `as_of_date`

Then in `data_pipeline.integration` add a function that reads this parquet
and emits the edge tensors `constellation_quant.graph.builder` can consume.

## Why this is Phase F (not earlier)

The lift from richer edges is real but moderate (the comparison papers see
2–4 pp lift). Filings + sentiment (Phases A–D) should come first because
that's where the bulk of the test-IC improvement is expected.
