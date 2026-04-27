# Phase I — Broader FRED macro series

## Goal

Extend the existing 4-series macro set (VIX, 10Y yield, DXY, SPY) with more
regime-discriminating series. Marginal lift over what's already there but
cheap to add.

## Source

[FRED API](https://fred.stlouisfed.org/docs/api/fred/) — free, requires a
free API key (one minute to get from the FRED website).

## Useful series (priority-ordered)

| FRED ID | What |
|---|---|
| `T10Y2Y` | 10Y - 2Y Treasury spread (recession indicator) |
| `T10Y3M` | 10Y - 3M Treasury spread |
| `BAA10Y` | Moody's Baa - 10Y (credit spread) |
| `BAMLH0A0HYM2` | BAML US High Yield OAS |
| `STLFSI4` | St Louis Fed Financial Stress Index |
| `UNRATE` | Unemployment rate |
| `NAPMNOI` | ISM Manufacturing PMI |
| `M2SL` | M2 money supply |
| `DFF` | Effective Fed funds rate |
| `DCOILWTICO` | WTI crude oil price |
| `GOLDAMGBD228NLBM` | Gold price (London PM fix) |
| `OFRFSI` | OFR Financial Stress Index |

## Implementation sketch

```python
# data_pipeline/fred/client.py
class FredClient:
    def __init__(self, api_key: str): ...
    def fetch_series(self, series_id: str) -> pd.DataFrame:
        """Returns daily/weekly index aligned to trading days."""
```

Then a `scripts/fetch_fred.py` that pulls all configured series and writes
`data/raw/macro/<series_id>.parquet`. Already-aligned with the macro
loading path in `constellation_quant.data.macro`.

## Why Phase I (low priority)

Macro is already in the model. More macro = more regime signal but
diminishing returns. Add it after the higher-priority filings/news
features are in place.
