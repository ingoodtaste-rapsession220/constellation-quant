"""Build the per-(ticker, date) wide features parquet ready for the
constellation_quant dataset to ingest.

Inputs:
  - drift.parquet, sentiment.parquet from compute_nlp_features
  - the manifest of CIK -> ticker (built from the SEC tickers manifest, or
    explicitly provided as a CSV)
  - the trading-days index from the existing constellation_quant data

Output:
  - <out-root>/processed/nlp_features/per_ticker_per_date.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _load_cik_to_ticker(cache_dir: Path) -> dict[str, str]:
    """Return zero-padded CIK -> ticker, from the SEC tickers manifest."""
    cache_file = cache_dir / "company_tickers.json"
    if not cache_file.exists():
        raise FileNotFoundError(
            f"SEC tickers manifest not found: {cache_file}. "
            "Run download_filings first (it caches this file)."
        )
    data = json.loads(cache_file.read_text())
    out: dict[str, str] = {}
    for entry in data.values():
        cik = str(entry["cik_str"]).zfill(10)
        ticker = entry["ticker"].upper()
        out[cik] = ticker
    return out


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--nlp-features-dir", required=True, type=Path,
                        help="Directory containing drift.parquet + sentiment.parquet.")
    parser.add_argument("--cache-dir", required=True, type=Path,
                        help="EDGAR cache dir (where company_tickers.json lives).")
    parser.add_argument("--trading-days-from", default="1990-01-01",
                        help="Start of the trading-day index (YYYY-MM-DD).")
    parser.add_argument("--trading-days-to", default="2024-12-31",
                        help="End of the trading-day index (YYYY-MM-DD).")
    parser.add_argument("--out-path", required=True, type=Path,
                        help="Output parquet path.")
    args = parser.parse_args()

    drift_df = pd.read_parquet(args.nlp_features_dir / "drift.parquet")
    sent_df = pd.read_parquet(args.nlp_features_dir / "sentiment.parquet")
    cik_to_ticker = _load_cik_to_ticker(args.cache_dir)

    # Use a calendar trading-day index. Real US trading-day calendars are
    # ~252 days/year; a Mon-Fri (B-day) approximation is close enough for
    # forward-fill purposes — when this is merged with the existing dataset's
    # trading-day index, only the matching days survive.
    trading_days = pd.bdate_range(
        args.trading_days_from, args.trading_days_to, freq="B"
    )
    logger.info("Trading days: %s → %s (%d business days)",
                trading_days[0].date(), trading_days[-1].date(), len(trading_days))

    from data_pipeline.integration.prepare_features import (
        build_filings_features,
        write_features_parquet,
    )

    features = build_filings_features(
        drift_df=drift_df,
        sentiment_df=sent_df,
        cik_to_ticker=cik_to_ticker,
        trading_days=trading_days,
    )
    out_path = write_features_parquet(features, args.out_path)
    logger.info("Done. Output: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
