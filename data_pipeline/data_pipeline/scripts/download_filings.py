"""Download SEC 10-K and 10-Q filings for a list of tickers.

Usage:
    python -m data_pipeline.scripts.download_filings \
        --tickers AAPL MSFT NVDA \
        --out-root /data/.../data \
        --user-agent "constellation-quant research nikraftarz@gmail.com" \
        --forms 10-K 10-Q \
        --date-from 1995-01-01

For full-universe downloads, pass --tickers-file pointing at a one-ticker-
per-line text file. The --date-from / --date-to flags constrain the time
window. Resume-safe: skips filings already in the manifest.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _parse_tickers(args) -> list[str]:
    if args.tickers_file:
        text = Path(args.tickers_file).read_text()
        return [t.strip().upper() for t in text.splitlines() if t.strip() and not t.startswith("#")]
    return [t.upper() for t in (args.tickers or [])]


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tickers", nargs="*", help="Tickers to download (space-separated).")
    parser.add_argument("--tickers-file", type=Path,
                        help="Path to a text file with one ticker per line.")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Root data directory (raw/edgar lives under here).")
    parser.add_argument("--user-agent", required=True,
                        help="EDGAR User-Agent string. Must contain a contact email.")
    parser.add_argument("--forms", nargs="+", default=["10-K", "10-Q"],
                        help="Form types to fetch.")
    parser.add_argument("--date-from", default=None,
                        help="Earliest filing date (YYYY-MM-DD).")
    parser.add_argument("--date-to", default=None,
                        help="Latest filing date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap total filings downloaded (smoke-test).")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Where to cache the company-tickers manifest.")
    args = parser.parse_args()

    tickers = _parse_tickers(args)
    if not tickers:
        logger.error("No tickers given (use --tickers or --tickers-file).")
        return 2

    # Late imports
    from data_pipeline.edgar.client import EdgarClient
    from data_pipeline.edgar.storage import FilingsStorage

    client = EdgarClient(
        user_agent=args.user_agent,
        cache_dir=args.cache_dir or (args.out_root / "cache" / "edgar"),
    )
    storage = FilingsStorage(args.out_root)
    seen = storage.already_downloaded()
    logger.info("Found %d filings already on disk", len(seen))

    n_downloaded = 0
    start = time.time()

    for ticker in tickers:
        cik = client.lookup_cik(ticker)
        if cik is None:
            logger.warning("Skipping %s — not found in SEC tickers manifest", ticker)
            continue

        filings = list(client.list_filings(
            cik,
            forms=tuple(args.forms),
            date_from=args.date_from,
            date_to=args.date_to,
        ))
        new_filings = [f for f in filings if (f.cik, f.accession) not in seen]
        logger.info(
            "%-6s CIK=%s — %d total, %d new",
            ticker, cik, len(filings), len(new_filings),
        )

        for f in new_filings:
            try:
                text = client.fetch_filing_text(f)
            except Exception as exc:
                logger.error("FAIL %s/%s: %r", f.cik, f.accession, exc)
                continue
            storage.save_raw_filing(f, text)
            n_downloaded += 1

            if args.limit is not None and n_downloaded >= args.limit:
                logger.info("Hit --limit %d; stopping early.", args.limit)
                elapsed = time.time() - start
                logger.info(
                    "Done — %d downloaded in %.1f s (%.2f filings/s)",
                    n_downloaded, elapsed, n_downloaded / max(1.0, elapsed),
                )
                return 0

    elapsed = time.time() - start
    logger.info(
        "Done — %d downloaded in %.1f s (%.2f filings/s)",
        n_downloaded, elapsed, n_downloaded / max(1.0, elapsed),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
