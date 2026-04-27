"""Parse the raw EDGAR filings on disk into structured per-section parquet.

Reads from <root>/raw/edgar/<CIK>/*.txt + <root>/raw/edgar/_manifest.csv,
writes <root>/processed/edgar/<CIK>/filings.parquet.

Usage:
    python -m data_pipeline.scripts.parse_filings \
        --out-root /data/.../data
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)5s | %(name)s | %(message)s",
    )
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Root data directory (raw + processed live under here).")
    parser.add_argument("--combined-out", type=Path, default=None,
                        help="Optional path to write a single combined parquet "
                             "of all parsed filings (in addition to per-CIK).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N filings (smoke-test).")
    args = parser.parse_args()

    from data_pipeline.edgar.parser import FilingsParser
    from data_pipeline.edgar.storage import FilingsStorage

    storage = FilingsStorage(args.out_root)
    parser_obj = FilingsParser()

    by_cik = defaultdict(list)
    total = 0
    start = time.time()

    for meta, text in storage.iter_raw():
        try:
            parsed = parser_obj.parse(
                text,
                cik=meta.cik,
                accession=meta.accession,
                form=meta.form,
                filing_date=meta.filing_date,
                period=meta.period,
            )
        except Exception as exc:
            logger.error("Parse FAIL %s/%s: %r", meta.cik, meta.accession, exc)
            continue
        by_cik[meta.cik].append(parsed)
        total += 1

        if total % 200 == 0:
            logger.info("Parsed %d filings (%.1f /s)",
                        total, total / max(1.0, time.time() - start))

        if args.limit is not None and total >= args.limit:
            logger.info("Hit --limit %d; stopping early.", args.limit)
            break

    # Write per-CIK parquets.
    for cik, parsed_list in by_cik.items():
        path = storage.save_parsed_filings(cik, parsed_list)
        logger.info("CIK=%s wrote %d parsed → %s", cik, len(parsed_list), path)

    if args.combined_out is not None:
        combined = storage.write_combined_parsed(args.combined_out)
        logger.info("Combined parquet → %s", combined)

    elapsed = time.time() - start
    logger.info(
        "Done — parsed %d filings across %d CIKs in %.1f s",
        total, len(by_cik), elapsed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
