"""Entry point: download all S&P 500 data (OHLCV, fundamentals, sentiment, membership).

    # Full pipeline (resume-safe; pick up from wherever you stopped)
    python scripts/download_data.py --resume

    # Skip one of the stages (useful on re-runs)
    python scripts/download_data.py --skip-sentiment

    # Single ticker (debug)
    python scripts/download_data.py --ticker AAPL --skip-membership --skip-sentiment

    # Rebuild the membership roster only
    python scripts/download_data.py --only-membership

Network calls are made only when this script is invoked with an active
environment — building / testing the package does not fetch any data.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path
from typing import List

from constellation_quant.data import (
    DataPaths,
    FundamentalsDownloader,
    MembershipRoster,
    PriceDownloader,
    SentimentDownloader,
    build_roster_from_sources,
    download_macro,
    validate_roster,
)
from constellation_quant.utils import get_logger, load_config, log_environment

log = get_logger(__name__)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config",   type=Path, default=Path("configs/data_config.yaml"))
    p.add_argument("--paths-config",  type=Path, default=Path("configs/paths.yaml"))
    p.add_argument("--resume",        action="store_true",
                   help="Skip tickers / stages already on disk.")
    p.add_argument("--ticker",        type=str, default=None,
                   help="Download a single ticker (price + fundamentals only).")

    # Stage toggles
    p.add_argument("--skip-membership",  action="store_true")
    p.add_argument("--skip-prices",      action="store_true")
    p.add_argument("--skip-fundamentals", action="store_true")
    p.add_argument("--skip-sentiment",   action="store_true")
    p.add_argument("--skip-macro",       action="store_true")
    p.add_argument("--only-macro",       action="store_true",
                   help="Pull just the four macro indicators and exit.")
    p.add_argument("--only-membership",  action="store_true",
                   help="Build the roster, then exit.")

    # Date range override (data_config is authoritative by default)
    p.add_argument("--start-date", type=str, default=None,
                   help="Override start date (YYYY-MM-DD). Default: earliest in data_config.")
    p.add_argument("--end-date",   type=str, default=None)

    return p.parse_args()


# ── Main pipeline ──────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    _ensure_path_env_vars()
    log.info("Environment: {}", log_environment())

    data_cfg  = load_config(args.data_config)
    paths_cfg = load_config(args.paths_config)

    paths = DataPaths.from_config(paths_cfg)
    paths.ensure_dirs()
    log.info("Data dir: {}", paths.data_dir)

    start_date, end_date = _resolve_date_range(args, data_cfg)
    log.info("Date range: {} → {}", start_date, end_date)

    # ── 0. Macro indicators (small, fast) ──────────────────────────────
    # Pulled before membership so a macro-only run can short-circuit.
    if not args.skip_macro:
        log.info("── Macro stage: VIX / TNX / DXY / SPY ──")
        download_macro(paths=paths, start=start_date, end=end_date)
    if args.only_macro:
        log.info("--only-macro set; exiting after macro stage.")
        return 0

    # ── 1. Membership roster ───────────────────────────────────────────
    if args.skip_membership:
        if not paths.membership_file.exists():
            raise SystemExit(
                f"--skip-membership passed but {paths.membership_file} doesn't exist. "
                "Run without the flag first to build the roster."
            )
        roster = MembershipRoster.load_json(paths.membership_file)
        log.info("Skipping membership stage; loaded roster from {}",
                 paths.membership_file)
    else:
        roster = _stage_membership(args, data_cfg, paths)
        if args.only_membership:
            return 0

    tickers = _select_tickers(args, roster)
    log.info("Ticker universe: {} symbols", len(tickers))

    # ── 2. Prices ──────────────────────────────────────────────────────
    if not args.skip_prices:
        log.info("── Stage 2 / 4: OHLCV prices ──")
        PriceDownloader(
            paths=paths,
            start=start_date,
            end=end_date,
            batch_size=int(data_cfg["download"].get("batch_size", 50)),
            max_retries=int(data_cfg["download"].get("max_retries", 5)),
            backoff_base=float(data_cfg["download"].get("retry_backoff_base", 2.0)),
        ).download_all(tickers, resume=args.resume)

    # ── 3. Fundamentals ────────────────────────────────────────────────
    if not args.skip_fundamentals:
        log.info("── Stage 3 / 4: Quarterly fundamentals ──")
        FundamentalsDownloader(
            paths=paths,
            max_retries=int(data_cfg["download"].get("max_retries", 5)),
            backoff_base=float(data_cfg["download"].get("retry_backoff_base", 2.0)),
        ).download_all(tickers, resume=args.resume)

    # ── 4. Sentiment ───────────────────────────────────────────────────
    if not args.skip_sentiment:
        log.info("── Stage 4 / 4: Sentiment ──")
        SentimentDownloader(paths=paths).download_all(tickers, resume=args.resume)

    log.info("Download pipeline complete.")
    return 0


# ── Stages ─────────────────────────────────────────────────────────────────


def _stage_membership(args, data_cfg, paths: DataPaths) -> MembershipRoster:
    log.info("── Stage 1 / 4: S&P 500 membership roster ──")
    if args.resume and paths.membership_file.exists() and not args.only_membership:
        roster = MembershipRoster.load_json(paths.membership_file)
        log.info("Loaded existing roster: {} snapshots", len(roster.snapshot_dates()))
    else:
        csv_url = data_cfg.get("universe", {}).get("membership_csv_url")
        roster = build_roster_from_sources(csv_url=csv_url, fallback_to_wikipedia=True)
        roster.save_json(paths.membership_file)

    errors = validate_roster(roster)
    if errors:
        log.warning("Roster validation produced {} issue(s):", len(errors))
        for e in errors[:10]:
            log.warning("  - {}", e)
        if len(errors) > 10:
            log.warning("  ... (+{} more)", len(errors) - 10)
    else:
        log.info("Roster validation: OK")
    return roster


def _select_tickers(args, roster: MembershipRoster) -> List[str]:
    if args.ticker:
        return [args.ticker.upper().strip()]
    return sorted(roster.all_tickers_ever())


# ── Helpers ────────────────────────────────────────────────────────────────


def _resolve_date_range(args, data_cfg: dict) -> tuple[str, str | None]:
    if args.start_date:
        start = args.start_date
    else:
        start = data_cfg["splits"]["train"]["start"]
    end = args.end_date or data_cfg["splits"]["test"]["end"] or date.today().isoformat()
    return start, end


def _ensure_path_env_vars() -> None:
    """Default PROJECT_ROOT / SCRATCH / DATA_DIR before configs are loaded.

    Lets developers run the script locally without explicitly exporting every
    path var. SLURM templates set these first, so this block is a no-op on
    HPC.
    """
    os.environ.setdefault("PROJECT_ROOT", str(Path.cwd().resolve()))
    os.environ.setdefault("SCRATCH",      os.environ["PROJECT_ROOT"] + "/.scratch")
    os.environ.setdefault("DATA_DIR",     os.environ["SCRATCH"] + "/constellation_quant")


if __name__ == "__main__":
    sys.exit(main())
