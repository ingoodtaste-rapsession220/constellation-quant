"""OHLCV price downloader via yfinance.

Downloads every ticker that ever appeared in the S&P 500 roster. Batched,
retry-with-backoff, resume-aware. Output: one parquet file per ticker under
`paths.raw_prices`, with columns:

    date, open, high, low, close, adj_close, volume, dividends, stock_splits

Delisted tickers are logged and skipped (yfinance returns an empty frame).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from constellation_quant.data._paths import DataPaths
from constellation_quant.utils import get_logger

log = get_logger(__name__)


class DownloadError(RuntimeError):
    """Raised when a ticker fetch returns no usable data after all retries."""


@dataclass
class DownloadReport:
    """Outcome of a multi-ticker download run."""

    succeeded: List[str] = field(default_factory=list)
    skipped:   List[str] = field(default_factory=list)  # already on disk
    failed:    Dict[str, str] = field(default_factory=dict)  # ticker -> err message

    @property
    def total(self) -> int:
        return len(self.succeeded) + len(self.skipped) + len(self.failed)

    def summary(self) -> str:
        return (
            f"downloaded={len(self.succeeded)} "
            f"skipped={len(self.skipped)} "
            f"failed={len(self.failed)} "
            f"total={self.total}"
        )


# Column mapping from yfinance (human-readable) -> our snake_case schema.
_YF_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}


class PriceDownloader:
    """Download and cache per-ticker OHLCV series as parquet files.

    Args:
        paths: Resolved DataPaths — `paths.raw_prices` must be writable.
        start: First date to request (inclusive). ISO string.
        end: Last date (exclusive), or None for "today".
        batch_size: Tickers per progress batch (for logging only).
        max_retries: Per-ticker retry cap on transient network errors.
        backoff_base: Exponential-backoff base (seconds): wait = base ** attempt.
        sleep_between: Small inter-ticker sleep to avoid YF rate limits.
    """

    def __init__(
        self,
        paths: DataPaths,
        start: str = "2000-01-01",
        end: Optional[str] = None,
        batch_size: int = 50,
        max_retries: int = 5,
        backoff_base: float = 2.0,
        sleep_between: float = 0.0,
    ):
        self.paths = paths
        self.start = start
        self.end = end
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.sleep_between = sleep_between

    # ── Public API ─────────────────────────────────────────────────────

    def download_all(
        self,
        tickers: Iterable[str],
        resume: bool = True,
    ) -> DownloadReport:
        """Download every ticker, respecting the resume flag."""
        ticker_list = sorted({t.upper().strip() for t in tickers if t})
        report = DownloadReport()
        if not ticker_list:
            log.warning("download_all called with empty ticker list.")
            return report

        self.paths.raw_prices.mkdir(parents=True, exist_ok=True)
        num_batches = (len(ticker_list) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_i = batch_idx * self.batch_size
            batch = ticker_list[start_i : start_i + self.batch_size]
            log.info(
                "Batch {}/{} ({} tickers)",
                batch_idx + 1, num_batches, len(batch),
            )
            for ticker in batch:
                self._process_one(ticker, resume, report)

        log.info("Price download complete: {}", report.summary())
        return report

    def download_one(self, ticker: str) -> Path:
        """Fetch one ticker and write to parquet. Returns the file path."""
        ticker = ticker.upper().strip()
        df = self._fetch_with_retries(ticker)
        if df is None or df.empty:
            raise DownloadError(f"{ticker}: yfinance returned no rows")
        out_path = self.paths.price_file(ticker)
        self._write_parquet(df, out_path)
        return out_path

    # ── Internals ──────────────────────────────────────────────────────

    def _process_one(self, ticker: str, resume: bool, report: DownloadReport) -> None:
        out_path = self.paths.price_file(ticker)
        if resume and out_path.exists():
            report.skipped.append(ticker)
            return
        try:
            self.download_one(ticker)
            report.succeeded.append(ticker)
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            log.warning("  [{}] failed: {}", ticker, msg)
            report.failed[ticker] = msg

        if self.sleep_between > 0:
            time.sleep(self.sleep_between)

    def _fetch_with_retries(self, ticker: str):
        """Retry a transient failure up to `max_retries` times.

        Empty-frame results (delisted tickers) are NOT retried — they're a
        fast permanent failure.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                df = self._fetch_one(ticker)
                if df is None or df.empty:
                    return df  # permanent — no retry
                return self._normalise(df)
            except Exception as exc:
                last_exc = exc
                wait = self.backoff_base ** attempt
                log.debug(
                    "  [{}] attempt {}/{} failed ({}); sleep {}s",
                    ticker, attempt + 1, self.max_retries, exc, wait,
                )
                time.sleep(wait)
        if last_exc is not None:
            raise last_exc
        return None

    def _fetch_one(self, ticker: str):
        """Single yfinance call. Isolated to make mocking trivial in tests."""
        import yfinance as yf

        t = yf.Ticker(ticker)
        df = t.history(
            start=self.start,
            end=self.end,
            auto_adjust=False,
            actions=True,
            raise_errors=False,
        )
        return df

    @staticmethod
    def _normalise(df):
        """Rename columns to snake_case and flatten the index to a `date` column."""
        import pandas as pd

        df = df.rename(columns=_YF_COLUMN_MAP)
        # yfinance returns a tz-aware DatetimeIndex; strip tz for parquet portability.
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.reset_index().rename(columns={"Date": "date", "index": "date"})
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

        required = ["open", "high", "low", "close", "adj_close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise DownloadError(f"yfinance payload missing columns: {missing}")
        for c in ("dividends", "stock_splits"):
            if c not in df.columns:
                df[c] = 0.0
        return df[["date"] + required + ["dividends", "stock_splits"]]

    @staticmethod
    def _write_parquet(df, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False, compression="snappy")
