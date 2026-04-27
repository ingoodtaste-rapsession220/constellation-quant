"""Quarterly fundamentals downloader via yfinance.

Pulls income statement, balance sheet, and cashflow data per ticker, plus
current ratios / shares-outstanding. Results are cast to long format and
written one parquet per ticker under `paths.raw_fundamentals`:

    date, metric, value

Downstream the feature engine pivots this into wide format and forward-fills
quarterly values to daily frequency.

yfinance's `quarterly_financials`, `quarterly_balance_sheet`, and
`quarterly_cashflow` attributes return wide DataFrames indexed by statement
line item, columns = quarter-end dates. We normalise those to long.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from constellation_quant.data._paths import DataPaths
from constellation_quant.utils import get_logger

log = get_logger(__name__)


class FundamentalsError(RuntimeError):
    """Raised when fundamentals are unavailable for a ticker."""


# Canonical line items we try to extract. yfinance labels vary across
# versions — we normalise using a case-insensitive contains-match. Missing
# items are simply absent from the output (handled by the feature engine).
METRIC_ALIASES: Dict[str, List[str]] = {
    "total_revenue":    ["total revenue"],
    "net_income":       ["net income"],
    "ebitda":           ["ebitda", "normalized ebitda"],
    "total_assets":     ["total assets"],
    "total_debt":       ["total debt"],
    "cash":             ["cash and cash equivalents", "cash and short term investments"],
    "stockholders_equity": ["stockholders equity", "total equity gross minority interest"],
    "shares_outstanding":  ["share issued", "ordinary shares number"],
    "operating_cashflow":  ["operating cash flow"],
    "capex":               ["capital expenditure"],
    "dividends_paid":      ["cash dividends paid"],
}


@dataclass
class FundamentalsReport:
    succeeded: List[str] = field(default_factory=list)
    skipped:   List[str] = field(default_factory=list)
    failed:    Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        total = len(self.succeeded) + len(self.skipped) + len(self.failed)
        return (
            f"fundamentals: downloaded={len(self.succeeded)} "
            f"skipped={len(self.skipped)} failed={len(self.failed)} total={total}"
        )


class FundamentalsDownloader:
    """Pull quarterly fundamentals for each ticker and save as long-format parquet."""

    def __init__(
        self,
        paths: DataPaths,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        sleep_between: float = 0.0,
    ):
        self.paths = paths
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.sleep_between = sleep_between

    # ── Public API ─────────────────────────────────────────────────────

    def download_all(
        self,
        tickers: Iterable[str],
        resume: bool = True,
    ) -> FundamentalsReport:
        ticker_list = sorted({t.upper().strip() for t in tickers if t})
        report = FundamentalsReport()
        self.paths.raw_fundamentals.mkdir(parents=True, exist_ok=True)

        for ticker in ticker_list:
            out_path = self.paths.fundamentals_file(ticker)
            if resume and out_path.exists():
                report.skipped.append(ticker)
                continue
            try:
                self.download_one(ticker)
                report.succeeded.append(ticker)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"
                log.warning("  [{}] fundamentals failed: {}", ticker, msg)
                report.failed[ticker] = msg
            if self.sleep_between > 0:
                time.sleep(self.sleep_between)

        log.info(report.summary())
        return report

    def download_one(self, ticker: str) -> Path:
        ticker = ticker.upper().strip()
        frames = self._fetch_with_retries(ticker)
        long_df = self._to_long(frames)
        if long_df.empty:
            raise FundamentalsError(f"{ticker}: no usable fundamentals extracted")
        out_path = self.paths.fundamentals_file(ticker)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_parquet(out_path, index=False, compression="snappy")
        return out_path

    # ── Internals ──────────────────────────────────────────────────────

    def _fetch_with_retries(self, ticker: str) -> Dict[str, "object"]:
        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self._fetch(ticker)
            except Exception as exc:
                last_exc = exc
                time.sleep(self.backoff_base ** attempt)
        if last_exc is not None:
            raise last_exc
        return {}

    def _fetch(self, ticker: str) -> Dict[str, "object"]:
        """Fetch raw quarterly frames from yfinance. Isolated for mocking."""
        import yfinance as yf

        t = yf.Ticker(ticker)
        return {
            "income":      t.quarterly_financials,
            "balance":     t.quarterly_balance_sheet,
            "cashflow":    t.quarterly_cashflow,
        }

    def _to_long(self, frames: Dict[str, "object"]) -> "object":
        """Melt wide quarterly frames into `date, metric, value` triples."""
        import pandas as pd

        rows: List[Dict] = []
        for section_name, df in frames.items():
            if df is None or df.empty:
                continue
            # yfinance: rows = line items, columns = quarter-end dates.
            for metric_key, aliases in METRIC_ALIASES.items():
                found_label = self._match_label(df.index, aliases)
                if found_label is None:
                    continue
                series = df.loc[found_label]
                for quarter_end, value in series.items():
                    if pd.isna(value):
                        continue
                    try:
                        ts = pd.to_datetime(quarter_end).normalize()
                    except (ValueError, TypeError):
                        continue
                    rows.append({"date": ts, "metric": metric_key, "value": float(value)})
            _ = section_name  # kept for potential future breakdown

        if not rows:
            return pd.DataFrame(columns=["date", "metric", "value"])
        out = pd.DataFrame(rows)
        out = out.drop_duplicates(subset=["date", "metric"], keep="first")
        return out.sort_values(["date", "metric"]).reset_index(drop=True)

    @staticmethod
    def _match_label(index, aliases: List[str]) -> Optional[str]:
        """Return the first index label whose lowercase string contains any alias."""
        index_lower = {str(lbl): str(lbl).lower() for lbl in index}
        for alias in aliases:
            for orig, lower in index_lower.items():
                if alias in lower:
                    return orig
        return None
