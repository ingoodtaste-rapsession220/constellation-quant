"""Data cleaning: split/dividend adjustment, spike removal, calendar alignment.

All functions are pure — they take DataFrames and return DataFrames, with no
I/O side effects. The `DataCleaner` class wires them together for batch
processing and logs every correction to the `CleaningReport`.

Assumptions about the input frames (from `PriceDownloader`):
    columns = [date, open, high, low, close, adj_close, volume,
               dividends, stock_splits]
    - `date` is a timezone-naive pandas Timestamp at midnight.
    - `adj_close` is yfinance's split+dividend-adjusted close.
    - Rows are sorted by date ascending, no duplicates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from constellation_quant.utils import get_logger

log = get_logger(__name__)


# ── Report ─────────────────────────────────────────────────────────────────


@dataclass
class CleaningReport:
    """Per-ticker tally of corrections applied during cleaning."""

    rows_in:            int = 0
    rows_out:           int = 0
    duplicates_dropped: int = 0
    spike_rows_removed: int = 0
    gaps_filled:        int = 0
    renamed_tickers:    Dict[str, str] = field(default_factory=dict)
    nan_price_rows:     int = 0


# ── Pure helpers ───────────────────────────────────────────────────────────


def drop_duplicates(df):
    """Drop rows with duplicate `date`, keeping the first occurrence."""
    before = len(df)
    cleaned = df.drop_duplicates(subset=["date"], keep="first")
    return cleaned.reset_index(drop=True), before - len(cleaned)


def drop_nan_prices(df):
    """Drop rows where the close or adj_close is NaN."""
    before = len(df)
    cleaned = df.dropna(subset=["close", "adj_close"])
    return cleaned.reset_index(drop=True), before - len(cleaned)


def detect_revert_spikes(df, threshold: float = 0.50):
    """Return a boolean mask over rows that are single-day revert spikes.

    A "revert spike" is defined as:
        |r_t| > threshold  AND  sign(r_t) != sign(r_{t+1})
        AND  |r_{t+1}| > threshold / 2
    where r is log return on `adj_close`. Real events (earnings gaps,
    acquisitions) usually persist, so this pattern almost always flags bad
    prints rather than genuine moves.

    Returns a pandas Series[bool] aligned with `df`.
    """
    import numpy as np
    import pandas as pd

    if len(df) < 3:
        return pd.Series([False] * len(df), index=df.index)

    returns = np.log(df["adj_close"].astype(float)).diff()
    big_move = returns.abs() > threshold
    next_return = returns.shift(-1)
    next_big = next_return.abs() > threshold / 2
    opposite_sign = np.sign(returns) * np.sign(next_return) < 0
    spike = big_move & next_big & opposite_sign
    return spike.fillna(False)


def remove_revert_spikes(df, threshold: float = 0.50) -> Tuple["object", int]:
    """Drop rows flagged by `detect_revert_spikes`. Returns (cleaned, count)."""
    mask = detect_revert_spikes(df, threshold=threshold)
    count = int(mask.sum())
    cleaned = df.loc[~mask].reset_index(drop=True)
    return cleaned, count


def forward_fill_small_gaps(df, max_gap_days: int = 2, calendar=None) -> Tuple["object", int]:
    """Forward-fill missing trading days up to `max_gap_days` in length.

    If a `calendar` (DatetimeIndex of trading days) is provided, align to it
    first, then ffill. Otherwise, align to the union of the ticker's own
    existing business days.

    Returns (filled_df, num_rows_added).
    """
    import pandas as pd

    if df.empty:
        return df, 0

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()

    if calendar is None:
        calendar = pd.date_range(df.index.min(), df.index.max(), freq="B")

    original_size = len(df)
    df = df.reindex(calendar)

    missing = df["close"].isna()
    gap_sizes = _contiguous_gap_sizes(missing)
    fillable = gap_sizes <= max_gap_days

    for col in ("open", "high", "low", "close", "adj_close"):
        if col in df.columns:
            filled = df[col].ffill()
            df[col] = df[col].where(~fillable, filled)
    for col in ("volume", "dividends", "stock_splits"):
        if col in df.columns:
            df[col] = df[col].where(~fillable, 0.0)

    df = df.dropna(subset=["close", "adj_close"])
    df = df.reset_index().rename(columns={"index": "date"})
    added = len(df) - original_size
    return df, max(added, 0)


def _contiguous_gap_sizes(mask):
    """For each True in `mask`, return the length of its contiguous True run."""
    import numpy as np
    import pandas as pd

    arr = mask.to_numpy(dtype=bool)
    n = len(arr)
    sizes = np.zeros(n, dtype=int)
    i = 0
    while i < n:
        if not arr[i]:
            i += 1
            continue
        j = i
        while j < n and arr[j]:
            j += 1
        sizes[i:j] = j - i
        i = j
    return pd.Series(sizes, index=mask.index)


def align_to_calendar(df, calendar):
    """Reindex to `calendar`, dropping rows outside it. Does not forward-fill."""
    import pandas as pd

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df[df["date"].isin(calendar)].reset_index(drop=True)


def apply_ticker_aliases(
    ticker: str,
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve ticker renames (e.g. FB -> META, GOOG -> GOOGL)."""
    if not aliases:
        return ticker
    return aliases.get(ticker.upper(), ticker)


# ── Orchestrator ───────────────────────────────────────────────────────────


class DataCleaner:
    """Run the full cleaning pipeline on a batch of price frames.

    Typical usage:
        cleaner = DataCleaner(config["cleaning"])
        cleaned, reports = cleaner.clean_batch({t: df for t, df in price_iter})
    """

    def __init__(self, config: Optional[Mapping] = None):
        self.cfg = dict(config or {})
        self.spike_threshold = float(self.cfg.get("max_single_day_move", 0.50))
        self.max_gap = int(self.cfg.get("forward_fill_max_gap_days", 2))
        self.aliases = dict(self.cfg.get("ticker_aliases", {}) or {})

    def clean_one(
        self,
        ticker: str,
        df,
        calendar=None,
    ) -> Tuple[str, "object", CleaningReport]:
        """Clean a single ticker frame. Returns (canonical_ticker, cleaned_df, report)."""
        report = CleaningReport(rows_in=len(df))

        canonical = apply_ticker_aliases(ticker, self.aliases)
        if canonical != ticker:
            report.renamed_tickers[ticker] = canonical

        if df is None or df.empty:
            report.rows_out = 0
            return canonical, df, report

        df, dupes = drop_duplicates(df)
        report.duplicates_dropped = dupes

        df, nan_rows = drop_nan_prices(df)
        report.nan_price_rows = nan_rows

        df, removed = remove_revert_spikes(df, threshold=self.spike_threshold)
        report.spike_rows_removed = removed

        df, added = forward_fill_small_gaps(df, max_gap_days=self.max_gap, calendar=calendar)
        report.gaps_filled = added

        report.rows_out = len(df)
        return canonical, df, report

    def clean_batch(
        self,
        frames: Mapping[str, "object"],
        calendar=None,
    ) -> Tuple[Dict[str, "object"], Dict[str, CleaningReport]]:
        """Clean multiple tickers; merge duplicates created by aliasing."""
        cleaned: Dict[str, "object"] = {}
        reports: Dict[str, CleaningReport] = {}

        for ticker, df in frames.items():
            canonical, cleaned_df, report = self.clean_one(ticker, df, calendar=calendar)
            reports[ticker] = report

            if canonical in cleaned:
                cleaned[canonical] = self._merge_alias_frames(cleaned[canonical], cleaned_df)
            else:
                cleaned[canonical] = cleaned_df

        return cleaned, reports

    @staticmethod
    def _merge_alias_frames(existing, incoming):
        """Concat and dedupe on date — used when two tickers alias to the same name."""
        import pandas as pd
        merged = pd.concat([existing, incoming], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="first")
        return merged.sort_values("date").reset_index(drop=True)


def build_trading_calendar(frames: Iterable["object"]):
    """Intersect (or union) the date columns of several frames to form a calendar.

    Here we use the UNION and trust downstream masking — if a stock wasn't
    trading on a given date (pre-IPO, post-delisting), the Dataset will mask
    it out.
    """
    import pandas as pd

    all_dates: List = []
    for df in frames:
        if df is None or df.empty:
            continue
        all_dates.extend(pd.to_datetime(df["date"]).tolist())
    if not all_dates:
        return pd.DatetimeIndex([])
    idx = pd.DatetimeIndex(sorted(set(all_dates))).normalize()
    return idx
