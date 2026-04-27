"""Macro / market-wide features.

Pulls four cross-sectional regime indicators from yfinance and serves them
as additional slow features that broadcast to every stock per prediction
date:

  ^VIX           — equity-market implied vol (fear gauge)
  ^TNX           — 10-year US Treasury yield (rate regime)
  DX-Y.NYB       — dollar index (DXY)
  SPY            — S&P 500 ETF (market state)

For each, we expose a 5-day log change at the prediction date. These four
numbers attach to every stock's slow-feature vector — same values across
the cross-section, but the model can learn that they modulate the
relevance of stock-specific features (e.g. high VIX → momentum patterns
weaken).

If `data/raw/macro/*.parquet` is missing, MacroFeatures.empty() is used
and the dataset falls back to its previous behaviour with no macro
features. So the feature set is opt-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from constellation_quant.data._paths import DataPaths
from constellation_quant.utils import get_logger

log = get_logger(__name__)


# Yahoo Finance tickers for each macro series.
MACRO_TICKERS: Dict[str, str] = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
    "spy": "SPY",
}

# Feature names emitted in fixed order. 5-day log change of each series.
# Lightweight and capture regime shifts without exploding the slow-feature
# count (4 macro features added; was 8 stock-specific slow features).
MACRO_FEATURE_COLUMNS = [
    "vix_change_5d",
    "tnx_change_5d",
    "dxy_return_5d",
    "spy_return_5d",
]

# Lookback window in trading days for the change calculation.
_MACRO_CHANGE_WINDOW = 5


def macro_dir(paths: DataPaths) -> Path:
    """Return the directory where macro parquets live."""
    return paths.data_dir / "raw" / "macro"


def macro_file(paths: DataPaths, name: str) -> Path:
    """Path of a single macro parquet (one per series)."""
    return macro_dir(paths) / f"{name}.parquet"


# ── Downloader ────────────────────────────────────────────────────────────


def download_macro(
    paths: DataPaths,
    start: str = "1990-01-01",
    end:   Optional[str] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """Fetch each macro ticker via yfinance and save as parquet.

    No-op for tickers that already have a parquet on disk unless `force=True`.
    Returns a dict {name: path-on-disk} for every series we have.
    """
    import yfinance as yf

    macro_dir(paths).mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    written: Dict[str, Path] = {}
    for name, ticker in MACRO_TICKERS.items():
        out_path = macro_file(paths, name)
        if out_path.exists() and not force:
            log.info("macro: {} already cached at {}", name, out_path.name)
            written[name] = out_path
            continue

        log.info("macro: fetching {} ({}) {} -> {}", name, ticker, start, end)
        try:
            df = yf.Ticker(ticker).history(
                start=start, end=end,
                auto_adjust=False, actions=False,
            )
        except Exception as exc:                                # noqa: BLE001
            log.warning("macro: {} fetch failed: {}", name, exc)
            continue
        if df is None or df.empty:
            log.warning("macro: {} returned empty frame", name)
            continue

        # Normalise to a small parquet with `date` and `close` columns.
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        close_col = "Close" if "Close" in df.columns else (
            "Adj Close" if "Adj Close" in df.columns else df.columns[1]
        )
        # yfinance returns tz-aware dates — strip TZ so the parquet round-trips
        # cleanly through `MacroFeatures.from_paths` without TypeError on
        # tz-naive vs tz-aware comparison.
        dates = pd.to_datetime(df[date_col])
        if dates.dt.tz is not None:
            dates = dates.dt.tz_localize(None)
        out = pd.DataFrame({
            "date":  dates.dt.normalize(),
            "close": pd.to_numeric(df[close_col], errors="coerce").astype("float32"),
        }).dropna()
        out.to_parquet(out_path, index=False)
        log.info("macro: wrote {} rows for {} -> {}", len(out), name, out_path)
        written[name] = out_path

    return written


# ── Loader ────────────────────────────────────────────────────────────────


@dataclass
class MacroFeatures:
    """Date-indexed cache of macro series. Returns 5-day log changes per date.

    Construction loads each series from `data/raw/macro/<name>.parquet`.
    Series with no parquet on disk are silently skipped — their feature
    contribution becomes zero. Use `MacroFeatures.is_empty()` to check.
    """
    series: Dict[str, pd.Series]                 # name -> Series indexed by date

    @classmethod
    def from_paths(cls, paths: DataPaths) -> "MacroFeatures":
        series: Dict[str, pd.Series] = {}
        for name in MACRO_TICKERS:
            p = macro_file(paths, name)
            if not p.exists():
                continue
            df = pd.read_parquet(p)
            dates = pd.to_datetime(df["date"])
            # yfinance returns tz-aware timestamps which then get serialised
            # into the parquet. Our pred_date timestamps are tz-naive so
            # comparing them against a tz-aware index throws TypeError on
            # `s.loc[:ts]` — strip the timezone here so both sides match.
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
            df["date"] = dates.dt.normalize()
            df = df.drop_duplicates(subset=["date"]).sort_values("date")
            series[name] = df.set_index("date")["close"].astype(float)
        if series:
            log.info("MacroFeatures loaded {} series: {}",
                     len(series), sorted(series.keys()))
        else:
            log.info("MacroFeatures: no parquet files found at {} — feature contribution will be zero",
                     macro_dir(paths))
        return cls(series=series)

    @classmethod
    def empty(cls) -> "MacroFeatures":
        return cls(series={})

    def is_empty(self) -> bool:
        return not self.series

    @property
    def n_features(self) -> int:
        return len(MACRO_FEATURE_COLUMNS)

    def get_features(self, pred_date: pd.Timestamp) -> np.ndarray:
        """Return a (4,) float32 array of macro changes at `pred_date`.

        Each element is `log(price_today / price_5_business_days_ago)` for
        the corresponding series. Missing series → zero contribution.
        Missing dates → zero (we never NaN-pollute the model input).
        """
        out = np.zeros(self.n_features, dtype=np.float32)
        if not self.series:
            return out
        ts = pd.Timestamp(pred_date).normalize()
        feature_to_series = {
            "vix_change_5d": "vix",
            "tnx_change_5d": "tnx",
            "dxy_return_5d": "dxy",
            "spy_return_5d": "spy",
        }
        for i, feat in enumerate(MACRO_FEATURE_COLUMNS):
            name = feature_to_series[feat]
            s = self.series.get(name)
            if s is None or s.empty:
                continue
            # Most-recent close at or before pred_date.
            valid = s.loc[:ts]
            if valid.size < _MACRO_CHANGE_WINDOW + 1:
                continue
            now = float(valid.iloc[-1])
            prev = float(valid.iloc[-(_MACRO_CHANGE_WINDOW + 1)])
            if now > 0 and prev > 0 and np.isfinite(now) and np.isfinite(prev):
                out[i] = float(np.log(now / prev))
        # Defensive: clip extreme values to ±1.0 (a 5-day log change of ±1 is
        # already a >170% move; anything larger is almost certainly a data
        # glitch, not signal worth feeding the model).
        np.clip(out, -1.0, 1.0, out=out)
        return out
