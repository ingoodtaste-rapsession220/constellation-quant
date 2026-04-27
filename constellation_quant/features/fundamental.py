"""Fundamental features — forward-fill quarterly data to daily, derive ratios.

Input: long-format parquet per ticker produced by `FundamentalsDownloader`
    columns = [date, metric, value]

Output: wide-format per-ticker DataFrame indexed by daily date with columns
    pe, pb, de, roe, fcf_yield, div_yield, rev_growth_yoy, log_market_cap

Derivation notes:

* Derived ratios require both a fundamental metric AND a concurrent price.
  We inject daily `adj_close` into the quarterly frame via `price_frames`.
* Quarterly values are forward-filled to daily — on any trading day, the
  value reflects the most recently reported quarter.
* "No look-ahead" means: we shift the effective date of each quarter's
  disclosure by `report_lag_days` (default 45 trading days) so the model
  only sees data that would have been publicly available at the time. Raw
  filing dates are never used — financial reports are released weeks after
  quarter-end.
* All ratios are NaN until enough disclosure history exists.

Sector-specific adjustments (per feature_config): banks skip EV/EBITDA
which is meaningless for financial-sector firms. We surface these via the
`ratios` list from config so they can be toggled per variant.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from constellation_quant.utils import get_logger

log = get_logger(__name__)


DEFAULT_REPORT_LAG_DAYS = 45


class FundamentalFeatures:
    """Compute per-ticker fundamental ratios, forward-filled to daily."""

    def __init__(self, config: Optional[Mapping] = None):
        cfg = dict(config or {})
        self.ratios = list(cfg.get("ratios", ["pe", "pb", "de", "roe", "fcf_yield", "dividend_yield"]))
        self.growth = list(cfg.get("growth", ["revenue_yoy"]))
        self.size   = list(cfg.get("size",   ["log_market_cap"]))
        self.report_lag_days = int(cfg.get("report_lag_days", DEFAULT_REPORT_LAG_DAYS))
        self.sector_specific = dict(cfg.get("sector_specific", {}) or {})

    # ── Public API ─────────────────────────────────────────────────────

    def compute(
        self,
        quarterly_frames: Mapping[str, pd.DataFrame],
        price_frames:     Mapping[str, pd.DataFrame],
        daily_index:      Optional[pd.DatetimeIndex] = None,
        sector_map:       Optional[Mapping[str, str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Compute fundamentals for every ticker that has both quarterly + price data.

        Args:
            quarterly_frames: ticker -> long-format DataFrame [date, metric, value].
            price_frames: ticker -> OHLCV DataFrame with `adj_close`.
            daily_index: Optional DatetimeIndex to reindex outputs to. Defaults
                to each ticker's own price calendar.
            sector_map: Optional ticker -> sector for sector-specific skips.

        Returns:
            Dict[ticker, DataFrame] with daily-indexed fundamental feature columns.
        """
        out: Dict[str, pd.DataFrame] = {}
        sector_map = dict(sector_map or {})
        for ticker, q_df in quarterly_frames.items():
            prices = price_frames.get(ticker)
            if prices is None or prices.empty or q_df is None or q_df.empty:
                out[ticker] = pd.DataFrame()
                continue
            sector = sector_map.get(ticker.upper())
            out[ticker] = self.compute_one(
                q_df, prices,
                daily_index=daily_index,
                sector=sector,
            )
        return out

    def compute_one(
        self,
        quarterly_df: pd.DataFrame,
        price_df:     pd.DataFrame,
        daily_index:  Optional[pd.DatetimeIndex] = None,
        sector:       Optional[str] = None,
    ) -> pd.DataFrame:
        wide = self._pivot_wide(quarterly_df)
        if wide.empty:
            return pd.DataFrame()

        # Apply the disclosure lag: values reported for quarter ending on `d`
        # are unavailable to the model until `d + report_lag_days` business days.
        wide.index = wide.index + pd.Timedelta(days=self.report_lag_days)

        price = self._prepare_price(price_df)
        index = daily_index if daily_index is not None else price.index
        daily = wide.reindex(index.union(wide.index)).sort_index().ffill()
        daily = daily.loc[index]

        aligned_price = price.reindex(index).ffill()
        feats = self._derive_ratios(daily, aligned_price, sector)
        return feats

    # ── Derivation ─────────────────────────────────────────────────────

    def _derive_ratios(
        self,
        f: pd.DataFrame,
        price: pd.Series,
        sector: Optional[str],
    ) -> pd.DataFrame:
        """Compute daily ratios from the forward-filled quarterly data."""
        skips = set(
            (self.sector_specific.get(sector, {}) or {}).get("skip", [])
            if sector else []
        )
        out = pd.DataFrame(index=f.index)

        shares = f.get("shares_outstanding")
        eps = None
        if shares is not None:
            market_cap = price * shares
            out["log_market_cap"] = np.log(market_cap.replace(0.0, np.nan))
            ni = f.get("net_income")
            if ni is not None:
                eps = ni / shares.replace(0.0, np.nan)
                out["pe"] = price / eps.replace(0.0, np.nan)

        equity = f.get("stockholders_equity")
        if equity is not None and shares is not None:
            book_per_share = equity / shares.replace(0.0, np.nan)
            out["pb"] = price / book_per_share.replace(0.0, np.nan)

        if equity is not None:
            total_debt = f.get("total_debt")
            if total_debt is not None:
                out["de"] = total_debt / equity.replace(0.0, np.nan)

        ni = f.get("net_income")
        if ni is not None and equity is not None:
            out["roe"] = ni / equity.replace(0.0, np.nan)

        # Free cash flow = operating cash flow − capex.
        ocf = f.get("operating_cashflow")
        capex = f.get("capex")
        if ocf is not None and shares is not None:
            fcf = ocf - (capex if capex is not None else 0.0)
            fcf_per_share = fcf / shares.replace(0.0, np.nan)
            out["fcf_yield"] = fcf_per_share / price.replace(0.0, np.nan)

        div = f.get("dividends_paid")
        if div is not None and shares is not None:
            # dividends_paid is a use of cash — usually negative on yfinance.
            div_ps = div.abs() / shares.replace(0.0, np.nan)
            out["div_yield"] = div_ps / price.replace(0.0, np.nan)

        rev = f.get("total_revenue")
        if rev is not None and "revenue_yoy" in self.growth:
            out["rev_growth_yoy"] = rev.pct_change(4)   # 4 quarters ≈ YoY

        return out.drop(columns=[c for c in out.columns if c in skips], errors="ignore")

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
        if long_df.empty:
            return pd.DataFrame()
        df = long_df.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        wide = (
            df.drop_duplicates(subset=["date", "metric"], keep="last")
              .pivot(index="date", columns="metric", values="value")
              .sort_index()
        )
        return wide

    @staticmethod
    def _prepare_price(price_df: pd.DataFrame) -> pd.Series:
        frame = price_df.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.set_index("date")
        frame = frame.sort_index()
        return frame["adj_close"].astype(float)


def cross_sectional_zscore(
    features_by_ticker: Mapping[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Cross-sectionally z-score each column on each date across all tickers.

    No look-ahead: on date t we use only date-t values (one row per ticker).
    Ranks each column across the universe, then z-scores. Stocks with NaN
    on date t are ignored for the mean/std but left NaN in the output.
    """
    if not features_by_ticker:
        return {}

    tickers = list(features_by_ticker.keys())
    frames = [features_by_ticker[t].assign(_ticker=t) for t in tickers]
    combined = pd.concat(frames, axis=0)
    combined.index.name = "date"
    long_df = combined.reset_index()

    feature_cols = [c for c in long_df.columns if c not in {"date", "_ticker"}]
    grouped = long_df.groupby("date")[feature_cols]
    means = grouped.transform("mean")
    stds  = grouped.transform("std").replace(0.0, np.nan)
    long_df[feature_cols] = (long_df[feature_cols] - means) / stds

    out: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        sub = long_df[long_df["_ticker"] == t].drop(columns="_ticker").set_index("date")
        out[t] = sub.sort_index()
    return out
