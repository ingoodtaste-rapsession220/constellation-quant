"""Prepare filings-derived features for integration with constellation_quant.

The constellation_quant dataset takes per-stock-per-day slow features as a
broadcast tensor. Filings are per-company-per-quarter events, so we need to:

  1. Take per-filing features (drift_qoq, drift_yoy, drift_peer, sentiment)
  2. Forward-fill them across all trading days from the filing date until
     the next filing
  3. Map cik -> ticker so the existing dataset code can join by ticker.

Output is a parquet keyed on (ticker, date) with the new feature columns,
ready to be merged into the existing slow_features path.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Columns we contribute to the slow-feature stack
NEW_SLOW_FEATURES = [
    "drift_qoq",
    "drift_yoy",
    "drift_peer",
    "sentiment_mean",
    "sentiment_frac_neg",
    "sentiment_frac_pos",
]


def build_filings_features(
    drift_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    cik_to_ticker: dict[str, str],
    trading_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build the per-(ticker, date) wide features frame.

    Parameters
    ----------
    drift_df : per-filing drift features. Columns at minimum:
        cik, period, form, drift_qoq, drift_yoy, drift_peer
    sentiment_df : per-filing sentiment summary. Columns at minimum:
        cik, period, form, section, mean_score, frac_negative, frac_positive
        (we use only the MD&A section).
    cik_to_ticker : dict mapping zero-padded CIK -> ticker.
    trading_days : the full trading-day index we should forward-fill across.

    Returns
    -------
    DataFrame with columns:
        ticker, date, drift_qoq, drift_yoy, drift_peer,
        sentiment_mean, sentiment_frac_neg, sentiment_frac_pos
    """
    # Filter sentiment to MD&A only.
    sent_mda = sentiment_df[sentiment_df["section"] == "mda"].copy()
    sent_mda = sent_mda.rename(columns={
        "mean_score": "sentiment_mean",
        "frac_negative": "sentiment_frac_neg",
        "frac_positive": "sentiment_frac_pos",
    })[[
        "cik", "period", "form",
        "sentiment_mean", "sentiment_frac_neg", "sentiment_frac_pos",
    ]]

    merged = drift_df.merge(
        sent_mda, on=["cik", "period", "form"], how="outer",
    )
    merged["ticker"] = merged["cik"].map(cik_to_ticker)
    merged = merged.dropna(subset=["ticker"])
    merged["period"] = pd.to_datetime(merged["period"])
    merged = merged.sort_values(["ticker", "period"]).reset_index(drop=True)

    # Forward-fill per ticker across trading days.
    out_frames = []
    for ticker, grp in merged.groupby("ticker"):
        grp = grp[["period"] + NEW_SLOW_FEATURES].copy()
        grp = grp.set_index("period")
        # Reindex onto trading_days, forward-filling from each filing date.
        # Filings dated AFTER a trading day shouldn't leak — only
        # forward-fill, never backfill.
        grp = grp.reindex(trading_days, method="ffill")
        grp = grp.reset_index().rename(columns={"index": "date"})
        grp["ticker"] = ticker
        out_frames.append(grp)

    if not out_frames:
        return pd.DataFrame(columns=["ticker", "date"] + NEW_SLOW_FEATURES)
    out = pd.concat(out_frames, ignore_index=True)
    cols = ["ticker", "date"] + NEW_SLOW_FEATURES
    return out[cols]


def write_features_parquet(
    features_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Write features to parquet, with a sane partition layout."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(out_path, index=False, compression="snappy")
    logger.info(
        "Wrote %d rows × %d cols → %s",
        len(features_df), len(features_df.columns), out_path,
    )
    return out_path
