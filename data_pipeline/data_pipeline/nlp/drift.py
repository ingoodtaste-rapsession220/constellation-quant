"""Quarter-over-quarter language-drift features.

The "Lazy Prices" signal (Cohen, Malloy, Pomorski 2020) measures how much a
company's filing language has drifted relative to its own recent history.
The classic implementation:

  drift_t = 1 - cosine_similarity(embed(filing_t), embed(filing_{t-1}))

We expose three flavours per (cik, period):
  - drift_qoq   — vs the company's most recent prior filing (same form)
  - drift_yoy   — vs the same company's filing one year earlier
  - drift_peer  — vs the average embedding of the company's GICS sector
                  in the same period (cross-sectional language drift)

These are scalar features per-stock-per-filing-date, ready to broadcast
to every trading day in the quarter following the filing for downstream
consumption.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine_similarity, with NaN handling for zero vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def compute_drift_features(
    embeddings_df: pd.DataFrame,
    *,
    section: str = "mda",
    sector_map: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """Build per-filing drift features.

    Parameters
    ----------
    embeddings_df : DataFrame with columns:
        cik, accession, form, filing_date, period, section_name, embedding
        where `embedding` is a list/np.ndarray of floats.
    section : which section to score on. The default ('mda') is the
        management's-discussion-and-analysis section, where Lazy Prices
        signal is strongest.
    sector_map : optional ticker/cik -> sector dict for the peer-drift
        feature. If None, peer drift is left as NaN.

    Returns
    -------
    DataFrame indexed by (cik, period, form), with columns:
        drift_qoq, drift_yoy, drift_peer
    """
    df = embeddings_df.copy()
    df = df[df["section_name"] == section].copy()
    if df.empty:
        return pd.DataFrame(columns=["drift_qoq", "drift_yoy", "drift_peer"])

    # Ensure embeddings are np.ndarrays
    df["embedding"] = df["embedding"].apply(
        lambda x: np.asarray(x, dtype=np.float32)
    )

    df["period_dt"] = pd.to_datetime(df["period"])
    df = df.sort_values(["cik", "form", "period_dt"]).reset_index(drop=True)

    # ------------- Q-over-Q drift: prev same form, same CIK
    drift_qoq = []
    drift_yoy = []
    grouped = df.groupby(["cik", "form"], sort=False)
    for (_, _), grp in grouped:
        embs = grp["embedding"].tolist()
        periods = grp["period_dt"].tolist()
        for i in range(len(grp)):
            qoq = float("nan")
            yoy = float("nan")
            if i > 0:
                qoq = _cosine_distance(embs[i], embs[i - 1])
            # YoY: most recent prior filing >= 270d earlier
            target_lo = periods[i] - pd.Timedelta(days=395)
            target_hi = periods[i] - pd.Timedelta(days=270)
            for j in range(i - 1, -1, -1):
                if target_lo <= periods[j] <= target_hi:
                    yoy = _cosine_distance(embs[i], embs[j])
                    break
            drift_qoq.append(qoq)
            drift_yoy.append(yoy)
    df["drift_qoq"] = drift_qoq
    df["drift_yoy"] = drift_yoy

    # ------------- peer drift: vs sector-mean embedding for the same period
    if sector_map is not None:
        df["sector"] = df["cik"].map(sector_map)
        peer_drifts = []
        # Bucket by (sector, period) — average embedding across same-sector peers
        sector_period_embeds: dict[tuple[str, pd.Timestamp], np.ndarray] = {}
        for (sec, per), grp in df.groupby(["sector", "period_dt"]):
            stack = np.stack(grp["embedding"].tolist())
            sector_period_embeds[(sec, per)] = stack.mean(axis=0)
        for _, row in df.iterrows():
            key = (row["sector"], row["period_dt"])
            if key in sector_period_embeds:
                peer_drifts.append(_cosine_distance(row["embedding"], sector_period_embeds[key]))
            else:
                peer_drifts.append(float("nan"))
        df["drift_peer"] = peer_drifts
    else:
        df["drift_peer"] = float("nan")

    out = (
        df[[
            "cik", "period", "form",
            "drift_qoq", "drift_yoy", "drift_peer",
        ]]
        .reset_index(drop=True)
    )
    return out
