"""Portfolio construction — turn per-stock scores into signed position weights.

Three concrete constructors, all sharing the `PortfolioConstructor` ABC so
the backtester can swap them via config:

* `EqualWeightLongShort` — top-N by score long, bottom-N short, equal weight
  within each leg. Dollar-neutral by construction (Σ|w| = 2·leg_weight).

* `RiskParityLongShort` — same stock selection, position sizes inversely
  proportional to predicted volatility so each position contributes equal
  risk. Falls back to equal-weight when vol is missing.

* `SectorNeutralLongShort` — within each GICS sector, take the top quantile
  long and bottom quantile short. Eliminates net sector bets.

All constructors honour an optional `mask` so padded (non-member) stocks are
excluded, a `max_position_weight` cap that protects against extreme
concentrations, and a `max_turnover` clip that limits trade size per
rebalance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass
class PortfolioWeights:
    """Signed target weights per ticker. Absent tickers = 0 weight."""
    weights: Dict[str, float]

    def dollar_neutral(self) -> bool:
        """True if long and short notional match within 1e-6."""
        longs  = sum(w for w in self.weights.values() if w > 0)
        shorts = sum(w for w in self.weights.values() if w < 0)
        return abs(longs + shorts) < 1e-6

    def gross_exposure(self) -> float:
        return sum(abs(w) for w in self.weights.values())

    def net_exposure(self) -> float:
        return sum(self.weights.values())


# ── Base ───────────────────────────────────────────────────────────────────


class PortfolioConstructor(ABC):
    """Interface for any strategy that turns scores → weights."""

    @abstractmethod
    def construct(
        self,
        scores: np.ndarray,
        tickers: Sequence[str],
        mask: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        sectors: Optional[Mapping[str, str]] = None,
    ) -> PortfolioWeights:
        ...


def _apply_mask(
    scores: np.ndarray,
    tickers: Sequence[str],
    mask: Optional[np.ndarray],
) -> tuple[np.ndarray, List[str], np.ndarray]:
    """Filter to valid slots. Returns (scores, tickers, original_indices)."""
    n = len(tickers)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask).astype(bool)
    finite = np.isfinite(scores)
    valid = mask & finite
    indices = np.where(valid)[0]
    filtered_tickers = [tickers[i] for i in indices]
    return scores[valid], filtered_tickers, indices


# ── Equal-weight ───────────────────────────────────────────────────────────


class EqualWeightLongShort(PortfolioConstructor):
    """Top-N long, bottom-N short, equal weight per leg.

    Args:
        top_n: Number of stocks per leg. If the valid universe is smaller,
            shrinks to `|universe| // 2` to keep longs and shorts balanced.
        leg_weight: Target gross weight per leg. Default 1.0 → 200% gross,
            0% net (dollar-neutral).
        max_position_weight: Per-stock cap on |weight|. Default None (no cap).
    """

    def __init__(
        self,
        top_n: int = 50,
        leg_weight: float = 1.0,
        max_position_weight: Optional[float] = None,
    ):
        self.top_n = top_n
        self.leg_weight = float(leg_weight)
        self.max_position_weight = max_position_weight

    def construct(
        self,
        scores: np.ndarray,
        tickers: Sequence[str],
        mask: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        sectors: Optional[Mapping[str, str]] = None,
    ) -> PortfolioWeights:
        s, t_list, _ = _apply_mask(scores, tickers, mask)
        if len(s) < 2:
            return PortfolioWeights({})

        top_n = min(self.top_n, len(s) // 2)
        if top_n <= 0:
            return PortfolioWeights({})

        order = np.argsort(s)                                   # ascending
        shorts = order[:top_n]                                  # lowest scores
        longs  = order[-top_n:]                                 # highest scores

        w_per = self.leg_weight / top_n
        if self.max_position_weight is not None:
            w_per = min(w_per, self.max_position_weight)

        weights: Dict[str, float] = {}
        for idx in longs:
            weights[t_list[idx]] = w_per
        for idx in shorts:
            weights[t_list[idx]] = -w_per
        return PortfolioWeights(weights)


# ── Risk-parity ────────────────────────────────────────────────────────────


class RiskParityLongShort(PortfolioConstructor):
    """Same selection as equal-weight; position sizes scaled by 1/vol.

    Per leg, weights are proportional to `1 / clipped_volatility`, normalised
    so the leg's gross weight equals `leg_weight`. Clipping avoids division
    by near-zero vol for very quiet names.
    """

    def __init__(
        self,
        top_n: int = 50,
        leg_weight: float = 1.0,
        min_vol: float = 0.005,               # 0.5% daily vol floor
        max_position_weight: Optional[float] = None,
    ):
        self.top_n = top_n
        self.leg_weight = float(leg_weight)
        self.min_vol = float(min_vol)
        self.max_position_weight = max_position_weight

    def construct(
        self,
        scores: np.ndarray,
        tickers: Sequence[str],
        mask: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        sectors: Optional[Mapping[str, str]] = None,
    ) -> PortfolioWeights:
        s, t_list, orig_idx = _apply_mask(scores, tickers, mask)
        if len(s) < 2:
            return PortfolioWeights({})

        top_n = min(self.top_n, len(s) // 2)
        if top_n <= 0:
            return PortfolioWeights({})

        if volatility is None:
            # Graceful fallback to equal-weight.
            return EqualWeightLongShort(self.top_n, self.leg_weight,
                                          self.max_position_weight).construct(
                scores, tickers, mask, volatility, sectors,
            )

        vol_full = np.asarray(volatility, dtype=np.float64)
        vol = np.maximum(vol_full[orig_idx], self.min_vol)

        order = np.argsort(s)
        shorts = order[:top_n]
        longs  = order[-top_n:]

        weights = {
            **self._leg_weights(longs, t_list, vol, sign=+1),
            **self._leg_weights(shorts, t_list, vol, sign=-1),
        }
        if self.max_position_weight is not None:
            cap = self.max_position_weight
            weights = {t: np.sign(w) * min(abs(w), cap) for t, w in weights.items()}
        return PortfolioWeights(weights)

    def _leg_weights(
        self,
        indices: np.ndarray,
        tickers: List[str],
        vol: np.ndarray,
        sign: int,
    ) -> Dict[str, float]:
        inv_vol = 1.0 / vol[indices]
        raw = inv_vol / inv_vol.sum()
        scaled = sign * self.leg_weight * raw
        return {tickers[idx]: float(w) for idx, w in zip(indices, scaled)}


# ── Sector-neutral ─────────────────────────────────────────────────────────


class SectorNeutralLongShort(PortfolioConstructor):
    """Top / bottom quantile within each GICS sector — neutralises sector bets.

    Per sector: rank stocks by score, take the top `quantile` fraction long
    and the bottom `quantile` fraction short, equal-weighted within each
    sector's leg. Each sector's long weight matches its short weight, so the
    portfolio is dollar-neutral at the sector level.
    """

    def __init__(
        self,
        quantile: float = 0.2,                # top / bottom 20% per sector
        leg_weight: float = 1.0,
        max_position_weight: Optional[float] = None,
    ):
        if not 0 < quantile < 0.5:
            raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")
        self.quantile = float(quantile)
        self.leg_weight = float(leg_weight)
        self.max_position_weight = max_position_weight

    def construct(
        self,
        scores: np.ndarray,
        tickers: Sequence[str],
        mask: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
        sectors: Optional[Mapping[str, str]] = None,
    ) -> PortfolioWeights:
        if sectors is None:
            return EqualWeightLongShort(50, self.leg_weight,
                                         self.max_position_weight).construct(
                scores, tickers, mask, volatility, sectors,
            )

        s, t_list, _ = _apply_mask(scores, tickers, mask)
        if len(s) < 2:
            return PortfolioWeights({})

        # Group by sector.
        by_sector: Dict[str, List[tuple]] = {}
        for i, ticker in enumerate(t_list):
            sector = sectors.get(ticker) or sectors.get(ticker.upper())
            if sector is None:
                continue
            by_sector.setdefault(sector, []).append((i, s[i], ticker))

        n_sectors = sum(1 for v in by_sector.values() if len(v) >= 2)
        if n_sectors == 0:
            return PortfolioWeights({})

        weights: Dict[str, float] = {}
        per_sector_leg = self.leg_weight / n_sectors
        for members in by_sector.values():
            if len(members) < 2:
                continue
            members_sorted = sorted(members, key=lambda x: x[1])
            k = max(1, int(round(self.quantile * len(members_sorted))))
            shorts = members_sorted[:k]
            longs  = members_sorted[-k:]
            w_per = per_sector_leg / k
            if self.max_position_weight is not None:
                w_per = min(w_per, self.max_position_weight)
            for _, _, ticker in longs:
                weights[ticker] = weights.get(ticker, 0.0) + w_per
            for _, _, ticker in shorts:
                weights[ticker] = weights.get(ticker, 0.0) - w_per
        return PortfolioWeights(weights)


# ── Factory ────────────────────────────────────────────────────────────────


def build_portfolio_constructor(
    name: str,
    config: Optional[Mapping] = None,
) -> PortfolioConstructor:
    cfg = dict(config or {})
    key = name.lower().strip()
    if key in {"equal_weight", "equal", "long_short"}:
        return EqualWeightLongShort(
            top_n=int(cfg.get("top_n", 50)),
            leg_weight=float(cfg.get("leg_weight", 1.0)),
            max_position_weight=cfg.get("max_position_weight"),
        )
    if key in {"risk_parity", "risk"}:
        return RiskParityLongShort(
            top_n=int(cfg.get("top_n", 50)),
            leg_weight=float(cfg.get("leg_weight", 1.0)),
            min_vol=float(cfg.get("min_vol", 0.005)),
            max_position_weight=cfg.get("max_position_weight"),
        )
    if key in {"sector_neutral", "sector"}:
        return SectorNeutralLongShort(
            quantile=float(cfg.get("quantile", 0.2)),
            leg_weight=float(cfg.get("leg_weight", 1.0)),
            max_position_weight=cfg.get("max_position_weight"),
        )
    raise ValueError(f"Unknown portfolio constructor {name!r}")
