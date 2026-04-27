"""Portfolio backtest engine.

Given a stream of daily score predictions, simulates:
    - weekly (or configurable) rebalance on the chosen `PortfolioConstructor`
    - daily P&L via per-stock total-return between rebalances
    - transaction costs deducted on rebalance (default 5 bps per trade)
    - position drift — between rebalances, weights evolve with returns

Outputs a `BacktestResult` with daily returns, equity curve, drawdown
series, turnover series, and summary statistics (Sharpe, max DD, annual
return, annual vol, average turnover, total transaction cost).

Dynamic S&P 500 membership is honoured: the `members_fn(date)` callback
returns the tradeable universe for each date. Positions in newly-removed
names are force-closed at the next rebalance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from constellation_quant.evaluation.portfolio import (
    PortfolioConstructor,
    PortfolioWeights,
)
from constellation_quant.utils import get_logger

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


# ── Inputs ─────────────────────────────────────────────────────────────────


@dataclass
class DailyPrediction:
    """One date of model output to feed the backtester."""
    date: pd.Timestamp
    tickers: Sequence[str]
    scores: np.ndarray                           # (N,)
    mask: Optional[np.ndarray] = None            # (N,) — True for tradeable
    volatility: Optional[np.ndarray] = None      # (N,) predicted vol for risk parity


# ── Outputs ────────────────────────────────────────────────────────────────


@dataclass
class BacktestResult:
    daily_returns: pd.Series                     # index=date, values=net return
    equity_curve:  pd.Series
    drawdown:      pd.Series
    turnover:      pd.Series                     # turnover per rebalance day
    costs:         pd.Series                     # transaction cost per rebalance
    positions:     pd.DataFrame                  # rows=date, cols=ticker

    sharpe:         float
    annual_return:  float
    annual_vol:     float
    max_drawdown:   float
    avg_turnover:   float
    total_cost:     float
    final_equity:   float

    def summary(self) -> Dict[str, float]:
        return {
            "sharpe":        float(self.sharpe),
            "annual_return": float(self.annual_return),
            "annual_vol":    float(self.annual_vol),
            "max_drawdown":  float(self.max_drawdown),
            "avg_turnover":  float(self.avg_turnover),
            "total_cost":    float(self.total_cost),
            "final_equity":  float(self.final_equity),
            "n_days":        float(len(self.daily_returns)),
        }


# ── Engine ─────────────────────────────────────────────────────────────────


class Backtester:
    """Long-short portfolio simulator with transaction costs.

    Args:
        constructor: Portfolio construction strategy.
        transaction_cost_bps: Per-trade slippage + commission estimate.
            Applied to the absolute dollar change in each position.
        rebalance_frequency: One of 'D' (every day), 'W' (weekly),
            'M' (monthly). Weekly rebalances align on Mondays.
        members_fn: Optional callable `date -> iterable[ticker]` that filters
            the tradeable universe per date. Unspecified → use everything
            from each `DailyPrediction`.
        initial_capital: Starting equity (cosmetic — results scale linearly).
        risk_free_rate_annual: For Sharpe. Default 0 (excess-return Sharpe).
    """

    REBALANCE_FREQS = {"D", "W", "M"}

    def __init__(
        self,
        constructor: PortfolioConstructor,
        transaction_cost_bps: float = 5.0,
        rebalance_frequency: str = "W",
        members_fn: Optional[Callable[[pd.Timestamp], Iterable[str]]] = None,
        initial_capital: float = 1.0,
        risk_free_rate_annual: float = 0.0,
    ):
        if rebalance_frequency not in self.REBALANCE_FREQS:
            raise ValueError(
                f"rebalance_frequency must be one of {sorted(self.REBALANCE_FREQS)}, "
                f"got {rebalance_frequency!r}"
            )
        self.constructor = constructor
        self.cost_bps = float(transaction_cost_bps)
        self.rebalance_frequency = rebalance_frequency
        self.members_fn = members_fn
        self.initial_capital = float(initial_capital)
        self.rf_annual = float(risk_free_rate_annual)

    # ── Main loop ──────────────────────────────────────────────────────

    def run(
        self,
        predictions: Sequence[DailyPrediction],
        price_frames: Mapping[str, pd.DataFrame],
    ) -> BacktestResult:
        """Simulate. `price_frames` must cover every date in predictions."""
        if not predictions:
            raise ValueError("predictions is empty")

        prices_wide = _build_prices_wide(price_frames)
        prices_wide = prices_wide.sort_index()

        # Align to the prediction dates + one prior day for the first drift.
        pred_by_date = {pd.Timestamp(p.date).normalize(): p for p in predictions}
        pred_dates = sorted(pred_by_date.keys())
        start = pred_dates[0]
        end = pred_dates[-1]
        trading_calendar = prices_wide.loc[start:end].index

        if len(trading_calendar) == 0:
            raise ValueError("No overlap between price data and prediction dates.")

        rebalance_mask = self._rebalance_days(trading_calendar)
        rebalance_set = set(trading_calendar[rebalance_mask])

        current_weights: Dict[str, float] = {}
        prev_prices: Dict[str, float] = {}

        daily_returns: List[float] = []
        turnovers: Dict[pd.Timestamp, float] = {}
        costs: Dict[pd.Timestamp, float] = {}
        positions_log: List[Dict[str, float]] = []
        dates_log: List[pd.Timestamp] = []

        for i, date in enumerate(trading_calendar):
            # 1. Drift yesterday's weights by today's realised returns.
            if i > 0 and current_weights:
                daily_ret, current_weights = self._drift(
                    current_weights, prev_prices, prices_wide.loc[date],
                )
            else:
                daily_ret = 0.0

            # 2. Rebalance if today is a scheduled rebalance day and we have
            #    a fresh prediction.
            cost_today = 0.0
            if date in rebalance_set and date in pred_by_date:
                new_weights, turnover = self._rebalance(
                    pred=pred_by_date[date],
                    prices_today=prices_wide.loc[date],
                    current=current_weights,
                )
                cost_today = turnover * self.cost_bps / 1e4
                daily_ret -= cost_today
                current_weights = new_weights
                turnovers[date] = turnover
                costs[date] = cost_today

            daily_returns.append(daily_ret)
            positions_log.append(dict(current_weights))
            dates_log.append(date)
            prev_prices = {
                ticker: float(prices_wide.loc[date, ticker])
                for ticker in current_weights
                if ticker in prices_wide.columns
                   and np.isfinite(prices_wide.loc[date, ticker])
            }

        return self._summarise(
            trading_calendar, daily_returns, turnovers, costs,
            positions_log, dates_log,
        )

    # ── Components ─────────────────────────────────────────────────────

    def _rebalance(
        self,
        pred: DailyPrediction,
        prices_today: pd.Series,
        current: Mapping[str, float],
    ) -> tuple[Dict[str, float], float]:
        """Build new weights and compute turnover (Σ|Δw|) vs current."""
        mask = pred.mask
        if self.members_fn is not None:
            members = set(self.members_fn(pred.date))
            tick_mask = np.array([t in members for t in pred.tickers], dtype=bool)
            mask = tick_mask if mask is None else (mask.astype(bool) & tick_mask)

        # Require positive price on the rebalance day.
        has_price = np.array(
            [(t in prices_today.index and np.isfinite(prices_today.get(t, np.nan)))
             for t in pred.tickers],
            dtype=bool,
        )
        mask = has_price if mask is None else (mask.astype(bool) & has_price)

        target = self.constructor.construct(
            scores=pred.scores,
            tickers=pred.tickers,
            mask=mask,
            volatility=pred.volatility,
        )
        target_weights = dict(target.weights)

        # Turnover: sum of |new - old| across the union of tickers.
        all_tickers = set(target_weights) | set(current)
        turnover = sum(
            abs(target_weights.get(t, 0.0) - current.get(t, 0.0))
            for t in all_tickers
        )
        return target_weights, float(turnover)

    @staticmethod
    def _drift(
        weights: Mapping[str, float],
        prev_prices: Mapping[str, float],
        prices_today: pd.Series,
    ) -> tuple[float, Dict[str, float]]:
        """Apply today's returns to each position; return (portfolio_return, drifted_weights)."""
        total_return = 0.0
        drifted: Dict[str, float] = {}
        for ticker, weight in weights.items():
            if ticker not in prev_prices:
                continue
            p_prev = prev_prices[ticker]
            p_now = prices_today.get(ticker, np.nan)
            if not np.isfinite(p_now) or p_prev <= 0:
                drifted[ticker] = weight
                continue
            ret = p_now / p_prev - 1.0
            total_return += weight * ret
            drifted[ticker] = weight * (1.0 + ret)
        return float(total_return), drifted

    def _rebalance_days(self, calendar: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask over `calendar` marking rebalance days."""
        if self.rebalance_frequency == "D":
            return np.ones(len(calendar), dtype=bool)
        if self.rebalance_frequency == "W":
            # Rebalance on the first trading day of each ISO week.
            weeks = calendar.isocalendar().week.values
            years = calendar.isocalendar().year.values
            seen = set()
            mask = np.zeros(len(calendar), dtype=bool)
            for i, (w, y) in enumerate(zip(weeks, years)):
                if (y, w) not in seen:
                    seen.add((y, w))
                    mask[i] = True
            return mask
        # Monthly
        months = calendar.month.values
        years = calendar.year.values
        seen = set()
        mask = np.zeros(len(calendar), dtype=bool)
        for i, (m, y) in enumerate(zip(months, years)):
            if (y, m) not in seen:
                seen.add((y, m))
                mask[i] = True
        return mask

    # ── Summary metrics ────────────────────────────────────────────────

    def _summarise(
        self,
        calendar: pd.DatetimeIndex,
        returns: List[float],
        turnovers: Dict[pd.Timestamp, float],
        costs: Dict[pd.Timestamp, float],
        positions_log: List[Dict[str, float]],
        dates_log: List[pd.Timestamp],
    ) -> BacktestResult:
        ret_s = pd.Series(returns, index=calendar, name="return")
        equity = self.initial_capital * (1.0 + ret_s).cumprod()
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max

        # Annualised metrics
        mean_daily = ret_s.mean()
        vol_daily = ret_s.std(ddof=0)
        rf_daily = self.rf_annual / TRADING_DAYS_PER_YEAR
        sharpe = (
            float((mean_daily - rf_daily) / vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR))
            if vol_daily > 1e-12 else float("nan")
        )
        annual_ret = float((1.0 + mean_daily) ** TRADING_DAYS_PER_YEAR - 1.0)
        annual_vol = float(vol_daily * np.sqrt(TRADING_DAYS_PER_YEAR))

        turnover_s = pd.Series(turnovers, name="turnover").sort_index()
        costs_s    = pd.Series(costs,     name="cost").sort_index()
        all_tickers = set().union(*(p.keys() for p in positions_log if p))
        positions_df = pd.DataFrame(positions_log, index=dates_log,
                                     columns=sorted(all_tickers)).fillna(0.0)

        return BacktestResult(
            daily_returns=ret_s,
            equity_curve=equity,
            drawdown=drawdown,
            turnover=turnover_s,
            costs=costs_s,
            positions=positions_df,
            sharpe=sharpe,
            annual_return=annual_ret,
            annual_vol=annual_vol,
            max_drawdown=float(drawdown.min()) if len(drawdown) else 0.0,
            avg_turnover=float(turnover_s.mean()) if len(turnover_s) else 0.0,
            total_cost=float(costs_s.sum()) if len(costs_s) else 0.0,
            final_equity=float(equity.iloc[-1]) if len(equity) else self.initial_capital,
        )


# ── Helpers ────────────────────────────────────────────────────────────────


def _build_prices_wide(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """(ticker -> OHLCV frame) -> wide DataFrame of adj_close per ticker."""
    cols: List[pd.Series] = []
    for ticker, df in frames.items():
        if df is None or df.empty:
            continue
        frame = df.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.set_index("date")
        px = frame["adj_close"].astype(float).sort_index()
        cols.append(px.rename(ticker))
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index()
