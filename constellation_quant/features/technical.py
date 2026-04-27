"""Technical indicators computed from OHLCV data.

All features are vectorised using pandas rolling operations. The primary API
is `TechnicalFeatures.compute(per_ticker_frames)` which iterates a dict of
per-ticker DataFrames (output of `DataCleaner.clean_batch`) and returns a
matching dict of feature DataFrames indexed by `date`.

Feature set (configured via `feature_config.yaml` > technical):
    - log_returns (1d, 5d, 20d)
    - realized_volatility (5d, 20d, 60d) — rolling std of 1d log returns
    - rsi_14
    - macd (line, signal, histogram)
    - bollinger_width (20d, 2σ)
    - atr_14
    - obv
    - relative_volume (current / 20d mean)
    - intraday_range ((high - low) / close)
    - gap ((open - prev_close) / prev_close)

The builder is idempotent: calling `.compute(df)` twice yields the same frame.
Missing values (propagated from rolling warmup) stay as NaN and are handled
downstream by the `Normalizer` (fill with cross-sectional median).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np
import pandas as pd

from constellation_quant.utils import get_logger

log = get_logger(__name__)


# Default config if feature_config.yaml's `technical` section is absent.
_DEFAULTS = {
    "log_returns": {"periods": [1, 5, 20]},
    "realized_volatility": {"periods": [5, 20, 60]},
    "rsi": {"period": 14},
    "macd": {"fast": 12, "slow": 26, "signal": 9},
    "bollinger": {"period": 20, "num_std": 2},
    "atr": {"period": 14},
    "obv": {"enabled": True},
    "relative_volume": {"window": 20},
    "intraday_range": {"enabled": True},
    "gap": {"enabled": True},
}


class TechnicalFeatures:
    """Compute technical indicators for each ticker."""

    def __init__(self, config: Optional[Mapping] = None):
        # Shallow merge of user config onto defaults so partial configs work.
        indicators = {}
        user_cfg = (config or {}).get("indicators", {}) or {}
        for key, default in _DEFAULTS.items():
            merged = dict(default)
            merged.update(user_cfg.get(key, {}) or {})
            indicators[key] = merged
        self.cfg = indicators

    # ── Public API ─────────────────────────────────────────────────────

    def compute(
        self,
        frames: Mapping[str, pd.DataFrame],
    ) -> Dict[str, pd.DataFrame]:
        """Compute features for each ticker. Returns a new dict; inputs untouched."""
        out: Dict[str, pd.DataFrame] = {}
        for ticker, df in frames.items():
            if df is None or df.empty:
                out[ticker] = pd.DataFrame()
                continue
            out[ticker] = self.compute_one(df)
        return out

    def compute_one(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features for a single ticker's OHLCV frame.

        Args:
            df: Must contain columns [date, open, high, low, close, adj_close, volume].
                `date` may be a column or the index.

        Returns:
            A DataFrame indexed by `date` whose columns are the computed
            indicator values. No lookahead: every value at date `t` uses only
            data from dates `<= t`.
        """
        frame = self._prepare(df)
        feats: Dict[str, pd.Series] = {}

        self._add_log_returns(frame, feats)
        self._add_volatility(feats)
        self._add_rsi(frame, feats)
        self._add_macd(frame, feats)
        self._add_bollinger(frame, feats)
        self._add_atr(frame, feats)
        self._add_obv(frame, feats)
        self._add_relative_volume(frame, feats)
        self._add_intraday_range(frame, feats)
        self._add_gap(frame, feats)

        return pd.DataFrame(feats, index=frame.index)

    # ── Indicators ─────────────────────────────────────────────────────

    def _add_log_returns(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        periods = self.cfg["log_returns"]["periods"]
        px = frame["adj_close"].astype(float)
        for p in periods:
            feats[f"ret_{p}d"] = np.log(px / px.shift(p))

    def _add_volatility(self, feats: Dict[str, pd.Series]) -> None:
        periods = self.cfg["realized_volatility"]["periods"]
        r1 = feats.get("ret_1d")
        if r1 is None:
            return
        for p in periods:
            feats[f"vol_{p}d"] = r1.rolling(p, min_periods=p).std()

    def _add_rsi(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        period = int(self.cfg["rsi"]["period"])
        delta = frame["adj_close"].astype(float).diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, np.nan)
        rsi = 100 - (100 / (1.0 + rs))
        feats[f"rsi_{period}"] = rsi

    def _add_macd(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        c = self.cfg["macd"]
        fast, slow, signal = int(c["fast"]), int(c["slow"]), int(c["signal"])
        px = frame["adj_close"].astype(float)
        ema_fast = px.ewm(span=fast, adjust=False).mean()
        ema_slow = px.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        feats["macd"] = macd_line
        feats["macd_signal"] = signal_line
        feats["macd_hist"] = macd_line - signal_line

    def _add_bollinger(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        c = self.cfg["bollinger"]
        n, k = int(c["period"]), float(c["num_std"])
        px = frame["adj_close"].astype(float)
        mean = px.rolling(n, min_periods=n).mean()
        std = px.rolling(n, min_periods=n).std()
        upper = mean + k * std
        lower = mean - k * std
        # Width, normalised by price, always positive.
        feats[f"bbw_{n}"] = (upper - lower) / mean

    def _add_atr(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        period = int(self.cfg["atr"]["period"])
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        prev_close = frame["close"].astype(float).shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        feats[f"atr_{period}"] = tr.ewm(
            alpha=1.0 / period, min_periods=period, adjust=False
        ).mean()

    def _add_obv(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        if not self.cfg["obv"].get("enabled", True):
            return
        close = frame["close"].astype(float)
        volume = frame["volume"].astype(float)
        direction = np.sign(close.diff().fillna(0.0))
        feats["obv"] = (direction * volume).cumsum()

    def _add_relative_volume(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        window = int(self.cfg["relative_volume"]["window"])
        vol = frame["volume"].astype(float)
        avg = vol.rolling(window, min_periods=window).mean()
        feats[f"relvol_{window}"] = vol / avg

    def _add_intraday_range(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        if not self.cfg["intraday_range"].get("enabled", True):
            return
        high = frame["high"].astype(float)
        low = frame["low"].astype(float)
        close = frame["close"].astype(float)
        feats["intraday_range"] = (high - low) / close.replace(0.0, np.nan)

    def _add_gap(self, frame: pd.DataFrame, feats: Dict[str, pd.Series]) -> None:
        if not self.cfg["gap"].get("enabled", True):
            return
        open_ = frame["open"].astype(float)
        prev_close = frame["close"].astype(float).shift(1)
        feats["gap"] = (open_ - prev_close) / prev_close

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure `date` is the index and rows are sorted ascending."""
        frame = df.copy()
        if "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
            frame = frame.set_index("date")
        frame = frame.sort_index()
        for col in ("open", "high", "low", "close", "adj_close", "volume"):
            if col not in frame.columns:
                raise KeyError(f"TechnicalFeatures requires column '{col}'")
        return frame

    def feature_names(self) -> list[str]:
        """Return the stable ordered list of feature column names."""
        names: list[str] = []
        names += [f"ret_{p}d" for p in self.cfg["log_returns"]["periods"]]
        names += [f"vol_{p}d" for p in self.cfg["realized_volatility"]["periods"]]
        names += [f"rsi_{self.cfg['rsi']['period']}"]
        names += ["macd", "macd_signal", "macd_hist"]
        names += [f"bbw_{self.cfg['bollinger']['period']}"]
        names += [f"atr_{self.cfg['atr']['period']}"]
        if self.cfg["obv"].get("enabled", True):
            names += ["obv"]
        names += [f"relvol_{self.cfg['relative_volume']['window']}"]
        if self.cfg["intraday_range"].get("enabled", True):
            names += ["intraday_range"]
        if self.cfg["gap"].get("enabled", True):
            names += ["gap"]
        return names
