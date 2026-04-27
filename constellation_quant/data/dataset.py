"""PyTorch Dataset that serves variable-size daily graph samples.

Each sample corresponds to one **prediction date** `t` and returns:

    features: (N_max, L, F)  float32  — padded per-stock lookback window
    targets:  (N_max,)       float32  — H-day forward log return
    mask:     (N_max,)       bool     — True for stocks present on `t` with
                                         both a complete lookback and a valid
                                         target at `t+H`
    tickers:  List[str]               — length N_max, padded with ""
    sectors:  (N_max,)       int64    — sector index per stock (0 = unknown)
    date:     pd.Timestamp            — the prediction date

Key design points:

*   **Variable-size graphs**. `N_max` is the union of every ticker that ever
    appeared in the roster (and has a price file on disk). Each date, the
    mask selects the ~500 actual members; everything else is zero-padded.

*   **No look-ahead leakage**. Feature window is [t − L + 1 … t]; the target
    uses adj_close(t + H) / adj_close(t) − 1. We never touch data past `t`
    when computing features.

*   **Chronological**. `__getitem__(i)` maps to the i-th valid prediction
    date in the dataset's date range — DataLoader `shuffle=True` would
    break time-series assumptions and must not be used.

*   **Feature engine optional**. Phase 1 uses raw OHLCV. Phase 2 injects a
    `FeatureEngine` callable that receives the per-ticker lookback frame
    (shape [L, raw_cols]) and returns the per-ticker feature matrix
    (shape [L, F]).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from constellation_quant.data._paths import DataPaths
from constellation_quant.data.macro import MacroFeatures
from constellation_quant.data.membership import MembershipRoster, _coerce_date
from constellation_quant.utils import get_logger

log = get_logger(__name__)


# Default raw columns loaded from the price parquet when no feature engine is
# supplied. Order matters: this becomes the F dimension.
DEFAULT_RAW_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]

# Feature set that the technical-features fast path produces. 15 channels
# per day, per ticker. Bigger than raw OHLCV and carries the momentum /
# volatility / flow signal the Informer actually needs.
TECHNICAL_FEATURE_COLUMNS = [
    "ret_5d", "ret_20d",            # ret_1d dropped — too noisy for 5-day horizon
    "vol_5d", "vol_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bbw_20",
    "atr_14",
    "log_volume", "rel_volume_20",
    "intraday_range", "gap",
]

# ── Fast / slow split ─────────────────────────────────────────────────
# FAST features genuinely vary day-to-day — feeding the temporal encoder
# 60 days of these is information, not redundancy.
# SLOW features are smoothed by construction (rolling/EMA over 14-20 days)
# so 60 daily values are nearly identical. We collapse them to a single
# last-day snapshot and route them through a small static MLP, freeing
# the temporal encoder to focus on the actually-varying signal.
FAST_FEATURE_COLUMNS = [
    "ret_5d",                       # 1-day return dropped — too noisy for the
                                     # 5-day forward horizon. Revisit later
                                     # as a smoothed (3-day MA) signal or
                                     # via weekly-bar resampling.
    "vol_5d",
    "log_volume", "rel_volume_20",
    "intraday_range", "gap",
]
SLOW_FEATURE_COLUMNS = [
    "ret_20d",
    "vol_20d",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bbw_20",
    "atr_14",
]
assert set(FAST_FEATURE_COLUMNS) | set(SLOW_FEATURE_COLUMNS) == set(TECHNICAL_FEATURE_COLUMNS), (
    "FAST + SLOW must partition TECHNICAL_FEATURE_COLUMNS exactly"
)
assert set(FAST_FEATURE_COLUMNS).isdisjoint(SLOW_FEATURE_COLUMNS), (
    "FAST and SLOW must be disjoint"
)
_FAST_IDX = np.array([TECHNICAL_FEATURE_COLUMNS.index(c) for c in FAST_FEATURE_COLUMNS],
                      dtype=np.int64)
_SLOW_IDX = np.array([TECHNICAL_FEATURE_COLUMNS.index(c) for c in SLOW_FEATURE_COLUMNS],
                      dtype=np.int64)


FeatureFn = Callable[[pd.DataFrame], np.ndarray]
"""A feature engine is any callable that maps a DataFrame of shape (L, n_raw)
to a float array of shape (L, F). See `features.feature_engine` (Phase 2)."""


@dataclass
class SampleShapes:
    """Summary of tensor shapes for sanity-checking a Dataset.

    `n_features` is the count flowing through the temporal encoder. In
    technical mode this is the FAST subset (7); in OHLCV / feature-engine
    mode this is the full input dim. `n_slow_features` is non-zero only
    when the dataset emits a slow static branch (technical mode).
    """
    n_max: int
    lookback: int
    n_features: int
    n_samples: int
    n_slow_features: int = 0


class DynaGraphDataset(Dataset):
    """Serves per-date samples of a variable-size S&P 500 graph.

    Args:
        paths: Resolved DataPaths.
        membership: Time-stamped S&P 500 membership roster.
        start_date, end_date: Date range for valid prediction dates (inclusive).
        lookback: Length of each feature window in trading days (L).
        horizon: Forward-return horizon in trading days (H).
        feature_engine: Optional callable transforming (L, raw_cols) -> (L, F).
            If None, uses DEFAULT_RAW_COLUMNS as features.
        sector_map: ticker -> sector name. Unknown tickers default to sector 0.
        tickers: Optional override of the ticker universe. Defaults to every
            ticker that ever appeared in the roster and has a parquet on disk.
        stride: Step between prediction dates (default 1 = daily).
        preload: Load all parquets into memory at init (default True). Setting
            False causes each __getitem__ to re-read from disk — slower but
            memory-friendly for 10k+ ticker universes.
    """

    def __init__(
        self,
        paths: DataPaths,
        membership: MembershipRoster,
        start_date: str | date_cls,
        end_date:   str | date_cls,
        lookback: int = 60,
        horizon:  int = 5,
        feature_engine: Optional[FeatureFn] = None,
        sector_map:     Optional[Mapping[str, str]] = None,
        tickers:        Optional[List[str]] = None,
        stride: int = 5,
        preload: bool = True,
        normalize: bool = True,
        features: str = "technical",          # "ohlcv" | "technical"
        purge_end: int = 0,
        epoch_offset: int = 0,
        macro_features: Optional["MacroFeatures"] = None,
    ):
        # stride default is now `horizon` (5) so consecutive training samples
        # have non-overlapping targets. Stride=1 produced ~5× overlapping
        # targets that inflated val IC and caused early-stopping to fire
        # on memorised local autocorrelation rather than real generalisation.
        #
        # `purge_end` removes the trailing `purge_end` prediction dates from
        # the usable range — set it to `horizon` when building the train
        # dataset so its last target window can't leak into the val split.
        #
        # `epoch_offset` shifts the stride start by N trading days so each
        # epoch can see a different non-overlapping subsequence. Trainer
        # rotates this from 0..stride-1 across consecutive train epochs to
        # multiply the unique training signal without breaking the
        # no-target-overlap invariant. Always 0 for val/test datasets.
        if lookback <= 0:
            raise ValueError(f"lookback must be positive, got {lookback}")
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if purge_end < 0:
            raise ValueError(f"purge_end must be >= 0, got {purge_end}")
        if not 0 <= epoch_offset < stride:
            raise ValueError(
                f"epoch_offset must be in [0, stride) = [0, {stride}); got {epoch_offset}"
            )

        self.paths = paths
        self.membership = membership
        self.start_date = pd.Timestamp(_coerce_date(start_date))
        self.end_date   = pd.Timestamp(_coerce_date(end_date))
        if self.start_date > self.end_date:
            raise ValueError(
                f"start_date {self.start_date.date()} > end_date {self.end_date.date()}"
            )

        self.lookback = lookback
        self.horizon  = horizon
        self.stride   = stride
        self.purge_end = purge_end
        self.epoch_offset = int(epoch_offset)
        self.feature_engine = feature_engine
        self.normalize = bool(normalize)
        if features not in {"ohlcv", "technical"}:
            raise ValueError(f"features must be 'ohlcv' or 'technical', got {features!r}")
        self.features = features

        self.tickers = self._resolve_ticker_universe(tickers)
        self.n_max = len(self.tickers)
        self._ticker_to_idx = {t: i for i, t in enumerate(self.tickers)}

        self.sector_labels, self._sector_tensor = self._build_sector_tensor(sector_map)

        # Price frames: ticker -> DataFrame indexed by normalised date.
        self._frames: Dict[str, pd.DataFrame] = {}
        # numpy caches — populated alongside preload. Empty in lazy mode.
        self._np_data:  Dict[str, np.ndarray] = {}   # (T, F) float32 of DEFAULT_RAW_COLUMNS
        self._np_index: Dict[str, np.ndarray] = {}   # (T,)   int64  of date in ns
        self._np_close: Dict[str, np.ndarray] = {}   # (T,)   float32 view of adj_close
        self._preload = preload
        if preload:
            self._load_all_frames()

        # Canonical trading calendar = union of all observed dates.
        self._calendar = self._build_calendar()

        # Prediction dates within [start_date, end_date] that have a complete
        # lookback and horizon *somewhere* in the universe.
        self._prediction_dates = self._build_prediction_dates()

        # Slow/fast split is on whenever the dataset emits its own technical
        # features and no feature_engine is overriding them. The split flag
        # decides whether `__getitem__` emits a `slow_features` tensor and
        # whether `_n_features` reports the FAST count or the full count.
        self._emits_split = (
            self.feature_engine is None and self.features == "technical"
        )

        # Optional macro / market-wide features (VIX, TNX, DXY, SPY 5d-changes).
        # Broadcast to every stock in the cross-section per date — same value
        # across the universe, but the model can learn that they modulate
        # stock-specific patterns. Concatenated to the slow feature vector
        # at __getitem__ time. Silently no-op when the loader is empty.
        self.macro_features = macro_features
        n_macro = (macro_features.n_features
                   if (macro_features is not None and not macro_features.is_empty()
                       and self._emits_split)
                   else 0)
        self._n_macro_features = n_macro

        # Cache feature dimensionality (requires one peek at the data).
        # In split mode this is the FAST subset — the count flowing through
        # the temporal encoder. The full pre-split width comes from
        # `_raw_feature_dim()`. Slow count includes any macro features
        # that get appended at sample time.
        self._n_features = self._infer_feature_dim()
        self._n_slow_features = (
            len(SLOW_FEATURE_COLUMNS) + n_macro if self._emits_split else 0
        )

        log.info(
            "Dataset ready: N_max={}, L={}, F={}{}, samples={}",
            self.n_max, self.lookback, self._n_features,
            f"+slow{self._n_slow_features}" if self._emits_split else "",
            len(self._prediction_dates),
        )

    # ── Public API ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._prediction_dates)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if not 0 <= idx < len(self._prediction_dates):
            raise IndexError(idx)
        pred_date = self._prediction_dates[idx]

        # Always build the full TECHNICAL_FEATURE_COLUMNS layout first (or
        # OHLCV, depending on `features` mode). Splitting into fast/slow
        # happens after cross-sectional normalisation so the slow snapshot
        # is normalised on the same basis as the fast window.
        raw_dim = self._raw_feature_dim()
        features = np.zeros((self.n_max, self.lookback, raw_dim), dtype=np.float32)
        targets  = np.zeros((self.n_max,), dtype=np.float32)
        vol_targets = np.zeros((self.n_max,), dtype=np.float32)
        mask     = np.zeros((self.n_max,), dtype=bool)
        tickers_out: List[str] = [""] * self.n_max

        members = self.membership.tickers_on(pred_date.date())
        for ticker in members:
            if ticker not in self._ticker_to_idx:
                continue
            slot = self._ticker_to_idx[ticker]
            tickers_out[slot] = ticker
            sample = self._build_sample(ticker, pred_date)
            if sample is None:
                continue
            features[slot] = sample.features
            targets[slot]  = sample.target
            vol_targets[slot] = sample.vol_target if np.isfinite(sample.vol_target) else 0.0
            mask[slot]     = True

        # Cross-sectional z-score across all valid stocks for this date.
        # Per-stock per-window normalisation produced 100×+ outliers for any
        # feature with a non-zero mean and small in-window std (log_volume,
        # RSI level, MACD, ATR). Normalising across the ~500-stock cross-
        # section instead has a stable std and preserves cross-sectional
        # rank — exactly what a ranking model needs.
        if self.normalize and mask.sum() >= 5:
            valid_idx = np.where(mask)[0]
            valid = features[valid_idx]                      # (n_valid, L, F)
            cs_mean = valid.mean(axis=0, keepdims=True)      # (1, L, F)
            cs_std  = valid.std(axis=0, keepdims=True)
            cs_std  = np.where(cs_std < 1e-6, 1.0, cs_std)
            features[valid_idx] = ((valid - cs_mean) / cs_std).astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        out: Dict[str, object] = {
            "targets":    torch.from_numpy(targets),
            "volatility": torch.from_numpy(vol_targets),
            "mask":       torch.from_numpy(mask),
            "sectors":    self._sector_tensor.clone(),
            "tickers":    tickers_out,
            "date":       pred_date,
        }

        if self._emits_split:
            # Technical mode: route fast features through the temporal
            # encoder (full window) and slow features as a static last-day
            # snapshot. The temporal encoder no longer wastes capacity on
            # 60 nearly-identical RSI / MACD values per stock.
            fast_view = features[:, :, _FAST_IDX]                  # (N, L, F_fast)
            slow_view = features[:, -1, _SLOW_IDX]                 # (N, F_slow)
            # Append broadcast macro features (same value for every stock
            # in the cross-section on this date). Silent no-op when the
            # loader is empty.
            if self._n_macro_features > 0 and self.macro_features is not None:
                macro_vec = self.macro_features.get_features(pred_date)   # (n_macro,)
                macro_broadcast = np.broadcast_to(
                    macro_vec[None, :], (self.n_max, macro_vec.shape[0]),
                )
                slow_view = np.concatenate([slow_view, macro_broadcast], axis=-1)
            out["features"]      = torch.from_numpy(fast_view.copy())
            out["slow_features"] = torch.from_numpy(slow_view.copy())
        else:
            # OHLCV / feature-engine path keeps the legacy single-tensor contract.
            out["features"] = torch.from_numpy(features)
        return out

    def shapes(self) -> SampleShapes:
        return SampleShapes(
            n_max=self.n_max,
            lookback=self.lookback,
            n_features=self._n_features,
            n_samples=len(self),
            n_slow_features=self._n_slow_features,
        )

    @property
    def n_slow_features(self) -> int:
        return self._n_slow_features

    def _raw_feature_dim(self) -> int:
        """Width of the pre-split feature array. Always TECHNICAL_FEATURE_COLUMNS
        in split mode; whatever the feature engine / OHLCV produces otherwise."""
        if self._emits_split:
            return len(TECHNICAL_FEATURE_COLUMNS)
        return self._n_features

    def prediction_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._prediction_dates)

    # ── Sample construction ────────────────────────────────────────────

    @dataclass
    class _Sample:
        features: np.ndarray
        target:   float
        vol_target: float = float("nan")    # std of next H daily log returns (forward-looking)

    def _build_sample(self, ticker: str, pred_date: pd.Timestamp) -> Optional["_Sample"]:
        # Fast path — numpy-only when no feature engine is supplied and we
        # have cached the ticker's arrays at init. This is the common case
        # for training; feature_engine users hit the pandas path below.
        if self.feature_engine is None and ticker in self._np_index:
            return self._build_sample_fast(ticker, pred_date)
        return self._build_sample_pandas(ticker, pred_date)

    def _build_sample_fast(
        self, ticker: str, pred_date: pd.Timestamp,
    ) -> Optional["_Sample"]:
        """Pure-numpy version: ~20× faster than the pandas path on 500-ticker batches."""
        idx = self._np_index[ticker]            # (T,) int64 ns
        data = self._np_data[ticker]            # (T, F) float32
        close = self._np_close[ticker]          # (T,)  float32

        pred_ns = pd.Timestamp(pred_date).value
        loc = int(np.searchsorted(idx, pred_ns, side="right"))
        if loc < self.lookback:
            return None
        if loc + self.horizon - 1 >= idx.size:
            return None

        window = data[loc - self.lookback : loc]
        px_now    = float(close[loc - 1])
        px_future = float(close[loc + self.horizon - 1])
        if px_now <= 0 or px_future <= 0 or not (np.isfinite(px_now) and np.isfinite(px_future)):
            return None

        # Forward volatility target: std of the H-1 daily log returns INSIDE
        # the target window (loc-1 → loc+H-1). Computed only when the future
        # closes are finite and positive; falls back to NaN otherwise.
        future_closes = close[loc - 1 : loc + self.horizon]
        if (future_closes.size >= 2
                and np.isfinite(future_closes).all()
                and (future_closes > 0).all()):
            daily = np.diff(np.log(future_closes.astype(np.float64)))
            vol_target = float(daily.std(ddof=0)) if daily.size >= 2 else float("nan")
        else:
            vol_target = float("nan")

        # Always return RAW features here. Normalisation is now cross-
        # sectional (across stocks per date), applied in __getitem__ once
        # the full date-level batch is assembled.
        feat = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=True)
        return self._Sample(
            features=feat,
            target=float(np.log(px_future / px_now)),
            vol_target=vol_target,
        )

    def _build_sample_pandas(
        self, ticker: str, pred_date: pd.Timestamp,
    ) -> Optional["_Sample"]:
        """Legacy pandas path — used when a `feature_engine` callable is supplied."""
        frame = self._get_frame(ticker)
        if frame is None or frame.empty:
            return None

        loc = int(frame.index.searchsorted(pred_date, side="right"))
        if loc < self.lookback:
            return None
        if loc + self.horizon - 1 >= len(frame):
            return None

        window = frame.iloc[loc - self.lookback : loc]
        px_now    = float(frame["adj_close"].iloc[loc - 1])
        px_future = float(frame["adj_close"].iloc[loc + self.horizon - 1])
        if px_now <= 0 or px_future <= 0 or not np.isfinite([px_now, px_future]).all():
            return None
        target = float(np.log(px_future / px_now))
        future_closes = frame["adj_close"].iloc[loc - 1 : loc + self.horizon].to_numpy(dtype=np.float64)
        if (future_closes.size >= 2
                and np.isfinite(future_closes).all()
                and (future_closes > 0).all()):
            daily = np.diff(np.log(future_closes))
            vol_target = float(daily.std(ddof=0)) if daily.size >= 2 else float("nan")
        else:
            vol_target = float("nan")

        if self.feature_engine is not None:
            feat = np.asarray(self.feature_engine(window), dtype=np.float32)
        else:
            feat = window[DEFAULT_RAW_COLUMNS].to_numpy(dtype=np.float32)
        if feat.shape[0] != self.lookback:
            return None
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return self._Sample(features=feat, target=target, vol_target=vol_target)

    # ── Preloading / frame access ──────────────────────────────────────

    def _load_all_frames(self) -> None:
        loaded = 0
        for ticker in self.tickers:
            frame = self._read_parquet(ticker)
            if frame is not None:
                self._frames[ticker] = frame
                self._cache_numpy(ticker, frame)
                loaded += 1
        log.info("Preloaded {} / {} ticker frames into memory.", loaded, self.n_max)

    def _cache_numpy(self, ticker: str, frame: pd.DataFrame) -> None:
        """Pre-extract numpy arrays per ticker so `_build_sample` is pandas-free.

        Materialising once at init is the single biggest speedup for the
        training loop — pandas `.loc[...]` + column selection + `.to_numpy()`
        on every `__getitem__` call dominated CPU time.

        When `features == "technical"`, we also precompute the ~15 technical
        indicators here so training sees a richer signal than raw OHLCV.
        The adj_close series is always kept separately for target computation.
        """
        missing = [c for c in DEFAULT_RAW_COLUMNS if c not in frame.columns]
        if missing:
            return  # skip; _build_sample falls back to the pandas path

        ohlcv = frame[DEFAULT_RAW_COLUMNS].to_numpy(dtype=np.float32)
        self._np_index[ticker] = frame.index.values.astype("datetime64[ns]").astype(np.int64)
        # `_np_close` is always the raw adj_close — used for target returns
        self._np_close[ticker] = ohlcv[:, DEFAULT_RAW_COLUMNS.index("adj_close")].copy()

        if self.features == "technical":
            self._np_data[ticker] = _compute_technical_features(ohlcv)
        else:
            self._np_data[ticker] = ohlcv

    def _get_frame(self, ticker: str) -> Optional[pd.DataFrame]:
        if self._preload:
            return self._frames.get(ticker)
        if ticker not in self._frames:
            self._frames[ticker] = self._read_parquet(ticker)
        return self._frames[ticker]

    def _read_parquet(self, ticker: str) -> Optional[pd.DataFrame]:
        path = self.paths.price_file(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.drop_duplicates(subset=["date"]).sort_values("date")
        return df.set_index("date")

    # ── Init helpers ───────────────────────────────────────────────────

    def _resolve_ticker_universe(self, override: Optional[List[str]]) -> List[str]:
        if override is not None:
            return sorted({t.upper().strip() for t in override if t})
        all_ever = self.membership.all_tickers_ever()
        on_disk = {
            t for t in all_ever
            if self.paths.price_file(t).exists()
        }
        return sorted(on_disk) if on_disk else sorted(all_ever)

    def _build_sector_tensor(
        self,
        sector_map: Optional[Mapping[str, str]],
    ) -> Tuple[List[str], torch.Tensor]:
        if not sector_map:
            return [], torch.zeros(self.n_max, dtype=torch.long)

        unique = sorted({s for s in sector_map.values() if s})
        label_to_idx = {lbl: i + 1 for i, lbl in enumerate(unique)}  # 0 reserved for "unknown"
        tensor = torch.zeros(self.n_max, dtype=torch.long)
        for ticker, sector in sector_map.items():
            slot = self._ticker_to_idx.get(ticker.upper())
            if slot is not None and sector in label_to_idx:
                tensor[slot] = label_to_idx[sector]
        return ["unknown"] + unique, tensor

    def _build_calendar(self) -> pd.DatetimeIndex:
        frames_iter = self._frames.values() if self._preload else ()
        dates: set = set()
        for f in frames_iter:
            dates.update(f.index.tolist())
        if not dates:
            # Non-preload path: use a business-day range as a permissive default.
            return pd.date_range(self.start_date - pd.Timedelta(days=365),
                                 self.end_date, freq="B")
        return pd.DatetimeIndex(sorted(dates))

    def _build_prediction_dates(self) -> List[pd.Timestamp]:
        cal = self._calendar
        in_range = cal[(cal >= self.start_date) & (cal <= self.end_date)]
        # Need at least `lookback` prior dates in the calendar and `horizon`
        # future dates for a sample to be constructible for *any* stock.
        earliest_allowed = cal[self.lookback - 1] if len(cal) >= self.lookback else None
        latest_allowed = cal[-self.horizon] if len(cal) >= self.horizon else None

        if earliest_allowed is None or latest_allowed is None:
            return []
        valid = in_range[(in_range >= earliest_allowed) & (in_range <= latest_allowed)]
        # epoch_offset shifts which subsequence we take. With stride=5 and
        # offset rotating 0..4 across epochs the union covers every valid
        # date once — same as stride=1 — but each epoch keeps non-overlapping
        # targets so the val IC remains honest.
        strided = list(valid[self.epoch_offset :: self.stride])
        # Purge trailing dates so the last target window cannot span into
        # the next split (train→val or val→test leakage guard).
        if self.purge_end > 0 and len(strided) > self.purge_end:
            strided = strided[: -self.purge_end]
        return strided

    # ── Epoch-rotation support ─────────────────────────────────────────

    def set_epoch_offset(self, offset: int) -> None:
        """Reseat the stride starting point. Used by the trainer to rotate
        through `stride` non-overlapping subsequences across epochs."""
        offset = int(offset) % self.stride
        if offset == self.epoch_offset:
            return
        self.epoch_offset = offset
        self._prediction_dates = self._build_prediction_dates()

    def _infer_feature_dim(self) -> int:
        if self.feature_engine is None:
            if self.features == "technical":
                # In split mode the temporal encoder only sees the FAST cols.
                return len(FAST_FEATURE_COLUMNS)
            return len(DEFAULT_RAW_COLUMNS)
        # Peek: find the first ticker/date combination that yields a window.
        for ticker in self.tickers:
            frame = self._get_frame(ticker)
            if frame is None or len(frame) < self.lookback:
                continue
            window = frame.iloc[: self.lookback]
            try:
                feat = np.asarray(self.feature_engine(window), dtype=np.float32)
            except Exception as exc:
                raise RuntimeError(
                    f"feature_engine failed on probe for {ticker}: {exc}"
                ) from exc
            if feat.ndim != 2 or feat.shape[0] != self.lookback:
                raise ValueError(
                    f"feature_engine must return shape (L, F); got {feat.shape}"
                )
            return int(feat.shape[1])
        # No frames available — we cannot probe. Assume default.
        return len(DEFAULT_RAW_COLUMNS)


def _compute_technical_features(ohlcv: np.ndarray) -> np.ndarray:
    """Compute ~15 technical indicators from raw OHLCV, pure numpy.

    Input:  (T, 6) array of [open, high, low, close, adj_close, volume]
    Output: (T, len(TECHNICAL_FEATURE_COLUMNS)) float32

    Used to pre-compute per-ticker features ONCE at Dataset init so the
    training hot path stays pandas-free. Each indicator is implemented in
    numpy with no Python loops — total init cost is ~30 ms per ticker.
    """
    opens     = ohlcv[:, 0]
    highs     = ohlcv[:, 1]
    lows      = ohlcv[:, 2]
    close     = ohlcv[:, 3]
    adj_close = np.maximum(ohlcv[:, 4], 1e-8)
    volume    = np.maximum(ohlcv[:, 5], 0.0)

    # ── Log returns over 5 / 20 trading days ────────────────────────────
    # 1d return dropped from the feature set — too noisy for 5-day horizon.
    # We still compute it internally as the basis for vol_5d / vol_20d.
    log_px = np.log(adj_close)
    _ret_1d_internal = _lag_diff(log_px, 1)            # used only for vol calc
    ret_5d  = _lag_diff(log_px, 5)
    ret_20d = _lag_diff(log_px, 20)

    # ── Realised volatility (rolling std of 1d log returns) ─────────────
    vol_5d  = _rolling_std(_ret_1d_internal, 5)
    vol_20d = _rolling_std(_ret_1d_internal, 20)

    # ── RSI-14 (Wilder's smoothing) ─────────────────────────────────────
    rsi_14 = _rsi_wilder(adj_close, period=14)

    # ── MACD (12 / 26 / 9) ──────────────────────────────────────────────
    ema_12 = _ema(adj_close, span=12)
    ema_26 = _ema(adj_close, span=26)
    macd = ema_12 - ema_26
    macd_signal = _ema(macd, span=9)
    macd_hist = macd - macd_signal

    # ── Bollinger band width (20d, 2σ), normalised by price ─────────────
    bb_mean = _rolling_mean(adj_close, 20)
    bb_std  = _rolling_std(adj_close, 20)
    bbw_20 = (4.0 * bb_std) / np.maximum(bb_mean, 1e-8)

    # ── ATR-14 (EMA of true range) ──────────────────────────────────────
    atr_14 = _atr(highs, lows, close, period=14)

    # ── Volume features ─────────────────────────────────────────────────
    log_volume = np.log1p(volume)
    vol_mean_20 = _rolling_mean(volume, 20)
    rel_volume_20 = volume / np.maximum(vol_mean_20, 1e-8)

    # ── Intraday range + overnight gap ──────────────────────────────────
    intraday_range = (highs - lows) / np.maximum(close, 1e-8)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    gap = (opens - prev_close) / np.maximum(prev_close, 1e-8)

    feats = np.stack([
        ret_5d, ret_20d,
        vol_5d, vol_20d,
        rsi_14,
        macd, macd_signal, macd_hist,
        bbw_20,
        atr_14,
        log_volume, rel_volume_20,
        intraday_range, gap,
    ], axis=1).astype(np.float32)

    # Warm-up rows at the start have NaN / inf due to lag/rolling ops.
    # Callers downstream reject samples whose window starts before
    # enough history exists, so leaving NaN here is fine — they get
    # nan_to_num'd at sample build time.
    return feats


def _lag_diff(x: np.ndarray, k: int) -> np.ndarray:
    """x[i] - x[i-k]. First k entries are NaN."""
    out = np.full_like(x, np.nan)
    if k < x.size:
        out[k:] = x[k:] - x[:-k]
    return out


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Simple rolling mean. First w-1 entries are NaN."""
    n = x.size
    if n < w:
        return np.full(n, np.nan, dtype=np.float32)
    cum = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    sums = cum[w:] - cum[:-w]
    out = np.full(n, np.nan, dtype=np.float32)
    out[w - 1:] = (sums / w).astype(np.float32)
    return out


def _rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    """Population rolling std via (E[x²] - E[x]²)½. Matches pandas with ddof=0."""
    n = x.size
    if n < w:
        return np.full(n, np.nan, dtype=np.float32)
    clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    cum1 = np.concatenate([[0.0], np.cumsum(clean)])
    cum2 = np.concatenate([[0.0], np.cumsum(clean * clean)])
    mean = (cum1[w:] - cum1[:-w]) / w
    meansq = (cum2[w:] - cum2[:-w]) / w
    var = np.maximum(meansq - mean * mean, 0.0)
    out = np.full(n, np.nan, dtype=np.float32)
    out[w - 1:] = np.sqrt(var).astype(np.float32)
    return out


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with pandas-compatible `span` parameterisation."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out.astype(np.float32)


def _rsi_wilder(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI with Wilder's exponential smoothing (alpha = 1 / period)."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    alpha = 1.0 / period

    avg_gain = np.empty_like(close, dtype=np.float64)
    avg_loss = np.empty_like(close, dtype=np.float64)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, close.size):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i - 1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i - 1]
    rs = avg_gain / np.maximum(avg_loss, 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi.astype(np.float32)


def _atr(highs: np.ndarray, lows: np.ndarray, close: np.ndarray,
          period: int = 14) -> np.ndarray:
    """Average True Range (EMA of true range, Wilder smoothing)."""
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([
        highs - lows,
        np.abs(highs - prev_close),
        np.abs(lows  - prev_close),
    ])
    alpha = 1.0 / period
    out = np.empty_like(tr, dtype=np.float64)
    out[0] = tr[0]
    for i in range(1, tr.size):
        out[i] = alpha * tr[i] + (1 - alpha) * out[i - 1]
    out[:period] = np.nan
    return out.astype(np.float32)


def collate_graph_samples(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack variable-size graph samples into a batch.

    Samples in a batch share the same N_max (enforced by the Dataset padding
    contract), so we just stack along a new batch dimension. `tickers` stays
    as list-of-lists and `date` becomes a list of Timestamps.
    """
    out: Dict[str, object] = {
        "features": torch.stack([b["features"] for b in batch]),    # (B, N, L, F)
        "targets":  torch.stack([b["targets"]  for b in batch]),    # (B, N)
        "mask":     torch.stack([b["mask"]     for b in batch]),    # (B, N)
        "sectors":  torch.stack([b["sectors"]  for b in batch]),    # (B, N)
        "tickers":  [b["tickers"] for b in batch],
        "date":     [b["date"]    for b in batch],
    }
    return out
