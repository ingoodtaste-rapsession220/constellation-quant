"""Phase 1 tests: data pipeline using synthetic data + mocks. No network I/O."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from constellation_quant.data import (
    CacheManager,
    DataCleaner,
    DataPaths,
    DownloadReport,
    FundamentalsDownloader,
    MembershipRoster,
    PriceDownloader,
    SentimentDownloader,
    StockTwitsSource,
    apply_ticker_aliases,
    detect_revert_spikes,
    drop_duplicates,
    drop_nan_prices,
    forward_fill_small_gaps,
    remove_revert_spikes,
    validate_roster,
)
from constellation_quant.data.membership import KNOWN_EVENTS, parse_fja05680_csv


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_paths(tmp_path: Path) -> DataPaths:
    """A DataPaths instance rooted in a tmp dir (no env vars)."""
    paths_cfg = {
        "data_dir":        str(tmp_path / "data"),
        "processed_data":  str(tmp_path / "data/processed"),
        "cache_dir":       str(tmp_path / "data/cache"),
        "graphs_dir":      str(tmp_path / "data/graphs"),
        "membership_file": str(tmp_path / "data/membership_roster.json"),
        "checkpoint_dir":  str(tmp_path / "ckpt"),
        "outputs_dir":     str(tmp_path / "outputs"),
    }
    paths = DataPaths.from_config(paths_cfg)
    paths.ensure_dirs()
    return paths


@pytest.fixture
def synthetic_price_frame() -> pd.DataFrame:
    """250 trading days of fake OHLCV with a deterministic random walk."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=250)
    returns = rng.normal(0.0005, 0.015, size=len(dates))
    close = 100 * np.exp(returns.cumsum())
    return pd.DataFrame({
        "date":         dates,
        "open":         close * (1 + rng.normal(0, 0.005, size=len(dates))),
        "high":         close * (1 + rng.uniform(0, 0.01, size=len(dates))),
        "low":          close * (1 - rng.uniform(0, 0.01, size=len(dates))),
        "close":        close,
        "adj_close":    close,
        "volume":       rng.integers(1_000_000, 10_000_000, size=len(dates)),
        "dividends":    0.0,
        "stock_splits": 0.0,
    })


@pytest.fixture
def tiny_roster() -> MembershipRoster:
    """A 3-snapshot roster with ~500 synthetic tickers (post-2001 for validator)."""
    base = {f"T{i:03d}" for i in range(500)}
    snapshots = {
        date(2010, 1, 4):  base,
        date(2013, 12, 23): base | {"META"},                  # add META
        date(2020, 12, 21): (base | {"META"}) | {"TSLA"},     # add TSLA
    }
    return MembershipRoster.from_daily_snapshots(snapshots)


# ── DataPaths ──────────────────────────────────────────────────────────────


def test_data_paths_ensure_dirs_creates_tree(tmp_paths: DataPaths):
    assert tmp_paths.raw_prices.exists()
    assert tmp_paths.cache_dir.exists()
    assert tmp_paths.graphs_dir.exists()


def test_data_paths_rejects_unresolved_env_var(tmp_path: Path):
    bad_cfg = {
        "data_dir":        "${UNSET_VAR}/data",
        "processed_data":  "${UNSET_VAR}/proc",
        "cache_dir":       "${UNSET_VAR}/cache",
        "graphs_dir":      "${UNSET_VAR}/g",
        "membership_file": "${UNSET_VAR}/m.json",
        "checkpoint_dir":  "${UNSET_VAR}/ckpt",
        "outputs_dir":     "${UNSET_VAR}/out",
    }
    paths = DataPaths.from_config(bad_cfg)
    with pytest.raises(ValueError, match="unresolved env var"):
        paths.ensure_dirs()


def test_price_file_path_uppercased(tmp_paths: DataPaths):
    assert tmp_paths.price_file("aapl").name == "AAPL.parquet"


# ── Membership ─────────────────────────────────────────────────────────────


def test_roster_tickers_on_uses_latest_snapshot_at_or_before(tiny_roster):
    assert "META" in tiny_roster.tickers_on(date(2015, 1, 1))  # between snapshots
    assert "TSLA" not in tiny_roster.tickers_on(date(2015, 1, 1))
    assert "TSLA" in tiny_roster.tickers_on(date(2021, 1, 1))


def test_roster_raises_before_earliest(tiny_roster):
    with pytest.raises(KeyError):
        tiny_roster.tickers_on(date(2000, 1, 1))


def test_roster_additions_removals(tiny_roster):
    assert tiny_roster.additions(date(2013, 12, 23)) == frozenset({"META"})
    assert tiny_roster.removals(date(2013, 12, 23)) == frozenset()


def test_roster_all_tickers_ever(tiny_roster):
    all_ts = tiny_roster.all_tickers_ever()
    assert "META" in all_ts and "TSLA" in all_ts
    assert len(all_ts) == 502  # 500 base + META + TSLA


def test_roster_json_roundtrip(tiny_roster, tmp_path):
    p = tmp_path / "roster.json"
    tiny_roster.save_json(p)
    loaded = MembershipRoster.load_json(p)
    assert loaded.snapshot_dates() == tiny_roster.snapshot_dates()
    for d in tiny_roster.snapshot_dates():
        assert loaded.tickers_on(d) == tiny_roster.tickers_on(d)


def test_validate_roster_passes_on_known_events(tiny_roster):
    errors = validate_roster(tiny_roster)
    # tiny_roster contains META on 2013-12-23 and TSLA on 2020-12-21 → no errors
    # about those. GOOG is NOT in the tiny_roster, so expect one error for GOOG.
    goog_errors = [e for e in errors if "GOOG" in e]
    assert goog_errors, "validator should flag missing GOOG"
    count_errors = [e for e in errors if "Implausible" in e]
    assert count_errors == [], f"unexpected count errors: {count_errors}"


def test_validate_roster_flags_tsla_missing():
    base = {f"T{i:03d}" for i in range(500)}
    snapshots = {date(2020, 12, 21): base}  # no TSLA on addition date → fail
    bad = MembershipRoster.from_daily_snapshots(snapshots)
    errors = validate_roster(bad)
    assert any("TSLA" in e for e in errors)


def test_parse_fja05680_csv():
    csv_text = (
        "date,tickers\n"
        "2020-12-21,\"AAPL,MSFT,TSLA\"\n"
        "2020-12-22,\"AAPL,MSFT,TSLA,NVDA\"\n"
    )
    snapshots = parse_fja05680_csv(csv_text)
    assert len(snapshots) == 2
    assert snapshots[date(2020, 12, 21)] == frozenset({"AAPL", "MSFT", "TSLA"})
    assert "NVDA" in snapshots[date(2020, 12, 22)]


# ── Price downloader ───────────────────────────────────────────────────────


def test_price_downloader_resume_skips_existing(tmp_paths, synthetic_price_frame):
    # Pre-populate AAPL parquet to simulate an existing download.
    (tmp_paths.raw_prices / "AAPL.parquet").parent.mkdir(parents=True, exist_ok=True)
    synthetic_price_frame.to_parquet(tmp_paths.price_file("AAPL"), index=False)

    dl = PriceDownloader(tmp_paths, start="2020-01-01")
    with patch.object(PriceDownloader, "_fetch_one") as mock_fetch:
        report = dl.download_all(["AAPL"], resume=True)
    mock_fetch.assert_not_called()
    assert report.skipped == ["AAPL"]
    assert report.succeeded == []


def test_price_downloader_reports_failures(tmp_paths):
    dl = PriceDownloader(tmp_paths, start="2020-01-01", max_retries=1, backoff_base=0.0)
    with patch.object(PriceDownloader, "_fetch_one", side_effect=RuntimeError("boom")):
        report = dl.download_all(["AAPL"], resume=False)
    assert report.succeeded == []
    assert "AAPL" in report.failed
    assert "RuntimeError" in report.failed["AAPL"]


def test_price_downloader_normalises_columns(synthetic_price_frame):
    yf_style = synthetic_price_frame.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "adj_close": "Adj Close", "volume": "Volume",
        "dividends": "Dividends", "stock_splits": "Stock Splits",
    }).set_index("date")
    yf_style.index.name = "Date"
    out = PriceDownloader._normalise(yf_style)
    assert list(out.columns[:7]) == ["date", "open", "high", "low", "close", "adj_close", "volume"]
    assert len(out) == len(synthetic_price_frame)


def test_price_downloader_empty_frame_raises(tmp_paths):
    dl = PriceDownloader(tmp_paths, start="2020-01-01", max_retries=1, backoff_base=0.0)
    with patch.object(PriceDownloader, "_fetch_one", return_value=pd.DataFrame()):
        report = dl.download_all(["AAPL"], resume=False)
    assert "AAPL" in report.failed
    assert "no rows" in report.failed["AAPL"]


def test_download_report_summary():
    r = DownloadReport(succeeded=["A"], skipped=["B"], failed={"C": "err"})
    assert "downloaded=1" in r.summary()
    assert r.total == 3


# ── Fundamentals ───────────────────────────────────────────────────────────


def test_fundamentals_to_long_melts_correctly(tmp_paths):
    dl = FundamentalsDownloader(tmp_paths)
    wide = pd.DataFrame(
        {
            pd.Timestamp("2023-03-31"): [100.0, 20.0],
            pd.Timestamp("2023-06-30"): [110.0, 22.0],
        },
        index=["Total Revenue", "Net Income"],
    )
    long = dl._to_long({"income": wide})
    assert set(long["metric"].unique()) >= {"total_revenue", "net_income"}
    assert len(long) == 4  # 2 metrics × 2 quarters


def test_fundamentals_match_label_case_insensitive(tmp_paths):
    dl = FundamentalsDownloader(tmp_paths)
    lbl = dl._match_label(["TOTAL REVENUE", "Net Income"], ["total revenue"])
    assert lbl == "TOTAL REVENUE"
    assert dl._match_label(["Other"], ["missing"]) is None


# ── Sentiment ──────────────────────────────────────────────────────────────


def test_sentiment_downloader_writes_empty_when_no_sources(tmp_paths):
    """If all sources return empty, that ticker goes into `failed`."""
    class _EmptySource(StockTwitsSource):
        def fetch(self, ticker):
            return self._empty_frame()
    dl = SentimentDownloader(paths=tmp_paths, sources=[_EmptySource()], sleep_between=0.0)
    report = dl.download_all(["AAPL"], resume=False)
    assert "AAPL" in report.failed


def test_sentiment_downloader_concatenates_sources(tmp_paths):
    class _FakeSource(StockTwitsSource):
        name = "fake"
        def fetch(self, ticker):
            return pd.DataFrame({
                "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "source": ["fake", "fake"],
                "score":  [0.5, -0.2],
                "volume": [10, 8],
            })
    dl = SentimentDownloader(paths=tmp_paths, sources=[_FakeSource()], sleep_between=0.0)
    report = dl.download_all(["AAPL"], resume=False)
    assert report.succeeded == ["AAPL"]
    out = pd.read_parquet(tmp_paths.sentiment_file("AAPL"))
    assert len(out) == 2
    assert out["score"].tolist() == [0.5, -0.2]


# ── Cleaner ────────────────────────────────────────────────────────────────


def test_drop_duplicates(synthetic_price_frame):
    dup = pd.concat([synthetic_price_frame, synthetic_price_frame.iloc[[0]]], ignore_index=True)
    cleaned, n = drop_duplicates(dup)
    assert n == 1
    assert len(cleaned) == len(synthetic_price_frame)


def test_drop_nan_prices(synthetic_price_frame):
    df = synthetic_price_frame.copy()
    df.loc[5, "close"] = np.nan
    cleaned, n = drop_nan_prices(df)
    assert n == 1


def test_detect_revert_spikes_flags_synthetic_spike():
    # 10 days flat; day 5 doubles; day 6 reverts
    dates = pd.bdate_range("2020-01-02", periods=10)
    prices = np.concatenate([np.full(5, 100.0), [210.0, 100.0], np.full(3, 100.0)])
    df = pd.DataFrame({"date": dates, "adj_close": prices})
    mask = detect_revert_spikes(df, threshold=0.5)
    assert bool(mask.iloc[5])  # the spike day
    assert not bool(mask.iloc[0])
    assert not bool(mask.iloc[3])


def test_remove_revert_spikes_drops_them():
    dates = pd.bdate_range("2020-01-02", periods=10)
    prices = np.concatenate([np.full(5, 100.0), [210.0, 100.0], np.full(3, 100.0)])
    df = pd.DataFrame({"date": dates, "adj_close": prices})
    cleaned, n = remove_revert_spikes(df, threshold=0.5)
    assert n == 1
    assert len(cleaned) == 9


def test_forward_fill_small_gaps_fills_one_day(synthetic_price_frame):
    # Remove day index 100 — expect it to be ffilled back.
    df = synthetic_price_frame.drop(index=100).reset_index(drop=True)
    filled, added = forward_fill_small_gaps(df, max_gap_days=2)
    assert added == 1
    assert len(filled) == len(synthetic_price_frame)


def test_forward_fill_skips_large_gaps(synthetic_price_frame):
    df = synthetic_price_frame.drop(index=range(100, 110)).reset_index(drop=True)
    filled, added = forward_fill_small_gaps(df, max_gap_days=2)
    assert added == 0


def test_apply_ticker_aliases():
    assert apply_ticker_aliases("FB", {"FB": "META"}) == "META"
    assert apply_ticker_aliases("AAPL", {"FB": "META"}) == "AAPL"
    assert apply_ticker_aliases("AAPL", None) == "AAPL"


def test_data_cleaner_batch_reports(synthetic_price_frame):
    cleaner = DataCleaner({"max_single_day_move": 0.5, "forward_fill_max_gap_days": 2})
    cleaned, reports = cleaner.clean_batch({"AAPL": synthetic_price_frame})
    assert "AAPL" in cleaned
    assert reports["AAPL"].rows_in == len(synthetic_price_frame)


# ── Cache ──────────────────────────────────────────────────────────────────


def test_cache_key_is_deterministic(tmp_path):
    cache = CacheManager(tmp_path)
    k1 = cache.compute_key({"a": 1, "b": 2})
    k2 = cache.compute_key({"b": 2, "a": 1})  # same dict, different order
    assert k1 == k2


def test_cache_key_changes_on_source_mtime(tmp_path):
    cache = CacheManager(tmp_path)
    src = tmp_path / "src.txt"
    src.write_text("v1")
    k1 = cache.compute_key({"x": 1}, source_files=[src])
    src.write_text("v2")
    # Force a different mtime — some filesystems round to 1s resolution.
    import os, time
    os.utime(src, (time.time() + 5, time.time() + 5))
    k2 = cache.compute_key({"x": 1}, source_files=[src])
    assert k1 != k2


def test_cache_put_get_pickle_roundtrip(tmp_path):
    cache = CacheManager(tmp_path)
    key = cache.compute_key({"kind": "arbitrary"})
    cache.put(key, {"hello": [1, 2, 3]})
    assert cache.get(key) == {"hello": [1, 2, 3]}


def test_cache_put_get_dataframe_roundtrip(tmp_path, synthetic_price_frame):
    cache = CacheManager(tmp_path)
    key = cache.compute_key({"kind": "df"})
    cache.put(key, synthetic_price_frame)
    out = cache.get(key)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(synthetic_price_frame)


def test_cache_invalidate_and_clear(tmp_path):
    cache = CacheManager(tmp_path)
    for i in range(3):
        key = cache.compute_key({"i": i})
        cache.put(key, {"x": i})
    assert cache.clear() == 3
    assert cache.compute_key({"i": 0}) not in [p.stem for p in tmp_path.glob("*.pkl")]


def test_cache_returns_none_on_miss(tmp_path):
    cache = CacheManager(tmp_path)
    assert cache.get("nonexistent_key_abcdef") is None


# ── Dataset ────────────────────────────────────────────────────────────────


def test_dataset_end_to_end(tmp_paths, synthetic_price_frame, tiny_roster):
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset, collate_graph_samples
    import torch  # noqa: F401

    # Restrict roster to 3 tickers in the relevant period and write parquets.
    members = {"T000", "T001", "T002"}
    snapshots = {date(2020, 1, 2): members, date(2020, 6, 1): members}
    roster = MembershipRoster.from_daily_snapshots(snapshots)
    for t in members:
        synthetic_price_frame.to_parquet(tmp_paths.price_file(t), index=False)

    ds = DynaGraphDataset(
        paths=tmp_paths,
        membership=roster,
        start_date="2020-05-01",
        end_date="2020-10-01",
        lookback=10,
        horizon=5,
        features="ohlcv",              # test raw-path shape expectations
    )
    assert len(ds) > 0
    sample = ds[0]

    assert sample["features"].shape == (3, 10, 6)   # N=3, L=10, F=6 raw cols
    assert sample["targets"].shape == (3,)
    assert sample["mask"].shape == (3,)
    assert sample["mask"].sum().item() == 3         # all three valid
    assert len(sample["tickers"]) == 3


def test_dataset_no_leakage(tmp_paths, synthetic_price_frame):
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset

    members = {"T000"}
    snapshots = {date(2020, 1, 2): members}
    roster = MembershipRoster.from_daily_snapshots(snapshots)
    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)

    ds = DynaGraphDataset(
        paths=tmp_paths,
        membership=roster,
        start_date="2020-05-01",
        end_date="2020-10-01",
        lookback=20,
        horizon=5,
        normalize=False,       # inspect raw values to verify no leakage
        features="ohlcv",      # raw path so column 4 = adj_close
    )
    sample = ds[0]
    pred_date = sample["date"]
    features = sample["features"][0]  # (L, F)

    # Compare last feature row (adj_close) against the frame at pred_date.
    expected_close = synthetic_price_frame.set_index("date").loc[pred_date, "adj_close"]
    assert abs(features[-1, 4].item() - expected_close) < 1e-3


def test_dataset_technical_features_shape(tmp_paths, synthetic_price_frame):
    """Technical-features path emits split fast/slow tensors + vol target."""
    pytest.importorskip("torch")
    import torch
    from constellation_quant.data.dataset import (
        DynaGraphDataset,
        FAST_FEATURE_COLUMNS,
        SLOW_FEATURE_COLUMNS,
    )

    members = {"T000"}
    snapshots = {date(2020, 1, 2): members}
    roster = MembershipRoster.from_daily_snapshots(snapshots)
    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)

    ds = DynaGraphDataset(
        paths=tmp_paths,
        membership=roster,
        start_date="2020-06-01",                   # past the 20d warm-up
        end_date="2020-10-01",
        lookback=30,
        horizon=5,
        features="technical",
    )
    sample = ds[0]
    # Fast features go through the temporal encoder (full window).
    assert sample["features"].shape == (1, 30, len(FAST_FEATURE_COLUMNS))
    # Slow features collapse to the last-day snapshot per stock.
    assert sample["slow_features"].shape == (1, len(SLOW_FEATURE_COLUMNS))
    # Volatility target alongside the ranking/return target.
    assert sample["volatility"].shape == (1,)
    # After z-score + nan_to_num, nothing NaN in the tensors passed to the model.
    assert torch.isfinite(sample["features"]).all()
    assert torch.isfinite(sample["slow_features"]).all()
    assert torch.isfinite(sample["volatility"]).all()
    # Shapes report should advertise both dims.
    shapes = ds.shapes()
    assert shapes.n_features == len(FAST_FEATURE_COLUMNS)
    assert shapes.n_slow_features == len(SLOW_FEATURE_COLUMNS)


def test_dataset_ohlcv_path_keeps_legacy_contract(tmp_paths, synthetic_price_frame):
    """OHLCV / feature-engine paths must NOT emit slow_features (no split)."""
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset

    members = {"T000"}
    snapshots = {date(2020, 1, 2): members}
    roster = MembershipRoster.from_daily_snapshots(snapshots)
    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)

    ds = DynaGraphDataset(
        paths=tmp_paths, membership=roster,
        start_date="2020-06-01", end_date="2020-10-01",
        lookback=30, horizon=5, features="ohlcv",
    )
    sample = ds[0]
    assert sample["features"].shape == (1, 30, 6)         # legacy raw OHLCV
    assert "slow_features" not in sample
    assert ds.shapes().n_slow_features == 0


def test_dataset_mask_excludes_non_members(tmp_paths, synthetic_price_frame):
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset

    # Two tickers with data, only one in the membership
    for t in ("T000", "T001"):
        synthetic_price_frame.to_parquet(tmp_paths.price_file(t), index=False)
    roster = MembershipRoster.from_daily_snapshots({date(2020, 1, 2): {"T000"}})

    ds = DynaGraphDataset(
        paths=tmp_paths,
        membership=roster,
        start_date="2020-05-01",
        end_date="2020-10-01",
        lookback=10,
        horizon=5,
        tickers=["T000", "T001"],  # force both into the universe
    )
    sample = ds[0]
    t_idx = {t: i for i, t in enumerate(ds.tickers)}
    assert bool(sample["mask"][t_idx["T000"]])
    assert not bool(sample["mask"][t_idx["T001"]])


def test_dataset_chronological(tmp_paths, synthetic_price_frame):
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset

    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)
    roster = MembershipRoster.from_daily_snapshots({date(2020, 1, 2): {"T000"}})

    ds = DynaGraphDataset(
        paths=tmp_paths,
        membership=roster,
        start_date="2020-05-01",
        end_date="2020-10-01",
        lookback=10,
        horizon=5,
    )
    dates = [ds[i]["date"] for i in range(len(ds))]
    assert dates == sorted(dates)


def test_dataset_epoch_offset_rotation_covers_all_dates_without_overlap(
    tmp_paths, synthetic_price_frame,
):
    """5 epochs × stride=5 should cover every valid date exactly once across
    epochs while keeping consecutive samples within an epoch ≥ horizon apart."""
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset

    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)
    roster = MembershipRoster.from_daily_snapshots({date(2020, 1, 2): {"T000"}})

    ds = DynaGraphDataset(
        paths=tmp_paths,
        membership=roster,
        start_date="2020-05-01",
        end_date="2020-10-01",
        lookback=10,
        horizon=5,
        stride=5,
        purge_end=0,
    )

    horizon = ds.horizon
    seen_per_offset: Dict[int, list] = {}
    for offset in range(ds.stride):
        ds.set_epoch_offset(offset)
        dates = [ds[i]["date"] for i in range(len(ds))]
        seen_per_offset[offset] = dates
        # Within an epoch, consecutive prediction dates must be at least
        # `horizon` calendar steps apart in the trading calendar — the
        # invariant the stride=horizon design protects.
        for a, b in zip(dates, dates[1:]):
            i_a = list(ds._calendar).index(a)            # noqa: SLF001
            i_b = list(ds._calendar).index(b)            # noqa: SLF001
            assert i_b - i_a >= horizon, (
                f"target windows overlap within epoch (offset={offset}): "
                f"{a.date()} → {b.date()} are only {i_b - i_a} bars apart"
            )

    # Across 5 offsets the union of predicted dates equals every stride=1
    # date in the same range — so we lose zero information across epochs.
    union = sorted({d for ds_dates in seen_per_offset.values() for d in ds_dates})
    ds_dense = DynaGraphDataset(
        paths=tmp_paths, membership=roster,
        start_date="2020-05-01", end_date="2020-10-01",
        lookback=10, horizon=5, stride=1, purge_end=0,
    )
    dense = [ds_dense[i]["date"] for i in range(len(ds_dense))]
    assert union == sorted(dense), "epoch-rotation should cover every dense-stride date once"


def test_macro_features_empty_returns_zeros():
    """When no parquets exist, get_features returns the zero vector."""
    from constellation_quant.data.macro import MacroFeatures, MACRO_FEATURE_COLUMNS
    m = MacroFeatures.empty()
    assert m.is_empty() is True
    assert m.n_features == len(MACRO_FEATURE_COLUMNS)
    out = m.get_features(pd.Timestamp("2020-01-15"))
    assert out.shape == (m.n_features,)
    assert (out == 0).all()


def test_macro_features_loads_and_computes(tmp_paths):
    """Round-trip: write fake macro parquets, load, query 5-day change."""
    from constellation_quant.data.macro import (
        MacroFeatures, macro_dir, macro_file, MACRO_FEATURE_COLUMNS,
    )
    macro_dir(tmp_paths).mkdir(parents=True, exist_ok=True)

    # Build a known 30-day series for VIX with known 5-day log change.
    dates = pd.bdate_range("2020-01-01", periods=30)
    vix_close = np.linspace(15.0, 30.0, 30, dtype=float)        # rising VIX
    pd.DataFrame({"date": dates, "close": vix_close}).to_parquet(
        macro_file(tmp_paths, "vix"), index=False,
    )

    m = MacroFeatures.from_paths(tmp_paths)
    assert m.is_empty() is False
    assert "vix" in m.series

    # Query a date where 5-day lookback exists.
    feat = m.get_features(dates[10])
    # vix_change_5d is the first feature.
    expected = np.log(vix_close[10] / vix_close[10 - 5])
    assert abs(feat[0] - expected) < 1e-5
    # Other features still zero (no parquet written for them).
    for i, name in enumerate(MACRO_FEATURE_COLUMNS):
        if name != "vix_change_5d":
            assert feat[i] == 0.0


def test_macro_features_handles_tz_aware_parquet(tmp_paths):
    """Regression guard: yfinance writes tz-aware dates; our loader must
    strip the timezone so `s.loc[:tz_naive_ts]` doesn't throw TypeError."""
    from constellation_quant.data.macro import MacroFeatures, macro_dir, macro_file
    macro_dir(tmp_paths).mkdir(parents=True, exist_ok=True)
    # Build a tz-AWARE series and write it — replicates yfinance's behaviour.
    dates = pd.date_range("2020-01-01", periods=20, freq="B", tz="America/New_York")
    pd.DataFrame({
        "date": dates,
        "close": np.linspace(15.0, 25.0, 20).astype("float32"),
    }).to_parquet(macro_file(tmp_paths, "vix"), index=False)

    m = MacroFeatures.from_paths(tmp_paths)
    assert "vix" in m.series
    # Index must be tz-naive after load.
    assert m.series["vix"].index.tz is None
    # Querying with a tz-naive Timestamp must not raise.
    feat = m.get_features(pd.Timestamp("2020-01-15"))
    assert feat.shape == (m.n_features,)
    assert np.isfinite(feat).all()


def test_macro_features_clip_extreme_outliers():
    """Values >|1| should be clipped (data-glitch protection)."""
    from constellation_quant.data.macro import MacroFeatures
    # Manually construct a series with a 100× spike → log change > 4
    dates = pd.bdate_range("2020-01-01", periods=10)
    vix = pd.Series([10.0] * 5 + [10000.0] * 5, index=dates).astype(float)
    m = MacroFeatures(series={"vix": vix})
    feat = m.get_features(dates[7])
    # Spike inside the 5-day window — the 5d log change is huge but should clip.
    assert abs(feat[0]) <= 1.0


def test_dataset_with_macro_features(tmp_paths, synthetic_price_frame):
    """Dataset emits slow_features extended by 4 macro broadcast values."""
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset
    from constellation_quant.data.macro import (
        MacroFeatures, macro_dir, macro_file, MACRO_FEATURE_COLUMNS,
    )

    members = {"T000"}
    snapshots = {date(2020, 1, 2): members}
    roster = MembershipRoster.from_daily_snapshots(snapshots)
    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)

    # Provide all 4 macro series as fake constants.
    macro_dir(tmp_paths).mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2020-01-01", periods=200)
    for name, mult in (("vix", 15.0), ("tnx", 2.0), ("dxy", 100.0), ("spy", 400.0)):
        s = pd.DataFrame({"date": dates, "close": [mult] * 200})
        s.to_parquet(macro_file(tmp_paths, name), index=False)

    macro = MacroFeatures.from_paths(tmp_paths)
    assert not macro.is_empty()

    ds = DynaGraphDataset(
        paths=tmp_paths, membership=roster,
        start_date="2020-06-01", end_date="2020-09-30",
        lookback=30, horizon=5, features="technical",
        macro_features=macro,
    )
    sample = ds[0]
    # slow_features now includes 8 stock-specific + 4 macro = 12 dims.
    assert sample["slow_features"].shape == (1, 8 + len(MACRO_FEATURE_COLUMNS))
    # All four macro values should be 0 (constant series → zero log change).
    macro_part = sample["slow_features"][0, 8:].numpy()
    assert (macro_part == 0).all()
    # Shapes report should advertise the extended slow count.
    assert ds.shapes().n_slow_features == 8 + len(MACRO_FEATURE_COLUMNS)


def test_dataset_epoch_offset_validates_range(tmp_paths, synthetic_price_frame):
    """epoch_offset must be in [0, stride)."""
    pytest.importorskip("torch")
    from constellation_quant.data.dataset import DynaGraphDataset

    synthetic_price_frame.to_parquet(tmp_paths.price_file("T000"), index=False)
    roster = MembershipRoster.from_daily_snapshots({date(2020, 1, 2): {"T000"}})

    with pytest.raises(ValueError, match="epoch_offset"):
        DynaGraphDataset(
            paths=tmp_paths, membership=roster,
            start_date="2020-05-01", end_date="2020-10-01",
            lookback=10, horizon=5, stride=5, epoch_offset=5,
        )
