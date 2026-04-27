"""Data pipeline: downloading, cleaning, caching, and serving market data."""

from constellation_quant.data._paths import DataPaths
from constellation_quant.data.membership import (
    MembershipRoster,
    build_roster_from_sources,
    validate_roster,
)
from constellation_quant.data.downloader import (
    DownloadError,
    DownloadReport,
    PriceDownloader,
)
from constellation_quant.data.fundamentals_downloader import (
    FundamentalsDownloader,
    FundamentalsError,
    FundamentalsReport,
)
from constellation_quant.data.sentiment_downloader import (
    FinVizSource,
    RedditSource,
    SentimentDownloader,
    SentimentReport,
    SentimentSource,
    StockTwitsSource,
)
from constellation_quant.data.cleaner import (
    CleaningReport,
    DataCleaner,
    apply_ticker_aliases,
    build_trading_calendar,
    detect_revert_spikes,
    drop_duplicates,
    drop_nan_prices,
    forward_fill_small_gaps,
    remove_revert_spikes,
)
from constellation_quant.data.cache import CacheEntry, CacheManager
from constellation_quant.data.macro import (
    MACRO_FEATURE_COLUMNS,
    MACRO_TICKERS,
    MacroFeatures,
    download_macro,
)

__all__ = [
    "DataPaths",
    # Membership
    "MembershipRoster",
    "build_roster_from_sources",
    "validate_roster",
    # Downloaders
    "PriceDownloader",
    "DownloadReport",
    "DownloadError",
    "FundamentalsDownloader",
    "FundamentalsReport",
    "FundamentalsError",
    "SentimentDownloader",
    "SentimentReport",
    "SentimentSource",
    "StockTwitsSource",
    "FinVizSource",
    "RedditSource",
    # Cleaning
    "DataCleaner",
    "CleaningReport",
    "drop_duplicates",
    "drop_nan_prices",
    "detect_revert_spikes",
    "remove_revert_spikes",
    "forward_fill_small_gaps",
    "apply_ticker_aliases",
    "build_trading_calendar",
    # Cache
    "CacheManager",
    "CacheEntry",
    # Macro
    "MacroFeatures",
    "MACRO_FEATURE_COLUMNS",
    "MACRO_TICKERS",
    "download_macro",
]


# ── Dataset is imported lazily — torch is heavy and may not be installed. ──


def __getattr__(name: str):
    if name in {"DynaGraphDataset", "collate_graph_samples", "SampleShapes"}:
        from constellation_quant.data import dataset as _dataset
        return getattr(_dataset, name)
    raise AttributeError(f"module 'constellation_quant.data' has no attribute {name!r}")
