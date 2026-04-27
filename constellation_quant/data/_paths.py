"""Resolve paths.yaml entries into concrete `Path` objects.

Separates "where things live on disk" from every module that needs to read or
write data. A single `DataPaths` instance is constructed from `paths.yaml` and
passed down through the downloader / cleaner / dataset pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class DataPaths:
    """Resolved filesystem paths for the data pipeline."""

    data_dir: Path
    raw_prices: Path
    raw_fundamentals: Path
    raw_sentiment: Path
    processed: Path
    cache_dir: Path
    graphs_dir: Path
    membership_file: Path
    checkpoint_dir: Path
    outputs_dir: Path

    @classmethod
    def from_config(cls, paths_cfg: Dict[str, Any]) -> "DataPaths":
        """Construct from the dict returned by `load_config('paths.yaml')`.

        Unresolved `${VAR}` references (env var missing) are kept verbatim —
        the caller decides whether to error or accept that the path is
        symbolic. `ensure_dirs()` will raise if asked to create one.
        """
        data_dir = Path(paths_cfg["data_dir"])
        return cls(
            data_dir=data_dir,
            raw_prices=data_dir / "raw" / "prices",
            raw_fundamentals=data_dir / "raw" / "fundamentals",
            raw_sentiment=data_dir / "raw" / "sentiment",
            processed=Path(paths_cfg["processed_data"]),
            cache_dir=Path(paths_cfg["cache_dir"]),
            graphs_dir=Path(paths_cfg["graphs_dir"]),
            membership_file=Path(paths_cfg["membership_file"]),
            checkpoint_dir=Path(paths_cfg["checkpoint_dir"]),
            outputs_dir=Path(paths_cfg["outputs_dir"]),
        )

    def ensure_dirs(self) -> None:
        """mkdir -p every data directory. Skips files (membership_file)."""
        dir_fields = {
            "data_dir", "raw_prices", "raw_fundamentals", "raw_sentiment",
            "processed", "cache_dir", "graphs_dir", "checkpoint_dir", "outputs_dir",
        }
        for f in fields(self):
            if f.name not in dir_fields:
                continue
            path: Path = getattr(self, f.name)
            if "$" in str(path):
                raise ValueError(
                    f"Cannot create {f.name}={path!r}: unresolved env var. "
                    "Set SCRATCH / DATA_DIR / PROJECT_ROOT before running."
                )
            path.mkdir(parents=True, exist_ok=True)

    def price_file(self, ticker: str) -> Path:
        """Parquet location for a single ticker's price series."""
        return self.raw_prices / f"{ticker.upper()}.parquet"

    def fundamentals_file(self, ticker: str) -> Path:
        """Parquet location for a single ticker's fundamentals."""
        return self.raw_fundamentals / f"{ticker.upper()}.parquet"

    def sentiment_file(self, ticker: str) -> Path:
        """Parquet location for a single ticker's sentiment."""
        return self.raw_sentiment / f"{ticker.upper()}.parquet"
