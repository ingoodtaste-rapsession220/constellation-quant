"""On-disk layout for raw + parsed filings.

Raw filings:
    <root>/raw/edgar/<CIK>/<accession>.txt

Parsed filings (one parquet per CIK, append-only):
    <root>/processed/edgar/<CIK>/filings.parquet

Manifest (one CSV at the root, tracks what we have):
    <root>/raw/edgar/_manifest.csv
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict
from pathlib import Path
from threading import Lock
from typing import Iterable, Optional

import pandas as pd

from data_pipeline.edgar.client import FilingMeta
from data_pipeline.edgar.parser import ParsedFiling

logger = logging.getLogger(__name__)


_MANIFEST_COLUMNS = [
    "cik", "accession", "form", "filing_date", "period",
    "primary_doc", "raw_path", "downloaded_at",
]


class FilingsStorage:
    """File-system-backed storage for raw and parsed filings."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.raw_dir = self.root / "raw" / "edgar"
        self.parsed_dir = self.root / "processed" / "edgar"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.raw_dir / "_manifest.csv"
        self._manifest_lock = Lock()
        self._ensure_manifest()

    # -------------------------------- manifest
    def _ensure_manifest(self) -> None:
        if not self.manifest_path.exists():
            with self.manifest_path.open("w", newline="") as f:
                csv.writer(f).writerow(_MANIFEST_COLUMNS)

    def already_downloaded(self) -> set[tuple[str, str]]:
        """Returns set of (cik, accession) already in the manifest."""
        out: set[tuple[str, str]] = set()
        if not self.manifest_path.exists():
            return out
        with self.manifest_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                out.add((row["cik"], row["accession"]))
        return out

    def _append_manifest(self, row: dict) -> None:
        with self._manifest_lock, self.manifest_path.open("a", newline="") as f:
            csv.writer(f).writerow([row.get(c, "") for c in _MANIFEST_COLUMNS])

    # -------------------------------- raw filings
    def save_raw_filing(self, filing: FilingMeta, text: str) -> Path:
        cik_dir = self.raw_dir / filing.cik
        cik_dir.mkdir(parents=True, exist_ok=True)
        path = cik_dir / f"{filing.accession}.txt"
        path.write_text(text, encoding="utf-8", errors="replace")

        from datetime import datetime, timezone
        row = {
            **asdict(filing),
            "raw_path": str(path.relative_to(self.root)),
            "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._append_manifest(row)
        return path

    def load_raw_filing(self, cik: str, accession: str) -> Optional[str]:
        path = self.raw_dir / cik / f"{accession}.txt"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="replace")

    def iter_raw(self) -> Iterable[tuple[FilingMeta, str]]:
        """Yield (FilingMeta, raw_text) for every entry in the manifest.

        Reading lazily so we don't blow memory on the full universe.
        """
        if not self.manifest_path.exists():
            return
        with self.manifest_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                meta = FilingMeta(
                    cik=row["cik"],
                    accession=row["accession"],
                    form=row["form"],
                    filing_date=row["filing_date"],
                    period=row["period"],
                    primary_doc=row["primary_doc"],
                )
                text = self.load_raw_filing(meta.cik, meta.accession)
                if text is not None:
                    yield meta, text

    # -------------------------------- parsed filings (parquet, one per CIK)
    def save_parsed_filings(self, cik: str, parsed: list[ParsedFiling]) -> Path:
        """Append (or overwrite) one parquet of parsed filings for a CIK."""
        cik_dir = self.parsed_dir / cik
        cik_dir.mkdir(parents=True, exist_ok=True)
        path = cik_dir / "filings.parquet"
        df = pd.DataFrame([
            {
                "cik": p.cik,
                "accession": p.accession,
                "form": p.form,
                "filing_date": p.filing_date,
                "period": p.period,
                "risk_factors": p.risk_factors,
                "mda": p.mda,
                "market_risk": p.market_risk,
                "raw_length": p.raw_length,
                "len_risk_factors": p.extracted_lengths.get("risk_factors", 0),
                "len_mda": p.extracted_lengths.get("mda", 0),
                "len_market_risk": p.extracted_lengths.get("market_risk", 0),
            }
            for p in parsed
        ])
        df.to_parquet(path, index=False, compression="snappy")
        return path

    def load_parsed_filings(self, cik: str) -> Optional[pd.DataFrame]:
        path = self.parsed_dir / cik / "filings.parquet"
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def iter_parsed_per_cik(self) -> Iterable[tuple[str, pd.DataFrame]]:
        """Yield (cik, parsed_df) for every CIK with parsed filings on disk."""
        for cik_dir in sorted(self.parsed_dir.iterdir()):
            if not cik_dir.is_dir():
                continue
            df = self.load_parsed_filings(cik_dir.name)
            if df is not None:
                yield cik_dir.name, df

    # -------------------------------- combined parsed parquet
    def write_combined_parsed(self, out_path: Path) -> Path:
        """Concatenate all per-CIK parsed parquets into one big parquet.

        Useful for the NLP stages, which want a single iteration order.
        """
        frames = []
        for _, df in self.iter_parsed_per_cik():
            frames.append(df)
        if not frames:
            raise RuntimeError("No parsed filings to combine.")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        full = pd.concat(frames, ignore_index=True)
        full.to_parquet(out_path, index=False, compression="snappy")
        return out_path
