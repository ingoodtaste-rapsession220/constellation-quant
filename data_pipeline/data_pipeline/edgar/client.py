"""SEC EDGAR HTTP client.

Polite, rate-limited, resume-safe wrapper around the SEC EDGAR REST surface.

EDGAR's fair-use policy:
  - Max 10 requests per second per IP
  - Must include a descriptive User-Agent with a real contact email
  - https://www.sec.gov/os/accessing-edgar-data

This client honours both. Retries with exponential backoff on transient
errors. Caches the company-tickers manifest locally.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


_RATE_LIMIT_QPS = 10.0     # SEC fair-use cap
_DEFAULT_TIMEOUT = 30
_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.HTTPError,
)


# --------------------------------------------------------------- rate limiter
class _RateLimiter:
    """Token-bucket rate limiter, thread-safe."""

    def __init__(self, qps: float):
        self._interval = 1.0 / qps
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_at = now + self._interval


# --------------------------------------------------------------- data classes
@dataclass(frozen=True)
class FilingMeta:
    cik: str            # zero-padded 10-digit string, e.g. "0000320193"
    accession: str      # accession number with dashes, e.g. "0000320193-24-000123"
    form: str           # "10-K", "10-Q", etc.
    filing_date: str    # YYYY-MM-DD
    period: str         # YYYY-MM-DD reporting period end
    primary_doc: str    # primary document filename, e.g. "aapl-20240928.htm"

    @property
    def accession_no_dashes(self) -> str:
        return self.accession.replace("-", "")

    def primary_doc_url(self) -> str:
        return (
            f"https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{self.accession_no_dashes}/{self.primary_doc}"
        )


# --------------------------------------------------------------- main client
class EdgarClient:
    """HTTP client for the SEC EDGAR submissions and archives API.

    Usage:
        client = EdgarClient(user_agent="constellation-quant research nikraftarz@gmail.com")
        cik = client.lookup_cik("AAPL")
        filings = list(client.list_filings(cik, forms=("10-K", "10-Q")))
        for f in filings:
            text = client.fetch_filing_text(f)
            ...
    """

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

    def __init__(
        self,
        user_agent: str,
        cache_dir: Optional[Path] = None,
        qps: float = _RATE_LIMIT_QPS,
        timeout: int = _DEFAULT_TIMEOUT,
    ):
        if "@" not in user_agent:
            raise ValueError(
                "EDGAR User-Agent must include a contact email "
                "(SEC fair-use policy)."
            )
        self.user_agent = user_agent
        self.timeout = timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._limiter = _RateLimiter(qps)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._tickers_cache: Optional[dict[str, str]] = None

    # -------------------------------- low-level GET (rate-limited + retried)
    @retry(
        retry=retry_if_exception_type(_RETRY_EXCEPTIONS),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _get(self, url: str) -> requests.Response:
        self._limiter.wait()
        resp = self._session.get(url, timeout=self.timeout)
        # 429 = rate-limit hit (shouldn't happen with our limiter, but be defensive)
        if resp.status_code == 429:
            time.sleep(2)
            raise requests.exceptions.HTTPError("429 Too Many Requests")
        # 5xx = transient; raise so tenacity retries
        if 500 <= resp.status_code < 600:
            raise requests.exceptions.HTTPError(f"{resp.status_code}")
        # 403/404 are permanent — surface them
        resp.raise_for_status()
        return resp

    # -------------------------------- ticker → CIK
    def _load_tickers_manifest(self) -> dict[str, str]:
        """Returns ticker -> CIK (10-digit zero-padded)."""
        if self._tickers_cache is not None:
            return self._tickers_cache

        cache_file = (
            self.cache_dir / "company_tickers.json" if self.cache_dir else None
        )
        if cache_file and cache_file.exists():
            data = json.loads(cache_file.read_text())
        else:
            data = self._get(self.TICKERS_URL).json()
            if cache_file:
                cache_file.write_text(json.dumps(data))

        # Map: { "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ... }
        mapping: dict[str, str] = {}
        for entry in data.values():
            ticker = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)
            mapping[ticker] = cik
        self._tickers_cache = mapping
        return mapping

    def lookup_cik(self, ticker: str) -> Optional[str]:
        """Returns 10-digit zero-padded CIK string, or None if unknown."""
        return self._load_tickers_manifest().get(ticker.upper())

    # -------------------------------- list filings for a CIK
    def list_filings(
        self,
        cik: str,
        forms: tuple[str, ...] = ("10-K", "10-Q"),
        date_from: Optional[str] = None,   # YYYY-MM-DD
        date_to: Optional[str] = None,
    ):
        """Yield FilingMeta entries for all filings of the given forms.

        Note: the recent submissions endpoint only includes the last ~1000
        filings. For older filings, a paginated submissions file is referenced
        (we follow it automatically).
        """
        cik = cik.zfill(10)
        url = self.SUBMISSIONS_URL.format(cik=cik)
        body = self._get(url).json()
        recent = body.get("filings", {}).get("recent", {})
        yield from self._yield_filings(cik, recent, forms, date_from, date_to)

        # Paginated older submissions
        for older in body.get("filings", {}).get("files", []):
            older_url = (
                f"https://data.sec.gov/submissions/{older['name']}"
            )
            older_body = self._get(older_url).json()
            yield from self._yield_filings(
                cik, older_body, forms, date_from, date_to
            )

    def _yield_filings(self, cik, table, forms, date_from, date_to):
        n = len(table.get("accessionNumber", []))
        for i in range(n):
            form = table["form"][i]
            if forms and form not in forms:
                continue
            filing_date = table["filingDate"][i]
            if date_from and filing_date < date_from:
                continue
            if date_to and filing_date > date_to:
                continue
            accession = table["accessionNumber"][i]
            period = table["reportDate"][i] or filing_date
            primary_doc = table["primaryDocument"][i]
            yield FilingMeta(
                cik=cik,
                accession=accession,
                form=form,
                filing_date=filing_date,
                period=period,
                primary_doc=primary_doc,
            )

    # -------------------------------- fetch the actual filing text
    def fetch_filing_text(self, filing: FilingMeta) -> str:
        """Returns the raw text of the primary document (HTML or .txt)."""
        url = filing.primary_doc_url()
        return self._get(url).text
