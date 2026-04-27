"""Sentiment downloader — optional feature used by ablation Model F+.

Per source, fetches recent messages/headlines, maps them to a per-date
sentiment score in [-1, +1], and writes long-format parquet per ticker:

    date, source, score, volume

`source` is the provider name ("stocktwits", "finviz", "reddit"), `score` is
the daily-aggregated sentiment, `volume` is the message/headline count used
to compute the score (useful as a "mention spike" feature downstream).

This module is best-effort — missing data is neutral (0.0), never fatal.
Individual sources can be disabled via `data_config.yaml`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from constellation_quant.data._paths import DataPaths
from constellation_quant.utils import get_logger

log = get_logger(__name__)


# ── Base source ────────────────────────────────────────────────────────────


class SentimentSource(ABC):
    """Interface for a sentiment provider. All scores normalised to [-1, +1]."""

    name: str = ""

    @abstractmethod
    def fetch(self, ticker: str) -> "object":
        """Return a DataFrame[date, source, score, volume]. May be empty."""

    @staticmethod
    def _empty_frame() -> "object":
        import pandas as pd
        return pd.DataFrame(columns=["date", "source", "score", "volume"])


# ── StockTwits ─────────────────────────────────────────────────────────────


class StockTwitsSource(SentimentSource):
    """StockTwits public API — user-tagged Bullish/Bearish → score.

    API: `https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json`

    Messages carry `entities.sentiment.basic` ∈ {Bullish, Bearish} when the
    author self-tagged. Untagged messages are excluded from the score but
    counted in `volume`. One stream request returns up to 30 messages, so
    this is really only suitable for *current* sentiment — historical
    backfill requires pagination or the paid API.
    """

    name = "stocktwits"
    BASE_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"

    def __init__(self, user_agent: str = "ConstellationQuant/0.1"):
        self.user_agent = user_agent

    def fetch(self, ticker: str) -> "object":
        import pandas as pd
        import requests

        url = self.BASE_URL.format(symbol=ticker.upper())
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": self.user_agent})
            r.raise_for_status()
            payload = r.json()
        except Exception as exc:
            log.debug("stocktwits fetch [{}] failed: {}", ticker, exc)
            return self._empty_frame()

        messages = payload.get("messages", [])
        if not messages:
            return self._empty_frame()

        rows: List[Dict] = []
        for msg in messages:
            created = msg.get("created_at")
            if not created:
                continue
            ts = pd.to_datetime(created, errors="coerce", utc=True)
            if pd.isna(ts):
                continue
            sentiment = (
                (msg.get("entities") or {})
                .get("sentiment", {}) or {}
            ).get("basic")
            rows.append({
                "date": ts.tz_convert(None).normalize() if ts.tzinfo else ts.normalize(),
                "sentiment": sentiment,
            })
        if not rows:
            return self._empty_frame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["score_raw"] = df["sentiment"].map({"Bullish": 1.0, "Bearish": -1.0})
        grouped = df.groupby(df["date"].dt.normalize()).agg(
            score=("score_raw", "mean"),
            volume=("sentiment", "size"),
        ).reset_index()
        grouped["score"] = grouped["score"].fillna(0.0)
        grouped["source"] = self.name
        return grouped[["date", "source", "score", "volume"]]


# ── FinViz ─────────────────────────────────────────────────────────────────


class FinVizSource(SentimentSource):
    """FinViz news headlines → headline volume with polarity via keyword dict.

    FinViz does not publish a numeric sentiment score. We approximate polarity
    with a small positive/negative keyword list. This is intentionally crude
    — finely-calibrated sentiment is future work; right now the primary
    signal is headline *volume* relative to baseline, which the feature
    engine picks up in `features/sentiment.py`.
    """

    name = "finviz"
    BASE_URL = "https://finviz.com/quote.ashx?t={symbol}"
    POS_WORDS = {
        "upgrade", "beat", "surge", "rally", "record", "outperform", "raise",
        "strong", "gain", "soar", "boost", "grow",
    }
    NEG_WORDS = {
        "downgrade", "miss", "fall", "drop", "plunge", "underperform", "cut",
        "weak", "loss", "slump", "decline", "sink",
    }

    def __init__(self, user_agent: str = "ConstellationQuant/0.1"):
        self.user_agent = user_agent

    def fetch(self, ticker: str) -> "object":
        import pandas as pd
        import requests
        from bs4 import BeautifulSoup

        url = self.BASE_URL.format(symbol=ticker.upper())
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": self.user_agent})
            r.raise_for_status()
        except Exception as exc:
            log.debug("finviz fetch [{}] failed: {}", ticker, exc)
            return self._empty_frame()

        soup = BeautifulSoup(r.text, "lxml")
        news_table = soup.find("table", {"id": "news-table"})
        if news_table is None:
            return self._empty_frame()

        rows: List[Dict] = []
        current_date: Optional[str] = None
        for tr in news_table.find_all("tr"):
            cells = tr.find_all("td")
            if len(cells) < 2:
                continue
            ts_raw = cells[0].get_text(strip=True)
            title = cells[1].get_text(strip=True)
            parsed_date, current_date = self._parse_finviz_time(ts_raw, current_date)
            if parsed_date is None:
                continue
            rows.append({"date": parsed_date, "headline": title.lower()})

        if not rows:
            return self._empty_frame()

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["pol"] = df["headline"].apply(self._polarity)
        grouped = df.groupby("date").agg(
            score=("pol", "mean"),
            volume=("pol", "size"),
        ).reset_index()
        grouped["score"] = grouped["score"].fillna(0.0)
        grouped["source"] = self.name
        return grouped[["date", "source", "score", "volume"]]

    @classmethod
    def _polarity(cls, headline: str) -> float:
        tokens = set(headline.split())
        pos = len(tokens & cls.POS_WORDS)
        neg = len(tokens & cls.NEG_WORDS)
        if pos == 0 and neg == 0:
            return 0.0
        return (pos - neg) / max(pos + neg, 1)

    @staticmethod
    def _parse_finviz_time(raw: str, current_date: Optional[str]):
        """FinViz timestamps are either 'Mon-DD-YY HH:MMAM' or just 'HH:MMAM'."""
        import pandas as pd

        parts = raw.split()
        if len(parts) == 2:
            try:
                d = pd.to_datetime(parts[0], format="%b-%d-%y").normalize()
                return d, parts[0]
            except ValueError:
                return None, current_date
        if current_date is not None:
            try:
                d = pd.to_datetime(current_date, format="%b-%d-%y").normalize()
                return d, current_date
            except ValueError:
                return None, current_date
        return None, current_date


# ── Reddit (stub — requires credentials) ───────────────────────────────────


class RedditSource(SentimentSource):
    """Placeholder for a Reddit sentiment scraper (requires praw + credentials).

    Returns empty DataFrame until the user wires up credentials via
    data_config.yaml. Kept as a class so the downloader's source list stays
    symmetric.
    """

    name = "reddit"

    def __init__(self, *, enabled: bool = False):
        self.enabled = enabled

    def fetch(self, ticker: str) -> "object":
        if not self.enabled:
            return self._empty_frame()
        log.warning("RedditSource enabled but not implemented — returning empty.")
        return self._empty_frame()


# ── Orchestrator ───────────────────────────────────────────────────────────


@dataclass
class SentimentReport:
    succeeded: List[str] = field(default_factory=list)
    skipped:   List[str] = field(default_factory=list)
    failed:    Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        total = len(self.succeeded) + len(self.skipped) + len(self.failed)
        return (
            f"sentiment: downloaded={len(self.succeeded)} "
            f"skipped={len(self.skipped)} failed={len(self.failed)} total={total}"
        )


class SentimentDownloader:
    """Iterate tickers × sources, concatenate into long-format parquet per ticker."""

    def __init__(
        self,
        paths: DataPaths,
        sources: Optional[Iterable[SentimentSource]] = None,
        sleep_between: float = 0.5,
    ):
        self.paths = paths
        self.sources = list(sources) if sources is not None else [
            StockTwitsSource(),
            FinVizSource(),
            RedditSource(enabled=False),
        ]
        self.sleep_between = sleep_between

    def download_all(
        self,
        tickers: Iterable[str],
        resume: bool = True,
    ) -> SentimentReport:
        import pandas as pd

        report = SentimentReport()
        self.paths.raw_sentiment.mkdir(parents=True, exist_ok=True)

        for ticker in sorted({t.upper().strip() for t in tickers if t}):
            out_path = self.paths.sentiment_file(ticker)
            if resume and out_path.exists():
                report.skipped.append(ticker)
                continue
            frames: List["object"] = []
            for src in self.sources:
                try:
                    df = src.fetch(ticker)
                    if df is not None and not df.empty:
                        frames.append(df)
                except Exception as exc:
                    log.warning("  [{}] source {} failed: {}", ticker, src.name, exc)
                if self.sleep_between > 0:
                    time.sleep(self.sleep_between)

            if not frames:
                report.failed[ticker] = "no sources returned data"
                continue
            combined = pd.concat(frames, ignore_index=True)
            combined["date"] = pd.to_datetime(combined["date"]).dt.tz_localize(None)
            combined = combined.sort_values(["date", "source"]).reset_index(drop=True)
            combined.to_parquet(out_path, index=False, compression="snappy")
            report.succeeded.append(ticker)

        log.info(report.summary())
        return report
