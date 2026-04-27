"""S&P 500 historical membership roster.

Eliminates survivorship bias by maintaining a time-stamped lookup of which
tickers were in the index on every trading day, using actual historical
addition / removal events — not the current 503 constituents projected
backwards.

Primary source:  fja05680/sp500 GitHub repo (CSV with per-date snapshots)
Fallback:        Wikipedia scrape (current list + component-changes page)
Cross-check:     Known events — Tesla added 2020-12-21, Meta added 2013-12-23

The `MembershipRoster` is serialised as JSON under `paths.membership_file` and
consumed by the downloader (to decide which tickers to fetch) and the Dataset
(to filter the graph to actual historical members at each date).
"""

from __future__ import annotations

import bisect
import io
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple

from constellation_quant.utils import get_logger

log = get_logger(__name__)


# ── Constants ───────────────────────────────────────────────────────────────

# Primary source. The file name in the fja05680 repo changes periodically as
# the maintainer re-publishes updated snapshots. The raw path pattern stays
# stable; we allow an override via data_config.yaml.
FJA05680_RAW_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes(08-17-2024).csv"
)

WIKIPEDIA_SP500_URL   = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKIPEDIA_CHANGES_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Selected_changes_to_the_list_of_S&P_500_components"

# Known events used for cross-validation. Key: ticker, Value: (action, date).
# These are well-documented in press releases; any roster that fails these
# checks is broken.
KNOWN_EVENTS: Dict[str, Tuple[str, date]] = {
    "TSLA": ("added", date(2020, 12, 21)),
    "META": ("added", date(2013, 12, 23)),   # was FB at time of addition
    "GOOG": ("added", date(2006, 4, 3)),     # initially as GOOG (Class C)
}

# Plausible membership-count bounds on any date after 2001.
# Historically 500 names; periods with multi-class listings (BRK.A/B,
# GOOG/GOOGL, NWS/NWSA, FOX/FOXA) pushed counts as high as ~518.
MIN_MEMBERS_POST_2001 = 490
MAX_MEMBERS_POST_2001 = 520

# Tolerance window around known-event dates. Public sources sometimes mark
# the snapshot on the day before or after the actual index change took
# effect (press-release vs first-trading-session discrepancy).
KNOWN_EVENT_TOLERANCE_DAYS = 3


# ── Data class ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MembershipRoster:
    """Time-stamped S&P 500 membership lookup.

    Internally stores daily snapshots: `date -> frozenset of tickers`.
    Lookups for unseen dates use the most-recent snapshot at or before the
    query date (bisect, O(log N)).

    Invariant: `_sorted_dates` is sorted ascending and matches the keys of
    `_snapshots` 1:1.
    """

    _snapshots: Mapping[date, FrozenSet[str]]
    _sorted_dates: Tuple[date, ...]

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def from_daily_snapshots(
        cls,
        snapshots: Mapping[date, Iterable[str]],
    ) -> "MembershipRoster":
        """Build from a mapping of date -> iterable of tickers."""
        if not snapshots:
            raise ValueError("Cannot build an empty MembershipRoster.")
        normalised = {
            d: frozenset(t.upper().strip() for t in tickers if t)
            for d, tickers in snapshots.items()
        }
        sorted_dates = tuple(sorted(normalised.keys()))
        return cls(_snapshots=normalised, _sorted_dates=sorted_dates)

    # ── Queries ────────────────────────────────────────────────────────

    def tickers_on(self, d: date | str) -> FrozenSet[str]:
        """Return the set of tickers in the S&P 500 on `d`.

        If `d` falls between snapshot dates, returns the most recent snapshot
        at or before `d`. Raises KeyError if `d` precedes the earliest
        snapshot.
        """
        query = _coerce_date(d)
        idx = bisect.bisect_right(self._sorted_dates, query) - 1
        if idx < 0:
            raise KeyError(
                f"No roster data on or before {query}. "
                f"Earliest snapshot is {self._sorted_dates[0]}."
            )
        return self._snapshots[self._sorted_dates[idx]]

    def all_tickers_ever(self) -> FrozenSet[str]:
        """Union of every ticker that ever appeared across all snapshots."""
        out: set[str] = set()
        for tickers in self._snapshots.values():
            out.update(tickers)
        return frozenset(out)

    def count_on(self, d: date | str) -> int:
        return len(self.tickers_on(d))

    def snapshot_dates(self) -> Tuple[date, ...]:
        return self._sorted_dates

    def additions(self, d: date | str) -> FrozenSet[str]:
        """Tickers added on `d` (i.e. in `d` but not in the previous snapshot)."""
        query = _coerce_date(d)
        idx = bisect.bisect_left(self._sorted_dates, query)
        if idx == 0 or self._sorted_dates[idx] != query:
            return frozenset()
        prev = self._snapshots[self._sorted_dates[idx - 1]]
        curr = self._snapshots[self._sorted_dates[idx]]
        return frozenset(curr - prev)

    def removals(self, d: date | str) -> FrozenSet[str]:
        """Tickers removed on `d`."""
        query = _coerce_date(d)
        idx = bisect.bisect_left(self._sorted_dates, query)
        if idx == 0 or self._sorted_dates[idx] != query:
            return frozenset()
        prev = self._snapshots[self._sorted_dates[idx - 1]]
        curr = self._snapshots[self._sorted_dates[idx]]
        return frozenset(prev - curr)

    # ── Serialization ──────────────────────────────────────────────────

    def save_json(self, path: Path) -> None:
        """Persist the roster as JSON: {"YYYY-MM-DD": [tickers, ...]}."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            d.isoformat(): sorted(self._snapshots[d])
            for d in self._sorted_dates
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        log.info("Saved roster: {} snapshots -> {}", len(self._sorted_dates), path)

    @classmethod
    def load_json(cls, path: Path) -> "MembershipRoster":
        with path.open("r") as f:
            raw: Dict[str, List[str]] = json.load(f)
        snapshots = {
            _coerce_date(date_str): tickers
            for date_str, tickers in raw.items()
        }
        return cls.from_daily_snapshots(snapshots)


# ── Parsers (pure — no network) ─────────────────────────────────────────────


def parse_fja05680_csv(csv_text: str) -> Dict[date, FrozenSet[str]]:
    """Parse the fja05680/sp500 CSV into daily snapshots.

    Expected schema:
        date,tickers
        "1996-01-02","MMM,AOS,ABT,ACN,..."

    Tickers within the quoted field are comma-separated (no spaces).
    """
    import pandas as pd

    df = pd.read_csv(io.StringIO(csv_text))
    lower_cols = [c.lower().strip() for c in df.columns]
    if "date" not in lower_cols or "tickers" not in lower_cols:
        raise ValueError(
            f"Unexpected fja05680 schema: columns={list(df.columns)}"
        )
    df.columns = lower_cols

    snapshots: Dict[date, FrozenSet[str]] = {}
    for _, row in df.iterrows():
        d = _coerce_date(row["date"])
        tickers = [t.strip().upper() for t in str(row["tickers"]).split(",") if t.strip()]
        snapshots[d] = frozenset(tickers)
    if not snapshots:
        raise ValueError("fja05680 CSV produced zero snapshots after parsing.")
    return snapshots


def parse_wikipedia_current_list(html: str) -> FrozenSet[str]:
    """Extract the present-day S&P 500 ticker list from the Wikipedia HTML."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise ValueError("Wikipedia list page: no #constituents table found.")

    tickers: List[str] = []
    for tr in table.find_all("tr")[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        sym = cells[0].get_text(strip=True)
        # Normalise Wikipedia's "BRK.B" style to yfinance's "BRK-B"
        sym = sym.replace(".", "-").upper()
        if sym:
            tickers.append(sym)
    if not tickers:
        raise ValueError("Wikipedia list page: no tickers extracted.")
    return frozenset(tickers)


def parse_wikipedia_changes(html: str) -> List[Tuple[date, str, str]]:
    """Extract (date, ticker, action) triples from the changes table.

    `action` is either "added" or "removed". Returns a list sorted
    chronologically. The Wikipedia page uses a merged-cell layout where a
    single change row can include both an addition and a removal.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    changes_table = None
    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(" ", strip=True).lower() for th in table.find_all("th")]
        if any("added" in h for h in headers) and any("removed" in h for h in headers):
            changes_table = table
            break
    if changes_table is None:
        raise ValueError("Wikipedia changes page: no matching table found.")

    events: List[Tuple[date, str, str]] = []
    for tr in changes_table.find_all("tr")[2:]:  # skip header rows
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if len(cells) < 3:
            continue
        try:
            event_date = _coerce_date(cells[0])
        except ValueError:
            continue
        added_sym = _clean_ticker_cell(cells[1]) if len(cells) > 1 else ""
        removed_sym = _clean_ticker_cell(cells[3]) if len(cells) > 3 else ""
        if added_sym:
            events.append((event_date, added_sym, "added"))
        if removed_sym:
            events.append((event_date, removed_sym, "removed"))
    events.sort(key=lambda e: e[0])
    return events


# ── Network layer (thin wrappers for testability) ──────────────────────────


def fetch_fja05680_csv(url: str = FJA05680_RAW_URL) -> str:
    """HTTP GET the fja05680 CSV. Raises on non-200."""
    import requests
    log.info("Fetching fja05680 roster: {}", url)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def fetch_wikipedia_page(url: str) -> str:
    import requests
    r = requests.get(url, timeout=30, headers={"User-Agent": "ConstellationQuant/0.1"})
    r.raise_for_status()
    return r.text


# ── Validation ─────────────────────────────────────────────────────────────


def validate_roster(roster: MembershipRoster) -> List[str]:
    """Run sanity checks. Returns a list of human-readable error strings.

    An empty list means the roster passed all checks. The caller decides
    whether to treat failures as hard errors or warnings.
    """
    errors: List[str] = []

    # 1. Known events — ticker must appear within ±KNOWN_EVENT_TOLERANCE_DAYS
    #    of the documented event date. Sources sometimes lag/lead by a day.
    from datetime import timedelta as _td
    for ticker, (action, event_date) in KNOWN_EVENTS.items():
        if action != "added":
            continue
        found = False
        for offset in range(KNOWN_EVENT_TOLERANCE_DAYS + 1):
            for d in (event_date + _td(days=offset), event_date - _td(days=offset)):
                try:
                    if ticker in roster.tickers_on(d):
                        found = True
                        break
                except KeyError:
                    continue
            if found:
                break
        if not found:
            errors.append(
                f"{ticker} not in roster within ±{KNOWN_EVENT_TOLERANCE_DAYS}d of {event_date}"
            )

    # 2. Membership counts in a plausible range on every post-2001 snapshot.
    for d in roster.snapshot_dates():
        if d.year < 2001:
            continue
        n = roster.count_on(d)
        if not (MIN_MEMBERS_POST_2001 <= n <= MAX_MEMBERS_POST_2001):
            errors.append(
                f"Implausible member count {n} on {d} "
                f"(expected {MIN_MEMBERS_POST_2001}-{MAX_MEMBERS_POST_2001})"
            )

    return errors


# ── Orchestration ──────────────────────────────────────────────────────────


def build_roster_from_sources(
    csv_url: Optional[str] = None,
    fallback_to_wikipedia: bool = True,
) -> MembershipRoster:
    """Fetch + parse the roster from the configured sources.

    Tries fja05680 first; if that fails and `fallback_to_wikipedia` is True,
    reconstructs from the Wikipedia current list + changes page. The
    Wikipedia path is lossy (only changes, no full history), so fja05680 is
    strongly preferred.
    """
    url = csv_url or FJA05680_RAW_URL
    try:
        text = fetch_fja05680_csv(url)
        snapshots = parse_fja05680_csv(text)
        return MembershipRoster.from_daily_snapshots(snapshots)
    except Exception as exc:
        if not fallback_to_wikipedia:
            raise
        log.warning("fja05680 fetch failed ({}); falling back to Wikipedia", exc)

    current = parse_wikipedia_current_list(fetch_wikipedia_page(WIKIPEDIA_SP500_URL))
    changes = parse_wikipedia_changes(fetch_wikipedia_page(WIKIPEDIA_CHANGES_URL))
    snapshots = _reconstruct_from_changes(current, changes)
    return MembershipRoster.from_daily_snapshots(snapshots)


def _reconstruct_from_changes(
    current: FrozenSet[str],
    changes: List[Tuple[date, str, str]],
) -> Dict[date, FrozenSet[str]]:
    """Walk the changes list backwards from today's roster to build daily snapshots."""
    members = set(current)
    today = date.today()
    snapshots: Dict[date, FrozenSet[str]] = {today: frozenset(members)}

    for event_date, ticker, action in sorted(changes, key=lambda e: e[0], reverse=True):
        if action == "added":
            members.discard(ticker)
        elif action == "removed":
            members.add(ticker)
        snapshots[event_date] = frozenset(members)
    return snapshots


# ── Helpers ────────────────────────────────────────────────────────────────


def _coerce_date(value: Any) -> date:
    """Accept `date`, `datetime`, or ISO-format strings; return `date`."""
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        s = value.strip()
        # Try ISO (YYYY-MM-DD) first — canonical; then a few Wikipedia formats
        for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        raise ValueError(f"Unrecognised date format: {value!r}")
    raise TypeError(f"Expected date/datetime/str, got {type(value).__name__}")


_TICKER_CELL_RE = re.compile(r"\b[A-Z][A-Z0-9.\-]{0,5}\b")


def _clean_ticker_cell(text: str) -> str:
    """Extract the first ticker symbol from a Wikipedia table cell."""
    m = _TICKER_CELL_RE.search(text)
    return m.group(0).replace(".", "-") if m else ""
