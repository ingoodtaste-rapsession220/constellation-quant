"""Filings parser — extract structured sections from raw 10-K / 10-Q HTML.

Focuses on the two sections that drive the "Lazy Prices" (Cohen, Malloy,
Pomorski 2020) language-drift signal:

  - Item 1A — Risk Factors   (10-K, sometimes 10-Q amended)
  - Item 7  — Management's Discussion and Analysis (MD&A)
  - Item 7A — Quantitative & Qualitative Disclosures (10-K only, optional)

10-Q filings have similar but renumbered items; we map them to a common
schema so downstream consumers see consistent column names.

Implementation notes
--------------------
SEC filings come in two flavours:
  1. Modern (post ~2001): inline XBRL HTML, with `<ix:...>` tags interleaved
     with regular HTML.
  2. Legacy: plain HTML, .txt with ASCII art tables, or SGML wrapper around
     embedded HTML/text.

We use BeautifulSoup with the "lxml" parser to handle both. Section-boundary
heuristics use case-insensitive regex on the visible text (after stripping
tags), since item numbering is typographically variable across filings.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# Case-insensitive section-boundary patterns.
# We match "Item 1A.", "ITEM 1A:", "Item&nbsp;1A", etc.
_SECTION_PATTERNS = {
    # 10-K sections
    "item_1a": re.compile(r"\bitem\s*1\s*a[\.\:\s]", re.IGNORECASE),
    "item_7":  re.compile(r"\bitem\s*7[\.\:\s]",      re.IGNORECASE),
    "item_7a": re.compile(r"\bitem\s*7\s*a[\.\:\s]",  re.IGNORECASE),
    "item_8":  re.compile(r"\bitem\s*8[\.\:\s]",      re.IGNORECASE),
    # 10-Q sections (Part I — financial info; Part II — other info)
    "part1_item2": re.compile(r"\bpart\s*i\s+item\s*2[\.\:\s]",  re.IGNORECASE),
    "part1_item3": re.compile(r"\bpart\s*i\s+item\s*3[\.\:\s]",  re.IGNORECASE),
    "part2_item1a": re.compile(r"\bpart\s*ii\s+item\s*1\s*a[\.\:\s]", re.IGNORECASE),
    "part2_item2":  re.compile(r"\bpart\s*ii\s+item\s*2[\.\:\s]",  re.IGNORECASE),
}


@dataclass
class ParsedFiling:
    """Common-schema parsed filing — shape independent of 10-K vs 10-Q."""

    cik: str
    accession: str
    form: str
    filing_date: str
    period: str
    risk_factors: str = ""    # Item 1A (10-K) or Part II Item 1A (10-Q)
    mda: str = ""             # Item 7 (10-K) or Part I Item 2 (10-Q)
    market_risk: str = ""     # Item 7A (10-K) or Part I Item 3 (10-Q), optional
    raw_length: int = 0
    extracted_lengths: dict[str, int] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.risk_factors or self.mda or self.market_risk)


# -----------------------------------------------------------------------------
class FilingsParser:
    """Stateless parser. Pass an HTML/txt blob, get a ParsedFiling back."""

    def parse(
        self,
        text: str,
        *,
        cik: str,
        accession: str,
        form: str,
        filing_date: str,
        period: str,
    ) -> ParsedFiling:
        # 1. Strip HTML, normalise whitespace.
        plain = self._html_to_text(text)
        plain = self._normalise_whitespace(plain)

        # 2. Extract sections according to form type.
        out = ParsedFiling(
            cik=cik,
            accession=accession,
            form=form,
            filing_date=filing_date,
            period=period,
            raw_length=len(plain),
        )

        if form.startswith("10-K"):
            out.risk_factors = self._extract_between(plain, "item_1a", "item_7")
            out.mda          = self._extract_between(plain, "item_7",  "item_7a", fallback_end="item_8")
            out.market_risk  = self._extract_between(plain, "item_7a", "item_8")
        elif form.startswith("10-Q"):
            # MD&A in 10-Q is Part I Item 2.
            out.mda          = self._extract_between(plain, "part1_item2", "part1_item3")
            out.market_risk  = self._extract_between(plain, "part1_item3", "part2_item1a", fallback_end="part2_item2")
            # Risk factors only present if updated (Part II Item 1A).
            out.risk_factors = self._extract_between(plain, "part2_item1a", "part2_item2")
        else:
            # 8-K and others — keep empty for now; use raw text length as
            # a smoke-test signal that something was downloaded.
            pass

        out.extracted_lengths = {
            "risk_factors": len(out.risk_factors),
            "mda": len(out.mda),
            "market_risk": len(out.market_risk),
        }
        return out

    # ------------------------------ helpers
    @staticmethod
    def _html_to_text(text: str) -> str:
        # The lxml parser handles both well-formed HTML and the slightly-broken
        # HTML inside SGML wrappers that some older filings use.
        soup = BeautifulSoup(text, "lxml")
        # Drop scripts, styles, tables (table content is mostly numeric noise
        # for our language-drift use case).
        for bad in soup(["script", "style", "table"]):
            bad.decompose()
        return soup.get_text(separator=" ")

    @staticmethod
    def _normalise_whitespace(text: str) -> str:
        # Collapse runs of whitespace; turn non-breaking spaces into normal ones.
        text = text.replace("\xa0", " ").replace("​", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _extract_between(
        text: str,
        start_key: str,
        end_key: str,
        *,
        fallback_end: Optional[str] = None,
    ) -> str:
        """Extract the substring between the FIRST start match and the
        FIRST subsequent end match. Returns "" if either is missing.
        """
        start_pat = _SECTION_PATTERNS[start_key]
        end_pat = _SECTION_PATTERNS[end_key]
        starts = list(start_pat.finditer(text))
        if not starts:
            return ""
        # Many filings have "Item 1A" referenced in the table of contents
        # before the actual section. Use the SECOND occurrence if available.
        s = starts[1].end() if len(starts) >= 2 else starts[0].end()

        # Find the first end-pattern occurrence after s.
        end_match = next((m for m in end_pat.finditer(text) if m.start() > s), None)
        if end_match is None and fallback_end is not None:
            fb_pat = _SECTION_PATTERNS[fallback_end]
            end_match = next(
                (m for m in fb_pat.finditer(text) if m.start() > s), None
            )
        if end_match is None:
            # Take to end-of-doc; will be trimmed by the consumer if too long.
            return text[s:].strip()
        return text[s : end_match.start()].strip()
