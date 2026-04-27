"""Generate the constellation-quant infographic SVG.

Run from the repo root: `python assets/generate_logo.py`
Output: assets/constellation-quant.svg
"""
from __future__ import annotations

import math
import random
from pathlib import Path

# --------------------------------------------------------------------- canvas
W, H = 1200, 630
OUT = Path(__file__).with_name("constellation-quant.svg")

# brand palette (white background)
BG = "#FFFFFF"
INK = "#0F172A"          # near-black for headings
INK_SUB = "#475569"      # body grey
STAR_DIM = "#94A3B8"     # generic stock dot
STAR_LIT = "#1E40AF"     # highlighted stock (deep blue)
ACCENT = "#F59E0B"       # amber (alpha)
EDGE = "#CBD5E1"         # constellation line dim
EDGE_HOT = "#1E40AF"     # constellation line hot
LONG = "#16A34A"         # green (top-50 long)
SHORT = "#DC2626"        # red (bottom-50 short)

random.seed(7)


# --------------------------------------------------------------------- panel 1: stars
def starfield():
    """Golden-angle spiral inside a circle. Returns list of (x, y, r, lit, ticker)."""
    cx, cy, r_max = 230, 320, 165
    n = 503  # actual S&P 500 securities count
    stars = []
    highlights = {
        # i: ticker — handpicked for visual spread
        7: "AAPL", 23: "MSFT", 41: "NVDA", 67: "GOOGL",
        97: "AMZN", 131: "META", 173: "BRK.B", 211: "TSLA",
        257: "JPM", 311: "V",
    }
    for i in range(n):
        # Fibonacci spiral: r = sqrt(i) * scale, theta steps by golden angle
        r = math.sqrt(i / n) * r_max
        theta = i * 137.508 * math.pi / 180
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        if (x - cx) ** 2 + (y - cy) ** 2 > r_max * r_max:
            continue
        lit = i in highlights
        rad = 5.5 if lit else random.choice([1.2, 1.4, 1.6, 1.8, 2.0])
        ticker = highlights.get(i)
        stars.append((x, y, rad, lit, ticker))
    return stars


def render_panel_stars():
    stars = starfield()
    parts = []
    # constellation lines — connect each highlighted star to its 1-2 nearest highlighted neighbours
    lit = [(x, y, t) for x, y, _, l, t in stars if l]
    for i, (x1, y1, t1) in enumerate(lit):
        # find closest 2 other highlighted
        dists = sorted(
            ((j, (x1 - x2) ** 2 + (y1 - y2) ** 2) for j, (x2, y2, _) in enumerate(lit) if j != i),
            key=lambda p: p[1],
        )[:2]
        for j, _ in dists:
            x2, y2, _ = lit[j]
            parts.append(
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{EDGE_HOT}" stroke-width="0.7" stroke-opacity="0.55"/>'
            )
    # also some dim background edges between random nearby stars
    dim_pool = [(x, y) for x, y, _, l, _ in stars if not l]
    for _ in range(35):
        a = random.choice(dim_pool)
        b = random.choice(dim_pool)
        d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
        if 200 < d2 < 2200:
            parts.append(
                f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                f'stroke="{EDGE}" stroke-width="0.5" stroke-opacity="0.4"/>'
            )
    # stars
    for x, y, r, lit_, ticker in stars:
        fill = STAR_LIT if lit_ else STAR_DIM
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}"/>')
        if lit_:
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r + 4}" fill="none" '
                f'stroke="{STAR_LIT}" stroke-width="0.8" stroke-opacity="0.35"/>'
            )
            # ticker label, offset
            parts.append(
                f'<text x="{x + 9:.1f}" y="{y + 3:.1f}" font-family="ui-monospace, SFMono-Regular, monospace" '
                f'font-size="10" fill="{INK}" font-weight="500">{ticker}</text>'
            )
    return "\n".join(parts)


# --------------------------------------------------------------------- panel 2: network
def render_panel_network():
    """Stylised stack: temporal encoder → GNN → heads."""
    cx = 600
    box_w, box_h, gap = 240, 56, 18
    top = 220
    blocks = [
        ("INFORMER", "temporal encoder · L=60", "#1E40AF"),
        ("R-GAT + HIERARCHY", "cross-stock graph · 503 nodes", "#7C3AED"),
        ("MULTI-TASK HEADS", "rank · return · vol", "#F59E0B"),
    ]
    parts = []
    # incoming arrow from panel 1
    parts.append(
        f'<path d="M 420 320 Q 460 320 480 320" stroke="{INK_SUB}" stroke-width="1.5" '
        f'fill="none" marker-end="url(#arrow)"/>'
    )
    for i, (title, sub, color) in enumerate(blocks):
        y = top + i * (box_h + gap)
        parts.append(
            f'<rect x="{cx - box_w / 2}" y="{y}" width="{box_w}" height="{box_h}" '
            f'rx="8" ry="8" fill="white" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<rect x="{cx - box_w / 2}" y="{y}" width="4" height="{box_h}" '
            f'rx="2" ry="2" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{cx - box_w / 2 + 16}" y="{y + 22}" font-family="Inter, system-ui, sans-serif" '
            f'font-size="12.5" font-weight="700" fill="{INK}" letter-spacing="0.3">{title}</text>'
        )
        parts.append(
            f'<text x="{cx - box_w / 2 + 16}" y="{y + 40}" font-family="Inter, system-ui, sans-serif" '
            f'font-size="11" fill="{INK_SUB}">{sub}</text>'
        )
        # vertical connector to next block
        if i < len(blocks) - 1:
            y_end = y + box_h
            y_next = top + (i + 1) * (box_h + gap)
            parts.append(
                f'<line x1="{cx}" y1="{y_end}" x2="{cx}" y2="{y_next}" '
                f'stroke="{INK_SUB}" stroke-width="1.2" stroke-dasharray="3,3"/>'
            )
    # outgoing arrow to panel 3
    parts.append(
        f'<path d="M 720 320 Q 760 320 780 320" stroke="{INK_SUB}" stroke-width="1.5" '
        f'fill="none" marker-end="url(#arrow)"/>'
    )
    return "\n".join(parts)


# --------------------------------------------------------------------- panel 3: ranked output
def render_panel_output():
    cx = 970
    parts = []
    # heading
    parts.append(
        f'<text x="{cx}" y="195" text-anchor="middle" font-family="Inter, system-ui, sans-serif" '
        f'font-size="13" font-weight="700" fill="{INK}" letter-spacing="0.5">CROSS-SECTIONAL RANK</text>'
    )
    parts.append(
        f'<text x="{cx}" y="212" text-anchor="middle" font-family="Inter, system-ui, sans-serif" '
        f'font-size="11" fill="{INK_SUB}">5-day forward return</text>'
    )

    # top picks (long)
    longs = ["NVDA", "META", "AVGO", "AMZN", "AAPL"]
    shorts = ["INTC", "F", "PFE", "BA", "WBA"]
    y0 = 240
    parts.append(
        f'<text x="{cx - 70}" y="{y0}" font-family="Inter, system-ui, sans-serif" '
        f'font-size="11" font-weight="700" fill="{LONG}" letter-spacing="0.4">▲ TOP 50 (LONG)</text>'
    )
    for i, t in enumerate(longs):
        parts.append(
            f'<text x="{cx - 70}" y="{y0 + 18 + i * 16}" font-family="ui-monospace, SFMono-Regular, monospace" '
            f'font-size="11" fill="{INK}">{i + 1:>2}. {t}</text>'
        )

    y1 = 360
    parts.append(
        f'<text x="{cx - 70}" y="{y1}" font-family="Inter, system-ui, sans-serif" '
        f'font-size="11" font-weight="700" fill="{SHORT}" letter-spacing="0.4">▼ BOTTOM 50 (SHORT)</text>'
    )
    for i, t in enumerate(shorts):
        parts.append(
            f'<text x="{cx - 70}" y="{y1 + 18 + i * 16}" font-family="ui-monospace, SFMono-Regular, monospace" '
            f'font-size="11" fill="{INK}">{i + 449 + 1:>3}. {t}</text>'
        )
    return "\n".join(parts)


# --------------------------------------------------------------------- compose
def build_svg() -> str:
    title = "constellation-quant"
    subtitle = "graph + temporal deep learning for cross-sectional equity ranking"
    metrics = (
        "503 securities · S&amp;P 500 · 1990–2024  |  "
        "best val IC 0.0284  ·  IR 0.213  ·  hit@50 60.9%"
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="constellation-quant infographic">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3"
            orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L8,3 z" fill="{INK_SUB}"/>
    </marker>
  </defs>

  <rect width="100%" height="100%" fill="{BG}"/>

  <!-- title bar -->
  <text x="60" y="80" font-family="Inter, system-ui, sans-serif" font-size="38"
        font-weight="800" fill="{INK}" letter-spacing="-0.5">{title}</text>
  <text x="60" y="112" font-family="Inter, system-ui, sans-serif" font-size="15"
        fill="{INK_SUB}">{subtitle}</text>

  <!-- subtle separator -->
  <line x1="60" y1="140" x2="{W - 60}" y2="140" stroke="{EDGE}" stroke-width="1"/>

  <!-- panel labels -->
  <text x="230" y="172" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
        font-size="13" font-weight="700" fill="{INK}" letter-spacing="0.5">UNIVERSE</text>
  <text x="230" y="190" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
        font-size="11" fill="{INK_SUB}">503 stocks · 60-day windows</text>
  <text x="600" y="172" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
        font-size="13" font-weight="700" fill="{INK}" letter-spacing="0.5">DEEP LEARNING</text>
  <text x="600" y="190" text-anchor="middle" font-family="Inter, system-ui, sans-serif"
        font-size="11" fill="{INK_SUB}">temporal · graph · multi-task</text>

  <!-- panel 1: stars -->
  {render_panel_stars()}

  <!-- panel 2: network -->
  {render_panel_network()}

  <!-- panel 3: rankings -->
  {render_panel_output()}

  <!-- footer -->
  <line x1="60" y1="555" x2="{W - 60}" y2="555" stroke="{EDGE}" stroke-width="1"/>
  <text x="60" y="582" font-family="ui-monospace, SFMono-Regular, monospace" font-size="11"
        fill="{INK_SUB}">{metrics}</text>
  <text x="{W - 60}" y="582" text-anchor="end" font-family="Inter, system-ui, sans-serif"
        font-size="11" font-weight="600" fill="{INK}">github.com/your-handle/constellation-quant</text>
</svg>
"""


if __name__ == "__main__":
    svg = build_svg()
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT}  ({len(svg):,} bytes)")
