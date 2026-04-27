"""Generate per-variant architecture diagrams as SVGs.

For each of the 9 variants (A through I) emit:
  assets/variants/<letter>.svg

Run from the repo root: `python assets/generate_variant_diagrams.py`
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------- canvas
W, H = 1180, 700
OUT_DIR = Path(__file__).parent / "variants"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Brand palette — white background
BG = "#FFFFFF"
INK = "#0F172A"
INK_SUB = "#475569"
INK_FAINT = "#94A3B8"
DIVIDER = "#E2E8F0"

# Component colours
C_INPUT = "#DBEAFE"      # light blue
C_INPUT_BD = "#3B82F6"
C_TEMPORAL = "#EDE9FE"   # light violet
C_TEMPORAL_BD = "#7C3AED"
C_SLOW = "#FEF3C7"       # light amber
C_SLOW_BD = "#F59E0B"
C_FUSION = "#FCE7F3"     # light pink
C_FUSION_BD = "#DB2777"
C_GNN = "#DCFCE7"        # light green
C_GNN_BD = "#16A34A"
C_HIER = "#CFFAFE"       # light cyan
C_HIER_BD = "#0891B2"
C_RESID = "#F1F5F9"      # light grey-blue
C_RESID_BD = "#64748B"
C_HEAD_RANK = "#FEE2E2"  # light red
C_HEAD_RANK_BD = "#DC2626"
C_HEAD_AUX = "#F1F5F9"
C_HEAD_AUX_BD = "#94A3B8"

# Disabled (faded) colours
C_DISABLED = "#F8FAFC"
C_DISABLED_BD = "#CBD5E1"
INK_DISABLED = "#94A3B8"


# ---------------------------------------------------------------- variant configs
VARIANTS = [
    ("A", "Informer only (no graph)",
        {"graph": None, "edges": [], "hierarchy": False,
         "membership": "fixed", "multi_scale": False, "fundamentals": False, "sentiment": False}),
    ("B", "+ static sector graph",
        {"graph": "GAT", "edges": ["sector"], "hierarchy": False,
         "membership": "fixed", "multi_scale": False, "fundamentals": False, "sentiment": False}),
    ("C", "+ fundamentals features",
        {"graph": "GAT", "edges": ["sector"], "hierarchy": False,
         "membership": "fixed", "multi_scale": False, "fundamentals": True, "sentiment": False}),
    ("D", "+ dynamic correlation edges",
        {"graph": "GAT", "edges": ["correlation"], "hierarchy": False,
         "membership": "fixed", "multi_scale": False, "fundamentals": True, "sentiment": False}),
    ("E", "+ multi-relational R-GAT",
        {"graph": "RGAT", "edges": ["correlation", "attention", "fundamental"], "hierarchy": False,
         "membership": "fixed", "multi_scale": False, "fundamentals": True, "sentiment": False}),
    ("F", "+ sentiment features",
        {"graph": "RGAT", "edges": ["correlation", "attention", "fundamental"], "hierarchy": False,
         "membership": "fixed", "multi_scale": False, "fundamentals": True, "sentiment": True}),
    ("G", "+ dynamic membership",
        {"graph": "RGAT", "edges": ["correlation", "attention", "fundamental"], "hierarchy": False,
         "membership": "dynamic", "multi_scale": False, "fundamentals": True, "sentiment": True}),
    ("H", "+ hierarchical super-nodes",
        {"graph": "RGAT", "edges": ["correlation", "attention", "fundamental"], "hierarchy": True,
         "membership": "dynamic", "multi_scale": False, "fundamentals": True, "sentiment": True}),
    ("I", "+ multi-scale lookback (20 + 60 + 120)",
        {"graph": "RGAT", "edges": ["correlation", "attention", "fundamental"], "hierarchy": True,
         "membership": "dynamic", "multi_scale": True, "fundamentals": True, "sentiment": True}),
]


# ----------------------------------------------------------------- block helpers
def box(x, y, w, h, fill, stroke, title, sub=None, faded=False):
    fill_eff = C_DISABLED if faded else fill
    stroke_eff = C_DISABLED_BD if faded else stroke
    title_color = INK_DISABLED if faded else INK
    sub_color = INK_DISABLED if faded else INK_SUB
    parts = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" ry="10" '
        f'fill="{fill_eff}" stroke="{stroke_eff}" stroke-width="1.6"/>',
        # Left accent stripe
        f'<rect x="{x}" y="{y}" width="4" height="{h}" rx="2" ry="2" fill="{stroke_eff}"/>',
    ]
    # Title text — support multi-line via newlines
    title_lines = title.split("\n")
    for i, line in enumerate(title_lines):
        parts.append(
            f'<text x="{x + w/2}" y="{y + 22 + i*15}" text-anchor="middle" '
            f'font-family="Inter, system-ui, sans-serif" font-size="13" font-weight="700" '
            f'fill="{title_color}" letter-spacing="0.2">{line}</text>'
        )
    if sub:
        sub_y0 = y + 22 + len(title_lines) * 15 + 8  # below the last title line
        for i, line in enumerate(sub.split("\n")):
            parts.append(
                f'<text x="{x + w/2}" y="{sub_y0 + i*14}" text-anchor="middle" '
                f'font-family="Inter, system-ui, sans-serif" font-size="11" '
                f'fill="{sub_color}">{line}</text>'
            )
    return parts


def arrow(x1, y1, x2, y2, faded=False, dashed=False, label=None, label_above=True):
    color = INK_DISABLED if faded else INK_SUB
    dash = ' stroke-dasharray="4,3"' if dashed else ""
    parts = [
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="1.5" marker-end="url(#arrow)"{dash}/>'
    ]
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2 + (-5 if label_above else 14)
        parts.append(
            f'<text x="{mx}" y="{my}" text-anchor="middle" '
            f'font-family="Inter, system-ui, sans-serif" font-size="10" '
            f'fill="{INK_FAINT}" font-style="italic">{label}</text>'
        )
    return parts


def chip(x, y, label, color, active=True):
    fill = color if active else C_DISABLED
    bd = color if active else C_DISABLED_BD
    txt = "white" if active else INK_DISABLED
    w = max(48, len(label) * 7 + 16)
    return [
        f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="11" ry="11" '
        f'fill="{fill}" stroke="{bd}" stroke-width="1"/>',
        f'<text x="{x + w/2}" y="{y + 15}" text-anchor="middle" '
        f'font-family="Inter, system-ui, sans-serif" font-size="11" '
        f'font-weight="600" fill="{txt}">{label}</text>',
    ], w


# ----------------------------------------------------------------- variant render
def render(letter, title, cfg):
    parts = []
    has_gnn = cfg["graph"] is not None
    has_hier = cfg["hierarchy"]
    multi_scale = cfg["multi_scale"]
    has_fund = cfg["fundamentals"]
    has_sent = cfg["sentiment"]
    is_dyn_memb = cfg["membership"] == "dynamic"

    # ============================================== top: title and chip strip
    parts.append(
        f'<text x="40" y="42" font-family="Inter, system-ui, sans-serif" '
        f'font-size="30" font-weight="800" fill="{INK}" letter-spacing="-0.3">'
        f'Variant {letter}</text>'
    )
    parts.append(
        f'<text x="160" y="42" font-family="Inter, system-ui, sans-serif" '
        f'font-size="16" fill="{INK_SUB}" font-weight="500">{title}</text>'
    )

    # delta chips on the right side
    chip_specs = [
        ("informer", C_TEMPORAL_BD, True),
        ("multi-scale", C_TEMPORAL_BD, multi_scale),
        ("slow MLP", C_SLOW_BD, True),
        ("gated fusion", C_FUSION_BD, True),
        ("GAT", C_GNN_BD, cfg["graph"] == "GAT"),
        ("R-GAT", C_GNN_BD, cfg["graph"] == "RGAT"),
        ("hierarchy", C_HIER_BD, has_hier),
        ("dyn membership", C_HIER_BD, is_dyn_memb),
        ("fundamentals", C_SLOW_BD, has_fund),
        ("sentiment", C_FUSION_BD, has_sent),
    ]
    chip_y = 70
    chip_x = 40
    for label, color, active in chip_specs:
        chip_parts, cw = chip(chip_x, chip_y, label, color, active)
        parts.extend(chip_parts)
        chip_x += cw + 6

    # divider
    parts.append(
        f'<line x1="40" y1="110" x2="{W - 40}" y2="110" stroke="{DIVIDER}" stroke-width="1"/>'
    )

    # ============================================== column anchors (x positions)
    col1_x, col1_w = 40, 170    # input boxes
    col2_x, col2_w = 250, 200   # temporal / slow MLP
    col3_x, col3_w = 490, 160   # gated fusion
    col4_x, col4_w = 690, 200   # GNN block
    col5_x, col5_w = 930, 100   # outer residual
    col6_x, col6_w = 1060, 100  # heads + output

    y_top, y_mid, y_bot = 145, 280, 415
    box_h = 80

    # ============================================== col1: inputs
    # Fast input (top)
    parts.extend(box(col1_x, y_top, col1_w, box_h, C_INPUT, C_INPUT_BD,
                     "Fast input", "60-day window\n6 features"))
    # Slow input (bottom)
    parts.extend(box(col1_x, y_top + box_h + 18, col1_w, box_h, C_INPUT, C_INPUT_BD,
                     "Slow snapshot",
                     ("8 technicals + 4 macro" if not has_fund else
                      "8 tech + 4 macro\n+ fundamentals" if not has_sent else
                      "8 tech + 4 macro\n+ fund + sentiment")))

    # ============================================== col2: temporal / slow MLP
    if multi_scale:
        # 3 stacked Informer blocks (one per timescale). The 3-stack itself
        # visually conveys "multi-scale" — no caption needed.
        scales = ["20-day", "60-day", "120-day"]
        sub_h = 22
        gap = 3
        # Centre the 3-stack vertically inside the box_h slot.
        total_h = 3 * sub_h + 2 * gap
        ms_y = y_top + (box_h - total_h) // 2
        for i, name in enumerate(scales):
            yy = ms_y + i * (sub_h + gap)
            parts.append(
                f'<rect x="{col2_x}" y="{yy}" width="{col2_w}" height="{sub_h}" '
                f'rx="5" ry="5" fill="{C_TEMPORAL}" stroke="{C_TEMPORAL_BD}" stroke-width="1.4"/>'
            )
            parts.append(
                f'<rect x="{col2_x}" y="{yy}" width="3" height="{sub_h}" '
                f'rx="1.5" ry="1.5" fill="{C_TEMPORAL_BD}"/>'
            )
            parts.append(
                f'<text x="{col2_x + 12}" y="{yy + 15}" '
                f'font-family="Inter, system-ui, sans-serif" font-size="11" '
                f'font-weight="700" fill="{INK}">Informer · {name}</text>'
            )
    else:
        parts.extend(box(col2_x, y_top, col2_w, box_h, C_TEMPORAL, C_TEMPORAL_BD,
                         "Informer encoder",
                         "ProbSparse attn\nd_model=64 · 1 layer"))
    # Slow MLP
    parts.extend(box(col2_x, y_top + box_h + 18, col2_w, box_h, C_SLOW, C_SLOW_BD,
                     "Slow MLP",
                     "12 → 32 → 16\nGELU · dropout 0.3"))

    # ============================================== col3: gated fusion (centred)
    fusion_y = y_mid
    parts.extend(box(col3_x, fusion_y, col3_w, box_h, C_FUSION, C_FUSION_BD,
                     "Gated fusion",
                     "per-channel σ gates\nfast ⊙ + slow ⊙"))

    # ============================================== col4: GNN block (conditional)
    gnn_y = y_mid
    gnn_h = box_h + (60 if has_hier else 0)
    if has_gnn:
        gnn_name = cfg["graph"]
        # Short-form edge labels so they fit inside the GNN box width
        edge_short = {
            "sector": "sector",
            "correlation": "corr",
            "attention": "attn",
            "fundamental": "fund",
        }
        edges_short = " · ".join(edge_short.get(e, e) for e in cfg["edges"])
        sub_text = f"{gnn_name} · 2 layers\nedges: {edges_short}"
        parts.extend(box(col4_x, gnn_y, col4_w, box_h, C_GNN, C_GNN_BD,
                         f"GNN — {gnn_name}", sub_text))
        # hierarchy as additional pill below
        if has_hier:
            hy = gnn_y + box_h + 8
            parts.extend(box(col4_x, hy, col4_w, 52, C_HIER, C_HIER_BD,
                             "+ hierarchy",
                             "stock ↔ sector(11) ↔ market(1)"))
    else:
        # disabled GNN (variant A) — pass-through indicator
        parts.extend(box(col4_x, gnn_y, col4_w, box_h, C_GNN, C_GNN_BD,
                         "GNN (disabled)", "pass-through\nno cross-stock mixing", faded=True))

    # ============================================== col5: outer residual
    parts.extend(box(col5_x, y_mid, col5_w, box_h, C_RESID, C_RESID_BD,
                     "Outer\nresidual", "gated mix\npre + post"))

    # ============================================== col6: heads (3 stacked)
    head_w = col6_w
    head_h = 26
    head_gap = 6
    base_y = y_mid + 4
    head_specs = [
        ("ranking", C_HEAD_RANK, C_HEAD_RANK_BD, "used"),
        ("return", C_HEAD_AUX, C_HEAD_AUX_BD, "aux=0"),
        ("vol", C_HEAD_AUX, C_HEAD_AUX_BD, "aux=0"),
    ]
    for i, (name, fill, bd, badge) in enumerate(head_specs):
        yy = base_y + i * (head_h + head_gap)
        parts.append(
            f'<rect x="{col6_x}" y="{yy}" width="{head_w}" height="{head_h}" '
            f'rx="6" ry="6" fill="{fill}" stroke="{bd}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<text x="{col6_x + 8}" y="{yy + 17}" '
            f'font-family="Inter, system-ui, sans-serif" font-size="11.5" '
            f'font-weight="700" fill="{INK}">{name}</text>'
        )
        parts.append(
            f'<text x="{col6_x + head_w - 8}" y="{yy + 17}" text-anchor="end" '
            f'font-family="Inter, system-ui, sans-serif" font-size="9.5" '
            f'fill="{INK_SUB}" font-style="italic">{badge}</text>'
        )

    # ============================================== arrows / connectors
    # input → temporal  (top row)
    parts.extend(arrow(col1_x + col1_w + 4, y_top + box_h / 2,
                       col2_x - 4, y_top + box_h / 2))
    # input slow → slow MLP (bottom row)
    parts.extend(arrow(col1_x + col1_w + 4, y_top + box_h + 18 + box_h / 2,
                       col2_x - 4, y_top + box_h + 18 + box_h / 2))
    # temporal → gated fusion (top down to centre-left)
    parts.extend(arrow(col2_x + col2_w + 4, y_top + box_h / 2,
                       col3_x - 4, y_mid + box_h / 2 - 14))
    # slow MLP → gated fusion (bottom up to centre-left)
    parts.extend(arrow(col2_x + col2_w + 4, y_top + box_h + 18 + box_h / 2,
                       col3_x - 4, y_mid + box_h / 2 + 14))
    # gated fusion → GNN
    parts.extend(arrow(col3_x + col3_w + 4, y_mid + box_h / 2,
                       col4_x - 4, y_mid + box_h / 2,
                       faded=not has_gnn))
    # GNN → outer residual
    parts.extend(arrow(col4_x + col4_w + 4, y_mid + box_h / 2,
                       col5_x - 4, y_mid + box_h / 2))
    # ALSO: pre-GNN bypass (skip connection)
    parts.append(
        f'<path d="M {col3_x + col3_w + 4} {y_mid + 8} '
        f'Q {(col3_x + col5_x) / 2} {y_mid - 30} {col5_x - 4} {y_mid + 8}" '
        f'fill="none" stroke="{C_RESID_BD}" stroke-width="1.2" '
        f'stroke-dasharray="4,3" marker-end="url(#arrow)"/>'
    )
    parts.append(
        f'<text x="{(col3_x + col5_x) / 2}" y="{y_mid - 33}" text-anchor="middle" '
        f'font-family="Inter, system-ui, sans-serif" font-size="10" '
        f'fill="{INK_SUB}" font-style="italic">skip (pre-GNN)</text>'
    )
    # outer residual → heads
    parts.extend(arrow(col5_x + col5_w + 4, y_mid + box_h / 2,
                       col6_x - 4, y_mid + box_h / 2))

    # ============================================== bottom: footnote / output
    # Place footer well below the deepest content (which is the hierarchy
    # block at y ≈ 420 for variants H, I).
    foot_y = 510
    parts.append(
        f'<line x1="40" y1="{foot_y - 18}" x2="{W - 40}" y2="{foot_y - 18}" '
        f'stroke="{DIVIDER}" stroke-width="1"/>'
    )
    # Per-variant signature — short labels so it fits in one line
    edge_short_map = {"sector": "sector", "correlation": "corr",
                      "attention": "attn", "fundamental": "fund"}
    feature_summary = []
    feature_summary.append("temporal = Informer×3 (multi-scale)" if multi_scale else "temporal = Informer")
    if has_gnn:
        edges_compact = "+".join(edge_short_map.get(e, e) for e in cfg["edges"])
        feature_summary.append(f"GNN = {cfg['graph']}[{edges_compact}]")
    else:
        feature_summary.append("GNN = disabled")
    if has_hier:
        feature_summary.append("hierarchy = sector + market")
    feature_summary.append(f"membership = {cfg['membership']}")
    feature_text = "  ·  ".join(feature_summary)
    parts.append(
        f'<text x="{W/2}" y="{foot_y}" text-anchor="middle" '
        f'font-family="ui-monospace, SFMono-Regular, monospace" font-size="11" '
        f'fill="{INK}">{feature_text}</text>'
    )
    # Output line
    parts.append(
        f'<text x="{W/2}" y="{foot_y + 22}" text-anchor="middle" '
        f'font-family="Inter, system-ui, sans-serif" font-size="12" '
        f'fill="{INK_SUB}" font-style="italic">→ daily score per stock → top-50 long / bottom-50 short</text>'
    )

    # variant progression strip (mini A-I dots) at very bottom
    strip_y = H - 30
    n = len(VARIANTS)
    span = 320
    sx0 = W - 40 - span
    parts.append(
        f'<text x="{sx0 - 6}" y="{strip_y + 4}" text-anchor="end" '
        f'font-family="Inter, system-ui, sans-serif" font-size="10" '
        f'fill="{INK_FAINT}">A → I:</text>'
    )
    for i, (lt, _t, _c) in enumerate(VARIANTS):
        cx = sx0 + (i + 0.5) * (span / n)
        active = lt == letter
        rad = 8 if active else 5
        fill = C_TEMPORAL_BD if active else INK_FAINT
        parts.append(
            f'<circle cx="{cx}" cy="{strip_y}" r="{rad}" fill="{fill}"/>'
        )
        parts.append(
            f'<text x="{cx}" y="{strip_y + 22}" text-anchor="middle" '
            f'font-family="Inter, system-ui, sans-serif" font-size="9" '
            f'font-weight="{700 if active else 500}" '
            f'fill="{INK if active else INK_FAINT}">{lt}</text>'
        )

    # ============================================== assemble
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
        f'width="{W}" height="{H}" role="img" '
        f'aria-label="constellation-quant variant {letter}: {title}">\n'
        '  <defs>\n'
        '    <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" '
        'orient="auto" markerUnits="strokeWidth">\n'
        f'      <path d="M0,0 L0,6 L8,3 z" fill="{INK_SUB}"/>\n'
        '    </marker>\n'
        '  </defs>\n'
        f'  <rect width="100%" height="100%" fill="{BG}"/>\n  '
        + "\n  ".join(parts)
        + "\n</svg>\n"
    )
    return svg


# ----------------------------------------------------------------- main
def main():
    for letter, title, cfg in VARIANTS:
        svg = render(letter, title, cfg)
        out = OUT_DIR / f"{letter}.svg"
        out.write_text(svg, encoding="utf-8")
        print(f"  wrote {out}  ({len(svg):,} bytes)")


if __name__ == "__main__":
    main()
