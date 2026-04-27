"""Report generation and visualizations (HTML/PDF)."""

from constellation_quant.outputs.report_builder import (
    ReportBuilder,
    VariantRun,
    comparison_table,
    executive_summary,
    html_to_pdf,
    load_variant_runs,
    regime_breakdown,
    significance_vs_previous,
)
from constellation_quant.outputs.visualizations import (
    ablation_bar_chart,
    drawdown_overlay,
    equity_curves_overlay,
    monthly_return_heatmap,
    rolling_ic_timeseries,
)

__all__ = [
    "ReportBuilder",
    "VariantRun",
    "load_variant_runs",
    "comparison_table",
    "significance_vs_previous",
    "executive_summary",
    "regime_breakdown",
    "html_to_pdf",
    "ablation_bar_chart",
    "drawdown_overlay",
    "equity_curves_overlay",
    "monthly_return_heatmap",
    "rolling_ic_timeseries",
]
