"""Turn per-variant backtest summaries into a single HTML / PDF report.

Reads `outputs/ablation/summaries/<variant>.json` (one file per variant —
produced by `scripts/evaluate.py`), optional `daily_returns.csv` and
`equity_curve.csv` for charts, then emits:

    outputs/final_report/
        ├── report.html            ← primary deliverable
        ├── report.pdf             ← optional (if reportlab available)
        ├── plots/
        │   ├── sharpe_bar.png
        │   ├── annual_return_bar.png
        │   ├── equity_overlay.png
        │   └── drawdown_overlay.png
        └── assets/                ← shared CSS, etc.

Significance tests run between each variant and the previous one in the
ablation order (A vs B, B vs C, ...), so readers can see at a glance which
component added genuine signal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import json

import numpy as np
import pandas as pd

from constellation_quant.evaluation import (
    bootstrap_sharpe_diff,
    paired_t_test_ic,
)
from constellation_quant.outputs.visualizations import (
    ablation_bar_chart,
    drawdown_overlay,
    equity_curves_overlay,
)
from constellation_quant.utils import get_logger

log = get_logger(__name__)


# ── Data containers ───────────────────────────────────────────────────────


@dataclass
class VariantRun:
    """One variant's artefacts after training + evaluation."""
    name:          str
    summary:       Dict[str, Any]         # parsed summary.json['backtest']
    regimes:       Dict[str, Any]
    daily_returns: Optional[pd.Series] = None
    equity:        Optional[pd.Series] = None
    drawdown:      Optional[pd.Series] = None


# ── Loading ────────────────────────────────────────────────────────────────


def load_variant_runs(summaries_dir: Path) -> Dict[str, VariantRun]:
    """Read every `<variant>.json` + sibling CSVs into VariantRun objects."""
    summaries_dir = Path(summaries_dir)
    if not summaries_dir.exists():
        return {}

    runs: Dict[str, VariantRun] = {}
    for summary_path in sorted(summaries_dir.glob("*.json")):
        name = summary_path.stem
        try:
            payload = json.loads(summary_path.read_text())
        except json.JSONDecodeError as exc:
            log.warning("Skipping {}: {}", summary_path, exc)
            continue
        daily = _read_series(summary_path.with_name(f"{name}_daily_returns.csv"))
        equity = _read_series(summary_path.with_name(f"{name}_equity_curve.csv"))
        drawdown = _read_series(summary_path.with_name(f"{name}_drawdown.csv"))
        runs[name] = VariantRun(
            name=name,
            summary=payload.get("backtest", {}),
            regimes=payload.get("regimes", {}),
            daily_returns=daily,
            equity=equity,
            drawdown=drawdown,
        )
    return runs


def _read_series(path: Path) -> Optional[pd.Series]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[:, 0]


# ── Comparison table ───────────────────────────────────────────────────────


def comparison_table(runs: Mapping[str, VariantRun]) -> pd.DataFrame:
    """One-row-per-variant summary: Sharpe, annual return, drawdown, turnover."""
    rows = []
    for name, run in runs.items():
        s = run.summary
        rows.append({
            "variant":       name,
            "sharpe":        s.get("sharpe", float("nan")),
            "annual_return": s.get("annual_return", float("nan")),
            "annual_vol":    s.get("annual_vol", float("nan")),
            "max_drawdown":  s.get("max_drawdown", float("nan")),
            "avg_turnover":  s.get("avg_turnover", float("nan")),
            "total_cost":    s.get("total_cost", float("nan")),
            "final_equity":  s.get("final_equity", float("nan")),
            "n_days":        s.get("n_days", float("nan")),
        })
    return pd.DataFrame(rows)


def significance_vs_previous(runs: Mapping[str, VariantRun]) -> List[Dict[str, Any]]:
    """Paired t-test + Sharpe bootstrap between consecutive variants.

    Uses daily_returns when available; otherwise produces NaN rows.
    Order follows `runs.keys()` — for the main sweep this is the canonical
    A..I chain so the table shows "what did each incremental component buy?".
    """
    names = list(runs.keys())
    rows: List[Dict[str, Any]] = []
    for i in range(1, len(names)):
        prev = runs[names[i - 1]]
        curr = runs[names[i]]
        row: Dict[str, Any] = {"comparison": f"{curr.name} vs {prev.name}"}
        if prev.daily_returns is None or curr.daily_returns is None:
            row.update({"p_value_ic": float("nan"),
                        "sharpe_diff": float("nan"),
                        "ci_low": float("nan"), "ci_high": float("nan")})
        else:
            common = prev.daily_returns.index.intersection(curr.daily_returns.index)
            a = curr.daily_returns.loc[common].to_numpy()
            b = prev.daily_returns.loc[common].to_numpy()
            ic_test = paired_t_test_ic(a, b)
            boot = bootstrap_sharpe_diff(a, b, n_bootstrap=500, block_size=5, seed=0)
            row.update({
                "p_value_ic": ic_test.pvalue,
                "sharpe_diff": boot.statistic,
                "ci_low":  boot.ci_low,
                "ci_high": boot.ci_high,
            })
        rows.append(row)
    return rows


# ── Report builder ─────────────────────────────────────────────────────────


class ReportBuilder:
    """Write comparison table + plots + narrative HTML.

    Args:
        runs: Output of `load_variant_runs(...)`.
        out_dir: Report root. Plots land under `out_dir/plots/`.
        template_path: Jinja2 HTML template. Default = packaged
            `constellation_quant/outputs/templates/report_template.html`.
    """

    PACKAGE_TEMPLATES = Path(__file__).with_suffix("").parent / "templates"

    def __init__(
        self,
        runs: Mapping[str, VariantRun],
        out_dir: Path,
        template_path: Optional[Path] = None,
        config_dump: Optional[Mapping[str, Any]] = None,
    ):
        self.runs = dict(runs)
        self.out_dir = Path(out_dir)
        self.template_path = Path(template_path) if template_path else None
        self.config_dump = dict(config_dump or {})

    def build(self, write_pdf: bool = False) -> Path:
        """Produce the HTML (and optional PDF) report. Returns the HTML path."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = self.out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        table = comparison_table(self.runs)
        sig_rows = significance_vs_previous(self.runs)
        exec_summary = executive_summary(self.runs, table)
        regime_table = regime_breakdown(self.runs)

        plot_paths = self._render_plots(table, plots_dir)
        html = self._render_html(
            table, sig_rows, plot_paths,
            exec_summary=exec_summary,
            regime_table_html=_df_to_html(regime_table),
            config_dump_json=json.dumps(self.config_dump, indent=2, default=str)
                if self.config_dump else "",
        )

        out_path = self.out_dir / "report.html"
        out_path.write_text(html)
        log.info("Report written to {}", out_path)

        if write_pdf:
            pdf_path = self.out_dir / "report.pdf"
            html_to_pdf(html, pdf_path)
        return out_path

    # ── Plots ──────────────────────────────────────────────────────────

    def _render_plots(self, table: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        if not table.empty:
            paths["sharpe"] = out_dir / "sharpe_bar.png"
            ablation_bar_chart(table, metric="sharpe", save_to=paths["sharpe"])
            paths["annual_return"] = out_dir / "annual_return_bar.png"
            ablation_bar_chart(table, metric="annual_return",
                                save_to=paths["annual_return"])

        # Equity + drawdown overlays need the daily series.
        equities = {n: r.equity for n, r in self.runs.items() if r.equity is not None}
        if equities:
            paths["equity"] = out_dir / "equity_overlay.png"
            equity_curves_overlay(equities, save_to=paths["equity"])
        drawdowns = {n: r.drawdown for n, r in self.runs.items() if r.drawdown is not None}
        if drawdowns:
            paths["drawdown"] = out_dir / "drawdown_overlay.png"
            drawdown_overlay(drawdowns, save_to=paths["drawdown"])
        return paths

    # ── HTML ───────────────────────────────────────────────────────────

    def _render_html(
        self,
        table: pd.DataFrame,
        sig_rows: List[Dict[str, Any]],
        plot_paths: Mapping[str, Path],
        exec_summary: Optional[Dict[str, Any]] = None,
        regime_table_html: str = "",
        config_dump_json: str = "",
    ) -> str:
        try:
            from jinja2 import Environment, FileSystemLoader, select_autoescape
            template_dir = self.template_path.parent if self.template_path else self.PACKAGE_TEMPLATES
            template_name = self.template_path.name if self.template_path else "report_template.html"
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html"]),
            )
            tpl = env.get_template(template_name)
            return tpl.render(
                table_html=_df_to_html(table),
                significance_rows=sig_rows,
                plots={k: _relative_plot(v, self.out_dir) for k, v in plot_paths.items()},
                variants=list(self.runs.keys()),
                exec_summary=exec_summary or {},
                regime_table_html=regime_table_html,
                config_dump_json=config_dump_json,
            )
        except (ImportError, FileNotFoundError) as exc:
            log.warning("Falling back to simple HTML: {}", exc)
            return self._fallback_html(table, sig_rows, plot_paths)

    def _fallback_html(
        self,
        table: pd.DataFrame,
        sig_rows: List[Dict[str, Any]],
        plot_paths: Mapping[str, Path],
    ) -> str:
        html = ["<!DOCTYPE html>", "<html><head><title>constellation-quant Ablation Report</title>",
                "<style>body{font-family:system-ui;max-width:1000px;margin:2em auto;padding:0 1em;}",
                "table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ccc;padding:6px;}",
                "th{background:#f3f3f3;text-align:left;}img{max-width:100%;margin:1em 0;}</style>",
                "</head><body>",
                "<h1>constellation-quant Ablation Report</h1>",
                "<h2>Variant comparison</h2>",
                _df_to_html(table),
                "<h2>Significance tests (vs previous variant)</h2>",
                _df_to_html(pd.DataFrame(sig_rows))]
        for name, path in plot_paths.items():
            html.append(f"<h2>{name}</h2><img src='{_relative_plot(path, self.out_dir)}'/>")
        html.append("</body></html>")
        return "\n".join(html)


# ── Enhanced sections ──────────────────────────────────────────────────────


def executive_summary(
    runs: Mapping[str, VariantRun],
    table: pd.DataFrame,
) -> Dict[str, Any]:
    """Top-line metrics for the report's opening card.

    Picks the best variant by Sharpe, reports its full stats, and computes
    the lift over the first variant in the sweep (typically Model A, the
    baseline).
    """
    if table.empty:
        return {}

    valid = table.dropna(subset=["sharpe"])
    if valid.empty:
        return {"best_variant": "—", "n_variants": len(runs)}

    best_row = valid.loc[valid["sharpe"].idxmax()]
    out: Dict[str, Any] = {
        "n_variants":   int(len(runs)),
        "best_variant": str(best_row["variant"]),
        "best_sharpe":  float(best_row["sharpe"]),
        "best_annual_return": float(best_row["annual_return"]),
        "best_max_drawdown":  float(best_row["max_drawdown"]),
        "best_avg_turnover":  float(best_row["avg_turnover"]),
    }
    first_variant = table.iloc[0]
    if first_variant["variant"] != best_row["variant"]:
        out["baseline_variant"] = str(first_variant["variant"])
        out["sharpe_lift"] = float(best_row["sharpe"] - first_variant["sharpe"])
        out["annual_return_lift"] = float(
            best_row["annual_return"] - first_variant["annual_return"]
        )
    return out


def regime_breakdown(runs: Mapping[str, VariantRun]) -> pd.DataFrame:
    """One row per (variant, regime). Pulls from each run's regimes dict."""
    rows: List[Dict[str, Any]] = []
    for name, run in runs.items():
        for regime, stats in (run.regimes or {}).items():
            if not isinstance(stats, Mapping):
                continue
            rows.append({
                "variant":       name,
                "regime":        regime,
                "n_days":        stats.get("n_days", float("nan")),
                "annual_return": stats.get("annual_return", float("nan")),
                "sharpe":        stats.get("sharpe", float("nan")),
                "max_drawdown":  stats.get("max_drawdown", float("nan")),
            })
    if not rows:
        return pd.DataFrame(columns=["variant", "regime", "n_days",
                                       "annual_return", "sharpe", "max_drawdown"])
    return pd.DataFrame(rows)


def html_to_pdf(html: str, pdf_path: Path) -> Optional[Path]:
    """Render HTML to PDF via WeasyPrint if available.

    Returns the PDF path on success, None when WeasyPrint isn't installed
    (prints a warning rather than crashing — the HTML report is the
    primary deliverable; PDF is a bonus).
    """
    try:
        from weasyprint import HTML
    except ImportError:
        log.warning("WeasyPrint not installed — skipping PDF export. "
                    "Install with: pip install weasyprint")
        return None
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html, base_url=str(pdf_path.parent)).write_pdf(str(pdf_path))
    log.info("PDF written to {}", pdf_path)
    return pdf_path


# ── Helpers ────────────────────────────────────────────────────────────────


def _df_to_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>no data</em></p>"
    return df.to_html(index=False, float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "—",
                        classes="data")


def _relative_plot(path: Path, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(path)
