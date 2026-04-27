"""Phase 6 tests: enhanced report — exec summary, regime breakdown, PDF hook."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from constellation_quant.outputs import (
    ReportBuilder,
    comparison_table,
    executive_summary,
    html_to_pdf,
    load_variant_runs,
    regime_breakdown,
)


def _write_fake_summaries(
    summaries_dir: Path,
    sharpe_by_variant: Dict[str, float],
    regime_presence: bool = True,
) -> None:
    """Write per-variant summary.json + daily CSVs into `summaries_dir`."""
    summaries_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(0)
    for v, sharpe in sharpe_by_variant.items():
        daily_mean = sharpe / np.sqrt(252)
        rets = rng.normal(daily_mean * 0.01, 0.01, size=len(dates))
        equity = pd.Series((1.0 + rets).cumprod(), index=dates)
        drawdown = (equity - equity.cummax()) / equity.cummax()

        pd.Series(rets, index=dates).to_csv(
            summaries_dir / f"{v}_daily_returns.csv", header=["return"],
        )
        equity.to_csv(summaries_dir / f"{v}_equity_curve.csv", header=["equity"])
        drawdown.to_csv(summaries_dir / f"{v}_drawdown.csv", header=["drawdown"])

        regimes = {}
        if regime_presence:
            regimes = {
                "bull":     {"n_days": 30, "annual_return": 0.15, "annual_vol": 0.12,
                              "sharpe": sharpe + 0.2, "max_drawdown": -0.05, "hit_rate": 0.6},
                "bear":     {"n_days": 20, "annual_return": -0.05, "annual_vol": 0.18,
                              "sharpe": sharpe - 0.3, "max_drawdown": -0.12, "hit_rate": 0.45},
                "high_vol": {"n_days": 15, "annual_return": 0.02, "annual_vol": 0.25,
                              "sharpe": sharpe - 0.1, "max_drawdown": -0.08, "hit_rate": 0.5},
                "all":      {"n_days": 60, "annual_return": 0.1, "annual_vol": 0.15,
                              "sharpe": sharpe, "max_drawdown": float(drawdown.min()),
                              "hit_rate": 0.55},
            }

        (summaries_dir / f"{v}.json").write_text(json.dumps({
            "backtest": {
                "sharpe": sharpe, "annual_return": 0.1, "annual_vol": 0.15,
                "max_drawdown": float(drawdown.min()), "avg_turnover": 0.4,
                "total_cost": 0.01, "final_equity": float(equity.iloc[-1]),
                "n_days": len(rets),
            },
            "regimes": regimes,
        }, indent=2))


# ── Executive summary ──────────────────────────────────────────────────────


def test_executive_summary_picks_best_variant(tmp_path):
    _write_fake_summaries(tmp_path, {"A": 0.3, "B": 0.9, "C": 0.5})
    runs = load_variant_runs(tmp_path)
    table = comparison_table(runs)
    summary = executive_summary(runs, table)
    assert summary["best_variant"] == "B"
    assert summary["best_sharpe"] == pytest.approx(0.9)
    assert summary["n_variants"] == 3
    # Lift computed vs the first variant in the table (A here).
    assert "sharpe_lift" in summary


def test_executive_summary_skips_lift_when_baseline_is_best(tmp_path):
    _write_fake_summaries(tmp_path, {"A": 1.5, "B": 0.9})
    runs = load_variant_runs(tmp_path)
    table = comparison_table(runs)
    summary = executive_summary(runs, table)
    assert summary["best_variant"] == "A"
    # A is the first row and the best → no lift field.
    assert "sharpe_lift" not in summary


def test_executive_summary_empty_table():
    assert executive_summary({}, pd.DataFrame()) == {}


# ── Regime breakdown ───────────────────────────────────────────────────────


def test_regime_breakdown_flattens_per_variant(tmp_path):
    _write_fake_summaries(tmp_path, {"A": 0.4, "B": 0.6})
    runs = load_variant_runs(tmp_path)
    df = regime_breakdown(runs)
    # 2 variants × 4 regimes = 8 rows
    assert len(df) == 8
    assert set(df["regime"]) == {"bull", "bear", "high_vol", "all"}
    assert set(df["variant"]) == {"A", "B"}


def test_regime_breakdown_skips_missing_regimes(tmp_path):
    _write_fake_summaries(tmp_path, {"A": 0.4}, regime_presence=False)
    runs = load_variant_runs(tmp_path)
    df = regime_breakdown(runs)
    assert df.empty


# ── HTML embeds the new sections ───────────────────────────────────────────


def test_report_html_contains_executive_summary(tmp_path):
    _write_fake_summaries(tmp_path / "summaries", {"A": 0.3, "B": 0.9, "C": 0.5})
    runs = load_variant_runs(tmp_path / "summaries")
    builder = ReportBuilder(runs=runs, out_dir=tmp_path / "report",
                              config_dump={"lookback": 60, "horizon": 5})
    path = builder.build()
    html = path.read_text()
    assert "Executive summary" in html or "Best variant" in html
    assert "B" in html                           # winning variant shown
    # Config dump embedded (jinja2 escaping turns < > but keeps content):
    assert "lookback" in html
    assert "horizon" in html


def test_report_html_omits_summary_when_no_data(tmp_path):
    builder = ReportBuilder(runs={}, out_dir=tmp_path)
    path = builder.build()
    html = path.read_text()
    # Fallback path still emits an HTML shell.
    assert "<html" in html or "<!DOCTYPE" in html


# ── PDF export hook ────────────────────────────────────────────────────────


def test_html_to_pdf_returns_none_without_weasyprint(tmp_path):
    """When WeasyPrint isn't installed, html_to_pdf returns None and warns —
    it does NOT crash or fail the report build."""
    import constellation_quant.outputs.report_builder as rb

    # Skip if the user happens to have weasyprint installed — the test is
    # about the graceful fallback path.
    try:
        import weasyprint                        # noqa: F401
        pytest.skip("weasyprint is installed — fallback path not exercised.")
    except ImportError:
        pass

    out = html_to_pdf("<html><body>Test</body></html>", tmp_path / "out.pdf")
    assert out is None
    # Report build called with write_pdf=True should still succeed.
    builder = ReportBuilder(runs={}, out_dir=tmp_path / "rep")
    path = builder.build(write_pdf=True)
    assert path.exists()
    assert not (tmp_path / "rep" / "report.pdf").exists()


# ── Config snapshot passes through ─────────────────────────────────────────


def test_config_dump_embedded_in_report(tmp_path):
    _write_fake_summaries(tmp_path / "summaries", {"A": 0.4, "B": 0.5})
    runs = load_variant_runs(tmp_path / "summaries")
    dump = {"temporal": {"name": "informer", "d_model": 256},
             "graph": {"gnn_name": "rgat"}}
    builder = ReportBuilder(
        runs=runs, out_dir=tmp_path / "report", config_dump=dump,
    )
    path = builder.build()
    html = path.read_text()
    # Key names appear in the JSON-serialised config dump.
    assert "informer" in html
    assert "rgat" in html
