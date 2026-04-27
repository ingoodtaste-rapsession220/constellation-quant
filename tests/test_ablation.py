"""Phase 5 tests: ablation generator + runner + report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import yaml

from constellation_quant.ablation import (
    AblationRunner,
    RunnerPaths,
    Variant,
    VariantGenerator,
    plan_graph_sweep,
    plan_temporal_sweep,
)
from constellation_quant.outputs import (
    ReportBuilder,
    VariantRun,
    comparison_table,
    load_variant_runs,
    significance_vs_previous,
)


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def base_model_cfg() -> Dict:
    return {
        "lookback": 60, "horizon": 5,
        "temporal": {"name": "informer", "d_model": 256, "n_heads": 8,
                      "dropout": 0.1, "e_layers": 3, "d_ff": 512,
                      "probsparse_factor": 5, "distil": True,
                      "use_learnable_pe": True, "pooling": "attention_weighted_mean"},
        "graph": {"enabled": True, "gnn_name": "rgat", "hidden_dim": 128,
                   "num_layers": 3, "edge_types": ["correlation", "fundamental"],
                   "dropout": 0.1, "residual": True, "attention_heads": 4},
        "edges": {"correlation": {"window": 30, "threshold": 0.5},
                   "fundamental": {"threshold": 0.7}},
        "hierarchy": {"enabled": True, "sector_nodes": 11, "market_node": 1,
                       "bidirectional": True},
        "heads": {"ranking": {"enabled": True}, "return": {"enabled": True},
                   "volatility": {"enabled": True}},
        "membership": {"mode": "dynamic"},
    }


@pytest.fixture
def base_feature_cfg() -> Dict:
    return {
        "technical":      {"enabled": True,  "indicators": {}},
        "fundamental":    {"enabled": True,  "ratios": ["pe", "pb"]},
        "sentiment":      {"enabled": False, "sources": {}},
        "graph_derived":  {"enabled": False},
        "normalization":  {"rolling_zscore_window": 252, "winsorize_std": 3.0},
    }


@pytest.fixture
def minimal_ablation_spec() -> Dict:
    return {
        "variants": [
            {"name": "A", "description": "Baseline",
             "overrides": {"graph": {"enabled": False, "gnn_name": "none"},
                            "hierarchy": {"enabled": False},
                            "membership": {"mode": "fixed"}},
             "features": ["technical"]},
            {"name": "B", "description": "Sector graph",
             "overrides": {"graph": {"gnn_name": "gat"},
                            "hierarchy": {"enabled": False},
                            "membership": {"mode": "fixed"}},
             "features": ["technical"],
             "edge_types": ["sector"]},
            {"name": "I", "description": "Full model",
             "overrides": {"multi_scale": True},
             "features": ["technical", "fundamental", "sentiment"],
             "edge_types": ["correlation", "attention", "fundamental"]},
        ],
        "sweeps": {
            "temporal_models": {
                "base_variant": "I",
                "values": ["informer", "lstm", "mamba"],
                "override_path": "temporal.name",
            },
            "graph_architectures": {
                "base_variant": "B",
                "values": ["gcn", "gat", "graphsage"],
                "override_path": "graph.gnn_name",
            },
        },
    }


# ── VariantGenerator ───────────────────────────────────────────────────────


def test_generate_variants_applies_overrides(
    base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    gen = VariantGenerator(base_model_cfg, base_feature_cfg)
    variants = gen.generate(minimal_ablation_spec)
    names = [v.name for v in variants]
    assert names == ["A", "B", "I"]

    # Model A should have graph disabled regardless of the base.
    a = variants[0]
    assert a.model_config["graph"]["enabled"] is False
    assert a.model_config["hierarchy"]["enabled"] is False
    assert a.model_config["membership"]["mode"] == "fixed"

    # Model B has sector edges injected by the edge_types list.
    b = variants[1]
    assert "sector" in b.model_config["graph"]["edge_types"]

    # Model I inherits the full base.
    i = variants[2]
    assert i.model_config["hierarchy"]["enabled"] is True


def test_feature_toggles_per_variant(
    base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    gen = VariantGenerator(base_model_cfg, base_feature_cfg)
    variants = {v.name: v for v in gen.generate(minimal_ablation_spec)}

    # A: technical only
    assert variants["A"].feature_config["technical"]["enabled"] is True
    assert variants["A"].feature_config["fundamental"]["enabled"] is False
    assert variants["A"].feature_config["sentiment"]["enabled"] is False

    # I: technical + fundamental + sentiment
    assert variants["I"].feature_config["technical"]["enabled"] is True
    assert variants["I"].feature_config["fundamental"]["enabled"] is True
    assert variants["I"].feature_config["sentiment"]["enabled"] is True


def test_secondary_sweep_swaps_one_field(
    base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    gen = VariantGenerator(base_model_cfg, base_feature_cfg)
    main_by_name = {v.name: v for v in gen.generate(minimal_ablation_spec)}
    temp_spec = minimal_ablation_spec["sweeps"]["temporal_models"]
    secondary = gen.generate_secondary_sweep("temporal_models", temp_spec, main_by_name)

    assert len(secondary) == 3
    # Only temporal.name differs — everything else matches base I.
    names = [v.name for v in secondary]
    assert "temporal_models__informer" in names
    assert "temporal_models__lstm"     in names
    assert "temporal_models__mamba"    in names
    # Verify the underlying override took effect.
    for v in secondary:
        if v.name.endswith("__lstm"):
            assert v.model_config["temporal"]["name"] == "lstm"


def test_secondary_sweep_unknown_base_raises(
    base_model_cfg, base_feature_cfg,
):
    gen = VariantGenerator(base_model_cfg, {})
    spec = {"base_variant": "NONEXISTENT", "values": [1, 2],
             "override_path": "x"}
    with pytest.raises(ValueError, match="NONEXISTENT"):
        gen.generate_secondary_sweep("x", spec, {})


def test_variant_generator_writes_configs(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    gen = VariantGenerator(base_model_cfg, base_feature_cfg)
    variants = gen.generate(minimal_ablation_spec)
    written = gen.write(variants, tmp_path)
    assert len(written) == len(variants)
    for p in written:
        assert p.exists()
    # Reading a written file should round-trip.
    round_trip = yaml.safe_load(written[0].read_text())
    assert "temporal" in round_trip


def test_variant_summary_snapshot(
    base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    gen = VariantGenerator(base_model_cfg, base_feature_cfg)
    variants = {v.name: v for v in gen.generate(minimal_ablation_spec)}
    s = variants["I"].summary()
    assert s["name"] == "I"
    assert s["membership"] == "dynamic"
    assert s["hierarchy"] is True


# ── AblationRunner ─────────────────────────────────────────────────────────


def _build_runner(tmp_path, base_model_cfg, base_feature_cfg, ablation_spec):
    return AblationRunner(
        ablation_config=ablation_spec,
        base_model_config=base_model_cfg,
        base_feature_config=base_feature_cfg,
        training_config_path=Path("configs/training_config.yaml"),
        paths_config_path=Path("configs/paths.yaml"),
        runner_paths=RunnerPaths(
            ablation_dir=tmp_path / "ablation_cfgs",
            output_dir=tmp_path / "ablation_out",
            summaries_dir=tmp_path / "ablation_out" / "summaries",
        ),
        training_script=Path("scripts/train.py"),
    )


def test_runner_plan_writes_configs_and_commands(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    runner = _build_runner(tmp_path, base_model_cfg, base_feature_cfg,
                            minimal_ablation_spec)
    plan = runner.plan(include_sweeps=False)
    # Three main variants.
    assert {v.name for v in plan.variants} == {"A", "B", "I"}
    assert set(plan.config_paths) == {"A", "B", "I"}
    for path in plan.config_paths.values():
        assert path.exists()
    # Commands include --variant-name and --model-config.
    for name, argv in plan.commands.items():
        assert "--variant-name" in argv
        assert name in argv


def test_runner_plan_only_filters(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    runner = _build_runner(tmp_path, base_model_cfg, base_feature_cfg,
                            minimal_ablation_spec)
    plan = runner.plan(only=["A", "I"], include_sweeps=False)
    assert {v.name for v in plan.variants} == {"A", "I"}


def test_runner_local_dry_run_invokes_no_subprocess(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    runner = _build_runner(tmp_path, base_model_cfg, base_feature_cfg,
                            minimal_ablation_spec)
    plan = runner.plan(include_sweeps=False)
    called: List[str] = []

    def executor(name, argv):
        called.append(name)
        return 0

    codes = runner.run_local(plan, resume=False, dry_run=True, executor=executor)
    # dry_run short-circuits before the executor.
    assert called == []
    assert all(c == 0 for c in codes.values())


def test_runner_local_resume_skips_completed(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    runner = _build_runner(tmp_path, base_model_cfg, base_feature_cfg,
                            minimal_ablation_spec)
    plan = runner.plan(include_sweeps=False)
    # Pre-create a summary for variant A.
    (runner.paths.summaries_dir / "A.json").write_text("{}")

    called: List[str] = []

    def executor(name, argv):
        called.append(name)
        return 0

    codes = runner.run_local(plan, resume=True, dry_run=False, executor=executor)
    assert "A" not in called
    assert {"B", "I"} <= set(called)
    assert all(c == 0 for c in codes.values())


def test_runner_emit_slurm_array(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    runner = _build_runner(tmp_path, base_model_cfg, base_feature_cfg,
                            minimal_ablation_spec)
    plan = runner.plan(include_sweeps=False)
    script = runner.emit_slurm_array(plan)
    assert script.exists()
    txt = script.read_text()
    assert "#SBATCH --array=0-2" in txt
    # Every variant name should appear as a case branch.
    for name in ("A", "B", "I"):
        assert f"  {name})" in txt


def test_secondary_sweep_plan_restricts_to_sweep_variants(
    tmp_path, base_model_cfg, base_feature_cfg, minimal_ablation_spec,
):
    runner = _build_runner(tmp_path, base_model_cfg, base_feature_cfg,
                            minimal_ablation_spec)
    plan = plan_temporal_sweep(runner)
    names = {v.name for v in plan.variants}
    assert names == {
        "temporal_models__informer",
        "temporal_models__lstm",
        "temporal_models__mamba",
    }


# ── Report builder ─────────────────────────────────────────────────────────


def _write_fake_summaries(summaries_dir: Path, variants: List[str],
                           sharpe_by_variant: Dict[str, float]) -> None:
    summaries_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.bdate_range("2024-01-02", periods=60)
    rng = np.random.default_rng(0)
    for v in variants:
        sharpe = sharpe_by_variant[v]
        daily_mean = sharpe / (np.sqrt(252))          # approx
        rets = rng.normal(daily_mean * 0.01, 0.01, size=len(dates))
        equity = pd.Series((1.0 + rets).cumprod(), index=dates)
        drawdown = (equity - equity.cummax()) / equity.cummax()
        pd.Series(rets, index=dates).to_csv(
            summaries_dir / f"{v}_daily_returns.csv", header=["return"],
        )
        equity.to_csv(summaries_dir / f"{v}_equity_curve.csv", header=["equity"])
        drawdown.to_csv(summaries_dir / f"{v}_drawdown.csv", header=["drawdown"])
        (summaries_dir / f"{v}.json").write_text(json.dumps({
            "backtest": {
                "sharpe": sharpe, "annual_return": 0.1, "annual_vol": 0.15,
                "max_drawdown": float(drawdown.min()), "avg_turnover": 0.4,
                "total_cost": 0.01, "final_equity": float(equity.iloc[-1]),
                "n_days": len(rets),
            },
            "regimes": {"all": {"n_days": len(rets), "annual_return": 0.1,
                                  "annual_vol": 0.15, "sharpe": sharpe,
                                  "max_drawdown": float(drawdown.min()),
                                  "hit_rate": 0.55}},
        }, indent=2))


def test_load_variant_runs_roundtrip(tmp_path):
    _write_fake_summaries(tmp_path, ["A", "B", "I"], {"A": 0.3, "B": 0.6, "I": 1.1})
    runs = load_variant_runs(tmp_path)
    assert set(runs) == {"A", "B", "I"}
    assert runs["A"].summary["sharpe"] == pytest.approx(0.3)
    assert runs["I"].daily_returns is not None
    assert runs["I"].equity is not None
    assert runs["I"].drawdown is not None


def test_comparison_table_has_every_variant(tmp_path):
    _write_fake_summaries(tmp_path, ["A", "B"], {"A": 0.3, "B": 0.7})
    runs = load_variant_runs(tmp_path)
    table = comparison_table(runs)
    assert set(table["variant"]) == {"A", "B"}
    # Sharpe column ordering in DF doesn't matter, but values should match.
    by_variant = dict(zip(table["variant"], table["sharpe"]))
    assert by_variant["A"] == pytest.approx(0.3)
    assert by_variant["B"] == pytest.approx(0.7)


def test_significance_rows_between_consecutive_variants(tmp_path):
    _write_fake_summaries(tmp_path, ["A", "B", "C"],
                           {"A": 0.3, "B": 0.7, "C": 1.0})
    runs = load_variant_runs(tmp_path)
    rows = significance_vs_previous(runs)
    # 3 variants → 2 pairwise comparisons.
    assert len(rows) == 2
    labels = [r["comparison"] for r in rows]
    assert "B vs A" in labels and "C vs B" in labels


def test_report_builder_emits_html(tmp_path):
    _write_fake_summaries(tmp_path / "summaries", ["A", "B", "I"],
                           {"A": 0.3, "B": 0.6, "I": 1.1})
    runs = load_variant_runs(tmp_path / "summaries")
    out_dir = tmp_path / "report"
    builder = ReportBuilder(runs=runs, out_dir=out_dir)
    path = builder.build()
    assert path.exists()
    html = path.read_text()
    # Variant table includes every name.
    for name in ("A", "B", "I"):
        assert name in html
    # Plots directory populated.
    plots = list((out_dir / "plots").glob("*.png"))
    assert len(plots) >= 2


def test_report_builder_empty_runs_still_emits(tmp_path):
    out_dir = tmp_path / "report"
    builder = ReportBuilder(runs={}, out_dir=out_dir)
    path = builder.build()
    assert path.exists()
    html = path.read_text()
    # Empty path should render the "no data" fallback table cell.
    assert "no data" in html or "<table" in html
