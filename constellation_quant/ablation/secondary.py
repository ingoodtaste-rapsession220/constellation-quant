"""Secondary-sweep drivers: temporal / graph / window / horizon / edge.

Each driver is a thin helper that plans a focused sub-sweep on top of a
fully-specified base variant (typically Model I). Shares the main runner's
machinery so secondary sweeps produce identical output artefacts (per-variant
config files + summaries) and can be reported alongside A–I.

Usage:

    runner = runner_from_paths(...)
    plan = plan_temporal_sweep(runner)
    runner.run_local(plan, dry_run=False)
    runner.emit_slurm_array(plan, script_path=out_dir / "temporal_array.sh")

Each `plan_*` returns a `LaunchPlan` that can be executed / emitted exactly
like the main sweep.
"""

from __future__ import annotations

from typing import Optional

from constellation_quant.ablation.config_generator import Variant
from constellation_quant.ablation.runner import AblationRunner, LaunchPlan


SWEEP_NAMES = (
    "temporal_models",
    "graph_architectures",
    "lookback_windows",
    "forecast_horizons",
    "edge_thresholds",
)


def _plan_single_sweep(runner: AblationRunner, sweep_name: str) -> LaunchPlan:
    """Plan a single named sweep. Falls back gracefully if the sweep is missing."""
    sweeps = runner.ablation_cfg.get("sweeps", {}) or {}
    if sweep_name not in sweeps:
        raise ValueError(f"Sweep {sweep_name!r} not in ablation_config.")
    # Restrict to only the secondary variants for this sweep — we don't want
    # to re-train A..I every time we touch a single dimension.
    main_variants = runner.generator.generate(runner.ablation_cfg)
    main_by_name = {v.name: v for v in main_variants}
    secondary = runner.generator.generate_secondary_sweep(
        sweep_name, sweeps[sweep_name], main_by_name,
    )
    only = [v.name for v in secondary]
    plan = runner.plan(only=only, include_sweeps=True)
    return plan


def plan_temporal_sweep(runner: AblationRunner) -> LaunchPlan:
    """Informer / LSTM / Transformer / TCN / Mamba — on the base variant.

    Per the spec, the base is Model I (full system). Swaps only `temporal.name`.
    """
    return _plan_single_sweep(runner, "temporal_models")


def plan_graph_sweep(runner: AblationRunner) -> LaunchPlan:
    """GCN / GAT / R-GAT / GraphSAGE — on Model D (dynamic correlation edges)."""
    return _plan_single_sweep(runner, "graph_architectures")


def plan_window_sweep(runner: AblationRunner) -> LaunchPlan:
    """L ∈ {20, 40, 60, 90, 120} + multi-scale extra."""
    return _plan_single_sweep(runner, "lookback_windows")


def plan_horizon_sweep(runner: AblationRunner) -> LaunchPlan:
    """H ∈ {1, 5, 10, 20}."""
    return _plan_single_sweep(runner, "forecast_horizons")


def plan_edge_sweep(runner: AblationRunner) -> LaunchPlan:
    """Correlation-edge threshold + top-K density sweep."""
    return _plan_single_sweep(runner, "edge_thresholds")


def plan_all_sweeps(runner: AblationRunner) -> dict[str, LaunchPlan]:
    """Plan all five secondary sweeps. Returns {sweep_name: plan}."""
    out: dict[str, LaunchPlan] = {}
    for name in SWEEP_NAMES:
        try:
            out[name] = _plan_single_sweep(runner, name)
        except ValueError as exc:
            # Sweep might not be defined in this ablation config.
            out[name] = LaunchPlan()
            _ = exc
    return out
