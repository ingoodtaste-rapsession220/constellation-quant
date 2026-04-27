"""Ablation study framework: runner, config generator, per-dimension sweeps, report."""

from constellation_quant.ablation.config_generator import (
    FEATURE_GROUPS,
    Variant,
    VariantGenerator,
)
from constellation_quant.ablation.runner import (
    AblationRunner,
    LaunchPlan,
    RunnerPaths,
    runner_from_paths,
)
from constellation_quant.ablation.secondary import (
    SWEEP_NAMES,
    plan_all_sweeps,
    plan_edge_sweep,
    plan_graph_sweep,
    plan_horizon_sweep,
    plan_temporal_sweep,
    plan_window_sweep,
)

__all__ = [
    "FEATURE_GROUPS",
    "Variant",
    "VariantGenerator",
    "AblationRunner",
    "LaunchPlan",
    "RunnerPaths",
    "runner_from_paths",
    "SWEEP_NAMES",
    "plan_all_sweeps",
    "plan_edge_sweep",
    "plan_graph_sweep",
    "plan_horizon_sweep",
    "plan_temporal_sweep",
    "plan_window_sweep",
]
