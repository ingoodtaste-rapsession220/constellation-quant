"""Ablation orchestrator — turn an `ablation_config.yaml` into concrete runs.

Three execution modes:

* `plan`   — just emit per-variant config files + a preview JSON. No runs.
* `local`  — execute `scripts/train.py` per variant sequentially (or across
             multiple GPUs) as subprocesses. Good for small local sweeps.
* `slurm`  — emit an `sbatch` array job script referencing the generated
             configs. The actual submission is left to the user so HPC
             scheduling policies remain explicit.

The runner never talks to wandb directly; each training run is expected to
write its own `summary.json` (see `scripts/evaluate.py`) and the report
generator picks those up afterwards. This keeps the runner testable and
detached from cluster infrastructure.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import yaml

from constellation_quant.ablation.config_generator import Variant, VariantGenerator
from constellation_quant.utils import get_logger, load_config

log = get_logger(__name__)


# ── Runner config ──────────────────────────────────────────────────────────


@dataclass
class RunnerPaths:
    """Where generated configs, logs, and summaries go."""
    ablation_dir:  Path           # configs/ablation/
    output_dir:    Path           # outputs/ablation/
    summaries_dir: Path           # outputs/ablation/summaries/

    @classmethod
    def under(cls, project_root: Path) -> "RunnerPaths":
        out = project_root / "outputs" / "ablation"
        return cls(
            ablation_dir=project_root / "configs" / "ablation",
            output_dir=out,
            summaries_dir=out / "summaries",
        )


# ── Launch plan ────────────────────────────────────────────────────────────


@dataclass
class LaunchPlan:
    """Everything the runner emits before execution — inspect or dry-run."""
    variants:         List[Variant]                = field(default_factory=list)
    config_paths:     Dict[str, Path]              = field(default_factory=dict)
    commands:         Dict[str, List[str]]         = field(default_factory=dict)
    slurm_script:     Optional[Path]               = None

    def summary(self) -> List[Dict[str, Any]]:
        return [v.summary() for v in self.variants]


# ── Main runner ────────────────────────────────────────────────────────────


class AblationRunner:
    """Plan + (optionally) execute an ablation sweep.

    Args:
        ablation_config: Parsed `ablation_config.yaml` (main sweep + secondaries).
        base_model_config: Base model config dict (deep-merged per variant).
        base_feature_config: Base feature config dict (feature groups toggled per variant).
        training_config_path: Path to `training_config.yaml` — passed through
            to every per-variant training command.
        paths_config_path: Path to `paths.yaml`.
        runner_paths: Where to write outputs.
        training_script: Path to the training CLI (default `scripts/train.py`).
    """

    def __init__(
        self,
        ablation_config:     Mapping[str, Any],
        base_model_config:   Mapping[str, Any],
        base_feature_config: Mapping[str, Any],
        training_config_path: Path,
        paths_config_path:   Path,
        runner_paths:        RunnerPaths,
        training_script:     Path = Path("scripts/train.py"),
        python_executable:   str  = None,
    ):
        self.ablation_cfg = dict(ablation_config)
        self.generator = VariantGenerator(base_model_config, base_feature_config)
        self.training_cfg_path = Path(training_config_path)
        self.paths_cfg_path    = Path(paths_config_path)
        self.paths             = runner_paths
        self.training_script   = Path(training_script)
        self.python            = python_executable or os.environ.get(
            "DYNAGRAPH_PYTHON", "python"
        )

    # ── Public API ─────────────────────────────────────────────────────

    def plan(
        self,
        only: Optional[Sequence[str]] = None,
        include_sweeps: bool = True,
    ) -> LaunchPlan:
        """Generate per-variant configs and commands without executing anything."""
        variants = self._resolve_variants(only=only, include_sweeps=include_sweeps)
        self.paths.ablation_dir.mkdir(parents=True, exist_ok=True)
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.summaries_dir.mkdir(parents=True, exist_ok=True)

        plan = LaunchPlan(variants=variants)
        written = self.generator.write(variants, self.paths.ablation_dir)
        for v, path in zip(variants, written):
            plan.config_paths[v.name] = path
            plan.commands[v.name] = self._build_train_command(v, path)

        # Persist the full plan summary.
        with (self.paths.output_dir / "plan.json").open("w") as f:
            json.dump(plan.summary(), f, indent=2)
        log.info("Plan: {} variants → {}", len(variants), self.paths.ablation_dir)
        return plan

    def run_local(
        self,
        plan: LaunchPlan,
        resume: bool = True,
        dry_run: bool = False,
        executor: Optional[Callable[[str, List[str]], int]] = None,
    ) -> Dict[str, int]:
        """Execute each variant as a subprocess sequentially.

        Args:
            plan: The output of `.plan(...)`.
            resume: If True, skip variants whose `summaries/<name>.json` already exists.
            dry_run: Print commands without executing.
            executor: Injection point for tests — `fn(variant_name, argv) -> exit_code`.
                Default uses `subprocess.run`.
        """
        if executor is None:
            executor = self._default_executor
        exit_codes: Dict[str, int] = {}
        for name, argv in plan.commands.items():
            summary = self.paths.summaries_dir / f"{name}.json"
            if resume and summary.exists():
                log.info("[{}] skip (summary exists).", name)
                exit_codes[name] = 0
                continue
            if dry_run:
                log.info("[{}] DRY-RUN: {}", name, shlex.join(argv))
                exit_codes[name] = 0
                continue
            log.info("[{}] launching...", name)
            code = int(executor(name, argv))
            exit_codes[name] = code
            if code != 0:
                log.warning("[{}] exited with code {}", name, code)
        return exit_codes

    def emit_slurm_array(
        self,
        plan: LaunchPlan,
        script_path: Optional[Path] = None,
        sbatch_header: Optional[str] = None,
    ) -> Path:
        """Write an sbatch array-job script that trains one variant per task."""
        script_path = Path(script_path or (self.paths.output_dir / "run_array.sh"))
        names = list(plan.commands.keys())
        n = len(names)
        if n == 0:
            raise ValueError("Nothing to submit — plan has zero variants.")

        header = sbatch_header or self._default_sbatch_header(n)
        lines = [
            "#!/bin/bash",
            header,
            "",
            "set -euo pipefail",
            "",
            "VARIANTS=(" + " ".join(shlex.quote(n) for n in names) + ")",
            'VARIANT="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"',
            "",
            "case \"$VARIANT\" in",
        ]
        for name, argv in plan.commands.items():
            cmd = " ".join(shlex.quote(a) for a in argv)
            lines += [f"  {name})", f"    {cmd}", "    ;;"]
        lines += ["esac", ""]

        script_path.write_text("\n".join(lines) + "\n")
        script_path.chmod(0o755)
        plan.slurm_script = script_path
        log.info("Wrote SLURM array script → {} ({} tasks)", script_path, n)
        return script_path

    # ── Variants resolution ────────────────────────────────────────────

    def _resolve_variants(
        self,
        only: Optional[Sequence[str]] = None,
        include_sweeps: bool = True,
    ) -> List[Variant]:
        """Build + filter all variants (main sweep + optional secondary sweeps)."""
        main_variants = self.generator.generate(self.ablation_cfg)
        main_by_name = {v.name: v for v in main_variants}

        all_variants: List[Variant] = list(main_variants)
        if include_sweeps:
            for sweep_name, sweep_spec in (self.ablation_cfg.get("sweeps", {}) or {}).items():
                try:
                    extra = self.generator.generate_secondary_sweep(
                        sweep_name, sweep_spec, main_by_name,
                    )
                    all_variants.extend(extra)
                except ValueError as exc:
                    log.warning("Skipping sweep {!r}: {}", sweep_name, exc)

        if only:
            wanted = set(only)
            all_variants = [v for v in all_variants if v.name in wanted]
            missing = wanted - {v.name for v in all_variants}
            if missing:
                log.warning("--only specified unknown variants: {}", sorted(missing))
        return all_variants

    # ── Command building ──────────────────────────────────────────────

    def _build_train_command(self, variant: Variant, model_cfg_path: Path) -> List[str]:
        feat_cfg_path = model_cfg_path.parent / f"features_{variant.name}.yaml"
        return [
            self.python,
            str(self.training_script),
            "--model-config",    str(model_cfg_path),
            "--training-config", str(self.training_cfg_path),
            "--feature-config",  str(feat_cfg_path),
            "--paths-config",    str(self.paths_cfg_path),
            "--variant-name",    variant.name,
            "--resume",
        ]

    def _default_executor(self, variant_name: str, argv: List[str]) -> int:
        log_path = self.paths.output_dir / f"{variant_name}.log"
        with log_path.open("w") as f:
            result = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT)
        return result.returncode

    def _default_sbatch_header(self, n_tasks: int) -> str:
        exec_cfg = self.ablation_cfg.get("execution", {}) or {}
        project = str(exec_cfg.get("wandb_project", "constellation_quant_ablation"))
        return (
            "#SBATCH --job-name=cq-ablation\n"
            "#SBATCH --partition=gpu\n"
            "#SBATCH --gres=gpu:1\n"
            "#SBATCH --cpus-per-task=8\n"
            "#SBATCH --mem=64G\n"
            "#SBATCH --time=48:00:00\n"
            f"#SBATCH --array=0-{n_tasks - 1}\n"
            "#SBATCH --output=logs/%x_%A_%a.out\n"
            "#SBATCH --error=logs/%x_%A_%a.err\n"
            "\n"
            f"# wandb project: {project}\n"
            "export WANDB_MODE=offline\n"
            'export WANDB_DIR="${SCRATCH}/wandb"\n'
        )


# ── Convenience loader ─────────────────────────────────────────────────────


def runner_from_paths(
    ablation_config_path: Path,
    model_config_path:    Path,
    feature_config_path:  Path,
    training_config_path: Path,
    paths_config_path:    Path,
    project_root: Optional[Path] = None,
) -> AblationRunner:
    """Construct an AblationRunner from filesystem config paths."""
    ablation_cfg = load_config(ablation_config_path)
    model_cfg    = load_config(model_config_path)
    feature_cfg  = load_config(feature_config_path)
    project_root = Path(project_root) if project_root else Path.cwd()
    return AblationRunner(
        ablation_config=ablation_cfg,
        base_model_config=model_cfg,
        base_feature_config=feature_cfg,
        training_config_path=Path(training_config_path),
        paths_config_path=Path(paths_config_path),
        runner_paths=RunnerPaths.under(project_root),
    )
