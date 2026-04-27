"""Generate concrete per-variant model configs from the ablation spec.

Each variant in `ablation_config.yaml` has an `overrides` dict; this module
deep-merges that onto the base `model_config.yaml` and applies two extra
transformations the ablation spec expresses declaratively:

    features:    list of feature-group names to enable in feature_config
    edge_types:  list of edge types to expose to the R-GAT / graph layer

Both lists translate into enable/disable flags + explicit `edge_types`
values written into the concrete config, so downstream code doesn't need to
reparse the ablation spec.

Produces:
    - For each variant, a dict representing the full `model_config` with all
      overrides applied.
    - Optionally, a full `feature_config` with the right groups enabled so
      the feature engine serves the expected feature set.
    - Optionally, written YAML files under `configs/ablation/` so SLURM
      array jobs can reference them via `--model-config`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import yaml

from constellation_quant.utils import get_logger, merge_configs

log = get_logger(__name__)


# ── Known feature groups (must match feature_config.yaml sections) ─────────

FEATURE_GROUPS = ("technical", "fundamental", "sentiment", "graph_derived")


# ── Data class ─────────────────────────────────────────────────────────────


@dataclass
class Variant:
    """One concrete variant ready for training."""
    name:           str
    description:    str
    model_config:   Dict[str, Any]
    feature_config: Dict[str, Any]
    features:       List[str] = field(default_factory=list)
    edge_types:     List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        return {
            "name":        self.name,
            "description": self.description,
            "features":    list(self.features),
            "edge_types":  list(self.edge_types),
            "graph":       self.model_config.get("graph", {}).get("gnn_name", "none"),
            "hierarchy":   bool(self.model_config.get("hierarchy", {}).get("enabled", False)),
            "membership":  self.model_config.get("membership", {}).get("mode", "fixed"),
        }


# ── Core generator ─────────────────────────────────────────────────────────


class VariantGenerator:
    """Deep-merge per-variant overrides onto the base model config.

    Args:
        base_model_config:  Parsed base model_config.yaml dict.
        base_feature_config: Parsed base feature_config.yaml dict.
    """

    def __init__(
        self,
        base_model_config:   Mapping[str, Any],
        base_feature_config: Optional[Mapping[str, Any]] = None,
    ):
        self.base_model = copy.deepcopy(dict(base_model_config))
        self.base_feature = copy.deepcopy(dict(base_feature_config or {}))

    # ── Main API ───────────────────────────────────────────────────────

    def generate(self, ablation_spec: Mapping[str, Any]) -> List[Variant]:
        """Produce concrete Variants for every entry under `variants:` in the spec."""
        raw_variants = list(ablation_spec.get("variants", []))
        variants: List[Variant] = []
        for entry in raw_variants:
            variants.append(self._build_variant(entry))
        return variants

    def generate_secondary_sweep(
        self,
        sweep_name: str,
        sweep_spec: Mapping[str, Any],
        base_variants: Mapping[str, Variant],
    ) -> List[Variant]:
        """Produce variants for a secondary sweep (temporal/graph/window/etc.)."""
        base_name = str(sweep_spec["base_variant"])
        if base_name not in base_variants:
            raise ValueError(
                f"sweep {sweep_name!r} references base_variant {base_name!r} "
                f"which is not defined in the main ablation spec."
            )
        base = base_variants[base_name]
        values = list(sweep_spec.get("values", []))
        path = str(sweep_spec.get("override_path", ""))
        extras = list(sweep_spec.get("extra", []))

        out: List[Variant] = []
        for value in values:
            override = _path_to_dict(path, value) if path else {}
            name = f"{sweep_name}__{_slugify(value)}"
            out.append(self._build_secondary(name, override, base, description=(
                f"{sweep_name}: {path}={value!r} (base={base_name})"
            )))
        for extra in extras:
            out.append(self._build_secondary(
                f"{sweep_name}__{extra['name']}",
                extra.get("overrides", {}),
                base,
                description=f"{sweep_name}: extra={extra['name']} (base={base_name})",
            ))
        return out

    def write(self, variants: Iterable[Variant], out_dir: Path) -> List[Path]:
        """Dump each variant's model config to `out_dir/model_<name>.yaml`.

        Feature configs land under `out_dir/features_<name>.yaml`. Returns
        the list of model-config paths written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for v in variants:
            model_path = out_dir / f"model_{v.name}.yaml"
            feat_path  = out_dir / f"features_{v.name}.yaml"
            with model_path.open("w") as f:
                yaml.safe_dump(v.model_config, f, sort_keys=False)
            with feat_path.open("w") as f:
                yaml.safe_dump(v.feature_config, f, sort_keys=False)
            paths.append(model_path)
        return paths

    # ── Internals ──────────────────────────────────────────────────────

    def _build_variant(self, entry: Mapping[str, Any]) -> Variant:
        name = str(entry["name"])
        overrides = dict(entry.get("overrides", {}) or {})
        features = list(entry.get("features", []))
        edge_types = list(entry.get("edge_types", []))

        # Inject edge_types into the graph block if present.
        if edge_types and "graph" in overrides:
            overrides["graph"] = {**overrides["graph"], "edge_types": edge_types}
        elif edge_types:
            overrides["graph"] = {"edge_types": edge_types}

        model_cfg = merge_configs(self.base_model, overrides)
        feature_cfg = self._apply_feature_toggles(features)

        return Variant(
            name=name,
            description=str(entry.get("description", "")),
            model_config=model_cfg,
            feature_config=feature_cfg,
            features=features,
            edge_types=edge_types,
        )

    def _build_secondary(
        self,
        name: str,
        override: Mapping[str, Any],
        base: Variant,
        description: str,
    ) -> Variant:
        model_cfg = merge_configs(base.model_config, dict(override))
        return Variant(
            name=name,
            description=description,
            model_config=model_cfg,
            feature_config=copy.deepcopy(base.feature_config),
            features=list(base.features),
            edge_types=list(base.edge_types),
        )

    def _apply_feature_toggles(self, features: List[str]) -> Dict[str, Any]:
        """Enable/disable feature groups to match a variant's features list."""
        feat_cfg = copy.deepcopy(self.base_feature) or {}
        active = set(features or [])

        for group in FEATURE_GROUPS:
            if group not in feat_cfg:
                feat_cfg[group] = {}
            feat_cfg[group]["enabled"] = group in active
        return feat_cfg


# ── Utilities ──────────────────────────────────────────────────────────────


def _path_to_dict(path: str, value: Any) -> Dict[str, Any]:
    """'a.b.c', 7 → {'a': {'b': {'c': 7}}}."""
    parts = path.split(".")
    out: Dict[str, Any] = {}
    cursor = out
    for key in parts[:-1]:
        cursor[key] = {}
        cursor = cursor[key]
    cursor[parts[-1]] = value
    return out


def _slugify(value: Any) -> str:
    s = str(value).lower()
    for ch in (".", " ", "/", ":", ",", "="):
        s = s.replace(ch, "_")
    return s or "unknown"
