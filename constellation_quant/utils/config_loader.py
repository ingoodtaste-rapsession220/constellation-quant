"""YAML config loading with Pydantic validation and inheritance.

Supports:
- Environment variable expansion: `$SCRATCH/data` -> `/scratch/user/data`
- Config inheritance: a config may declare `extends: base.yaml` to merge fields
- Deep merge: nested dicts are merged recursively; lists are replaced wholesale
- Pydantic validation: schemas in `constellation_quant.utils.schemas` (populated as modules land)
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_ENV_VAR_PATTERN = re.compile(r"\$\{?([A-Z_][A-Z0-9_]*)\}?")


def _expand_env(value: Any) -> Any:
    """Recursively expand $VAR and ${VAR} in string values."""
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            var = m.group(1)
            return os.environ.get(var, m.group(0))
        return _ENV_VAR_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge override into base. Lists are replaced, not concatenated.

    Nested dicts are merged recursively. Scalar values in `override` win.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(
    path: str | Path,
    schema: Optional[Type[T]] = None,
    expand_env: bool = True,
    _seen: Optional[set] = None,
) -> Dict[str, Any] | T:
    """Load a YAML config, resolving `extends` inheritance chains.

    Args:
        path: Path to the YAML file.
        schema: Optional Pydantic model to validate against. If provided,
            returns a validated instance; otherwise returns a plain dict.
        expand_env: If True, expand $VAR references against `os.environ`.
        _seen: Internal — tracks visited paths to detect inheritance cycles.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ValueError: If an inheritance cycle is detected.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    seen = _seen or set()
    if path in seen:
        raise ValueError(f"Cyclic config inheritance detected at {path}")
    seen = seen | {path}

    with path.open("r") as f:
        raw = yaml.safe_load(f) or {}

    if "extends" in raw:
        parent_rel = raw.pop("extends")
        parent_path = (path.parent / parent_rel).resolve()
        parent = load_config(parent_path, schema=None, expand_env=False, _seen=seen)
        raw = merge_configs(parent, raw)

    if expand_env:
        raw = _expand_env(raw)

    if schema is not None:
        return schema(**raw)
    return raw
