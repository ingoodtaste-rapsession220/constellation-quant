"""Reproducibility helpers: seed all RNGs, log the run environment.

Call `set_seed()` at the start of every training/eval entry point, then
`log_environment()` after wandb is initialised to record git hash and library
versions in the run metadata.
"""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA).

    Args:
        seed: Integer seed applied to all RNGs.
        deterministic: If True, enable CUDA deterministic algorithms. May slow
            training but guarantees bit-identical results across runs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if _HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _git_hash() -> Optional[str]:
    """Return the current HEAD commit hash, or None if not a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_environment() -> Dict[str, Any]:
    """Return a dict of environment metadata for wandb/log traceability."""
    env: Dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_hash": _git_hash(),
        "numpy_version": np.__version__,
    }
    if _HAS_TORCH:
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_name"] = torch.cuda.get_device_name(0)
    return env
