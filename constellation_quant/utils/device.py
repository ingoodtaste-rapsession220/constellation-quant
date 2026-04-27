"""GPU / CPU device management with DDP support.

Usage patterns:

    # Single-GPU / CPU
    device = get_device()
    model.to(device)

    # Multi-GPU (DDP) — called inside a torchrun-launched process
    local_rank = init_distributed()
    device = get_device(local_rank=local_rank)
    model = DDP(model.to(device), device_ids=[local_rank])

Environment variables read (set automatically by `torchrun`):
    LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
"""

from __future__ import annotations

import os
from typing import Optional

try:
    import torch
    import torch.distributed as dist
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def get_device(local_rank: Optional[int] = None) -> "torch.device":
    """Return the appropriate torch device.

    Args:
        local_rank: If provided (DDP context), returns `cuda:{local_rank}`.
            Otherwise returns the first CUDA device, MPS (Apple Silicon), or CPU.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for get_device()")
    if local_rank is not None and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def init_distributed(backend: str = "nccl") -> int:
    """Initialise the default process group for DDP.

    Reads `RANK`, `WORLD_SIZE`, `LOCAL_RANK` from the environment (set by
    torchrun). No-op if `WORLD_SIZE` is unset or 1.

    Returns:
        The local rank of this process. 0 for single-process runs.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch is required for init_distributed()")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return 0

    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        # NCCL on GPU, gloo on CPU
        effective_backend = backend if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=effective_backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank


def is_main_process() -> bool:
    """True on rank 0 (or when not running distributed)."""
    if not _HAS_TORCH:
        return True
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def world_size() -> int:
    """Total number of distributed processes (1 when not distributed)."""
    if not _HAS_TORCH or not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def barrier() -> None:
    """Synchronise all distributed processes. No-op when not distributed."""
    if _HAS_TORCH and dist.is_available() and dist.is_initialized():
        dist.barrier()
