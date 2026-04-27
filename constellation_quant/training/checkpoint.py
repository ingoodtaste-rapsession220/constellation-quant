"""Checkpoint save / load.

One file = one full training snapshot:
    model state, optimizer state, scheduler state, epoch, best metric,
    the config dict that produced it (for traceability), and the git hash
    at save time.

The Trainer calls `CheckpointManager.save_best(...)` whenever val_IC
improves and `.save_periodic(...)` on every Nth epoch. On start-up it calls
`.find_resume(...)` — the newest complete checkpoint for the current
variant — so preemption-restarted jobs pick up where they stopped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch

from constellation_quant.utils import get_logger, log_environment

log = get_logger(__name__)


@dataclass
class Checkpoint:
    model_state:     Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    epoch:           int
    best_metric:     float
    config:          Dict[str, Any]
    meta:            Dict[str, Any]


class CheckpointManager:
    def __init__(
        self,
        ckpt_dir: Path,
        variant_name: str = "default",
        keep_last_n: int = 3,
    ):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.variant = variant_name
        self.keep_last_n = keep_last_n

    # ── Save ───────────────────────────────────────────────────────────

    def save_best(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        best_metric: float,
        config: Mapping[str, Any],
    ) -> Path:
        payload = self._build_payload(model, optimizer, scheduler, epoch, best_metric, config)
        path = self.dir / f"{self.variant}_best.pt"
        torch.save(payload, path)
        log.info("[ckpt] saved best -> {} (epoch {}, metric {:.5f})",
                 path.name, epoch, best_metric)
        return path

    def save_periodic(
        self,
        model,
        optimizer,
        scheduler,
        epoch: int,
        best_metric: float,
        config: Mapping[str, Any],
    ) -> Path:
        payload = self._build_payload(model, optimizer, scheduler, epoch, best_metric, config)
        path = self.dir / f"{self.variant}_epoch{epoch:04d}.pt"
        torch.save(payload, path)
        log.info("[ckpt] saved periodic -> {}", path.name)
        self._prune_old()
        return path

    # ── Load ───────────────────────────────────────────────────────────

    def find_resume(self) -> Optional[Path]:
        """Newest epoch-XXXX checkpoint for this variant, or `best` if nothing else."""
        pattern = re.compile(rf"^{re.escape(self.variant)}_epoch(\d+)\.pt$")
        best_epoch = -1
        best_path: Optional[Path] = None
        for p in self.dir.glob(f"{self.variant}_epoch*.pt"):
            m = pattern.match(p.name)
            if m and int(m.group(1)) > best_epoch:
                best_epoch = int(m.group(1))
                best_path = p
        if best_path is not None:
            return best_path
        best = self.dir / f"{self.variant}_best.pt"
        return best if best.exists() else None

    @staticmethod
    def load(path: Path, map_location: Any = "cpu") -> Checkpoint:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        return Checkpoint(
            model_state=payload["model_state"],
            optimizer_state=payload["optimizer_state"],
            scheduler_state=payload.get("scheduler_state"),
            epoch=int(payload["epoch"]),
            best_metric=float(payload["best_metric"]),
            config=dict(payload.get("config", {})),
            meta=dict(payload.get("meta", {})),
        )

    def load_into(
        self,
        path: Path,
        model,
        optimizer=None,
        scheduler=None,
        strict: bool = True,
    ) -> Checkpoint:
        ckpt = self.load(path)
        model.load_state_dict(ckpt.model_state, strict=strict)
        if optimizer is not None and ckpt.optimizer_state is not None:
            optimizer.load_state_dict(ckpt.optimizer_state)
        if scheduler is not None and ckpt.scheduler_state is not None:
            try:
                scheduler.load_state_dict(ckpt.scheduler_state)
            except Exception as exc:
                log.warning("[ckpt] scheduler restore failed ({}); reinitialising", exc)
        return ckpt

    # ── Internals ──────────────────────────────────────────────────────

    def _build_payload(self, model, optimizer, scheduler, epoch, best_metric, config):
        return {
            "model_state":     _unwrap_ddp(model).state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "epoch":           epoch,
            "best_metric":     float(best_metric),
            "config":          dict(config),
            "meta":            log_environment(),
        }

    def _prune_old(self) -> None:
        if self.keep_last_n <= 0:
            return
        ckpts = sorted(
            self.dir.glob(f"{self.variant}_epoch*.pt"),
            key=lambda p: p.stat().st_mtime,
        )
        for old in ckpts[:-self.keep_last_n]:
            old.unlink(missing_ok=True)


def _unwrap_ddp(model):
    """Return the underlying model if wrapped in DistributedDataParallel."""
    return model.module if hasattr(model, "module") else model
