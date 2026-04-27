"""Main training loop.

Iterates the training Dataset in chronological order (no shuffle), in
mini-batches of `batch_size` time steps using gradient accumulation.
For each date we run the model, compute the multi-task loss, accumulate
gradients, and step the optimizer once per mini-batch.

Supports:
    - mixed precision (fp16 / bfloat16) via `torch.amp`
    - distributed data parallel (each rank handles a disjoint slice of dates)
    - checkpoint resume (newest epoch checkpoint for the current variant)
    - early stopping on val_IC (max mode)
    - wandb logging (offline on HPC, set by env)

`sample_adapter` abstracts the "turn a Dataset sample into model inputs +
targets" step — the trainer knows nothing about graph construction or
edge types, keeping it reusable across ablation variants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constellation_quant.training.checkpoint import CheckpointManager
from constellation_quant.training.losses import MultiTaskLoss
from constellation_quant.training.scheduler import build_scheduler
from constellation_quant.training.validator import Validator
from constellation_quant.utils import get_logger, is_main_process

log = get_logger(__name__)


SampleAdapter = Callable[[Dict[str, Any], torch.device], Dict[str, Any]]
"""Turn a Dataset sample into a dict of tensors on device. Returns keys:
    features, mask, targets (ranking target), sector_indices, edges (dict),
    plus any optional auxiliary targets (return, volatility).
"""


@dataclass
class TrainerConfig:
    optimizer_cfg: Mapping[str, Any]
    scheduler_cfg: Mapping[str, Any]
    loop_cfg:      Mapping[str, Any]
    loss_cfg:      Mapping[str, Any]
    regularization_cfg: Mapping[str, Any]
    mixed_precision_cfg: Mapping[str, Any]
    early_stopping_cfg: Mapping[str, Any]
    checkpoint_cfg:    Mapping[str, Any]
    wandb_cfg:         Mapping[str, Any]
    variant_name:      str
    checkpoint_dir:    Path


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = float("-inf")
    epochs_without_improvement: int = 0
    history: List[Dict[str, float]] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        val_dataset,
        sample_adapter: SampleAdapter,
        config: TrainerConfig,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.model = model
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.sample_adapter = sample_adapter
        self.cfg = config
        self.device = device
        self.rank = rank
        self.world_size = world_size

        self.optimizer = self._build_optimizer()
        self.steps_per_epoch = max(1, len(train_dataset) // self._effective_batch())
        self.scheduler = build_scheduler(
            self.optimizer, dict(config.scheduler_cfg),
            steps_per_epoch=self.steps_per_epoch,
        )
        self.loss_fn = MultiTaskLoss(dict(config.loss_cfg))
        self.validator = Validator(device=device, top_n=50)

        # Mixed precision
        mp_cfg = dict(config.mixed_precision_cfg or {})
        self.mp_enabled = bool(mp_cfg.get("enabled", False)) and device.type == "cuda"
        mp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            str(mp_cfg.get("dtype", "float16")), torch.float16,
        )
        self.mp_dtype = mp_dtype
        self.grad_scaler = torch.amp.GradScaler(device.type, enabled=self.mp_enabled and mp_dtype is torch.float16)

        # Checkpointing
        self.ckpt = CheckpointManager(
            ckpt_dir=config.checkpoint_dir,
            variant_name=config.variant_name,
            keep_last_n=int(config.checkpoint_cfg.get("keep_last_n", 3)),
        )
        self.state = TrainState()

        # wandb
        self.wandb = _maybe_init_wandb(config.wandb_cfg, config.variant_name,
                                       rank, config_dump={
                                           "model": getattr(model, "describe", lambda: {})(),
                                           "optimizer": dict(config.optimizer_cfg),
                                           "scheduler": dict(config.scheduler_cfg),
                                           "loop": dict(config.loop_cfg),
                                       })

    # ── Public API ─────────────────────────────────────────────────────

    def fit(self, resume: bool = True) -> TrainState:
        if resume:
            self._maybe_resume()

        max_epochs = int(self.cfg.loop_cfg.get("max_epochs", 100))
        patience = int(self.cfg.early_stopping_cfg.get("patience", 15))
        enable_es = bool(self.cfg.early_stopping_cfg.get("enabled", True))

        for epoch in range(self.state.epoch, max_epochs):
            self.state.epoch = epoch
            train_stats = self._train_one_epoch(epoch)

            val_stats = self.validator.evaluate(
                model=self._underlying_model(),
                dataset=self.val_ds,
                prepare_sample=self.sample_adapter,
            )
            self._log_epoch(epoch, train_stats, val_stats)

            metric = float(val_stats.get("ic_mean", float("nan")))
            improved = metric > self.state.best_metric and torch.isfinite(torch.tensor(metric))
            if improved:
                self.state.best_metric = metric
                self.state.epochs_without_improvement = 0
                if is_main_process():
                    self.ckpt.save_best(self._underlying_model(), self.optimizer,
                                        self.scheduler, epoch, metric, self.cfg.__dict__)
            else:
                self.state.epochs_without_improvement += 1

            if is_main_process() and (epoch + 1) % int(self.cfg.checkpoint_cfg.get(
                "save_periodic_every", 10)) == 0:
                self.ckpt.save_periodic(self._underlying_model(), self.optimizer,
                                         self.scheduler, epoch, self.state.best_metric,
                                         self.cfg.__dict__)

            # Scheduler step
            self._scheduler_step(metric)

            if enable_es and self.state.epochs_without_improvement >= patience:
                log.info("Early stopping: no val_IC improvement for {} epochs", patience)
                break

        return self.state

    # ── Epoch ──────────────────────────────────────────────────────────

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        per_component_sum: Dict[str, float] = {}

        clip_norm = float(self.cfg.regularization_cfg.get("gradient_clip_norm", 0.0))
        batch_size = self._effective_batch()

        # Rotate stride start so each epoch covers a different non-overlapping
        # subsequence of dates. Across `stride` epochs we visit every valid
        # date once. Val/test never get rotated — their offset stays at 0.
        rotate = getattr(self.train_ds, "set_epoch_offset", None)
        if callable(rotate):
            stride = int(getattr(self.train_ds, "stride", 1))
            rotate(epoch % max(stride, 1))

        indices = self._epoch_date_indices(epoch)
        self.optimizer.zero_grad(set_to_none=True)

        for step, (start, end) in enumerate(_batched_ranges(indices, batch_size)):
            mini_total = self._mini_batch_forward(start, end)
            total_loss += float(mini_total.total.detach().cpu())
            for name, val in mini_total.per_component.items():
                per_component_sum[name] = per_component_sum.get(name, 0.0) + float(val.cpu())
            total_steps += 1

            if self.grad_scaler.is_enabled():
                self.grad_scaler.unscale_(self.optimizer)
            if clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)

            if self.grad_scaler.is_enabled():
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            # one-cycle scheduler steps per batch, not per epoch.
            if getattr(self.scheduler, "_schedule_step_per_batch", False):
                self.scheduler.step()

        avg_loss = total_loss / max(total_steps, 1)
        return {
            "train_loss": avg_loss,
            **{f"train_{k}": v / max(total_steps, 1)
               for k, v in per_component_sum.items()},
        }

    def _mini_batch_forward(self, start: int, end: int):
        """One batched forward+backward over `[start, end)` dates.

        Gathers B dates, stacks their tensors into a (B, N, ...) batch, runs
        the model once (Informer sees all B*N stock-windows in parallel — the
        main throughput win on H100), computes the loss once, and does a
        single backward pass. Per-date sequential looping was the pre-batching
        pattern and left the GPU idle between dates.
        """
        B = end - start
        per_date_inputs: List[Dict[str, Any]] = []
        for i in range(start, end):
            sample = self.train_ds[i]
            per_date_inputs.append(self.sample_adapter(sample, self.device))

        features = torch.stack([p["features"] for p in per_date_inputs])           # (B,N,L,F)
        mask     = torch.stack([p["mask"]     for p in per_date_inputs])           # (B,N)
        targets  = torch.stack([p["targets"]  for p in per_date_inputs])           # (B,N)
        sectors  = torch.stack([p["sector_indices"] for p in per_date_inputs])     # (B,N)
        edges_list = [p.get("edges") for p in per_date_inputs]
        slow_features = (
            torch.stack([p["slow_features"] for p in per_date_inputs])             # (B,N,F_slow)
            if all("slow_features" in p for p in per_date_inputs) else None
        )

        with torch.amp.autocast(self.device.type, dtype=self.mp_dtype, enabled=self.mp_enabled):
            out = self.model(
                features=features,
                mask=mask,
                edges=edges_list,
                sector_indices=sectors,
                slow_features=slow_features,
            )
            predictions: Dict[str, torch.Tensor] = {"ranking": out.scores}
            targets_dict: Dict[str, torch.Tensor] = {"ranking": targets}
            if out.ret is not None and all("return" in p for p in per_date_inputs):
                predictions["return"] = out.ret
                targets_dict["return"] = torch.stack(
                    [p["return"] for p in per_date_inputs]
                )
            if (out.volatility is not None
                    and all("volatility" in p for p in per_date_inputs)):
                predictions["volatility"] = out.volatility
                targets_dict["volatility"] = torch.stack(
                    [p["volatility"] for p in per_date_inputs]
                )

            comps = self.loss_fn(predictions, targets_dict, mask=mask)

        if self.grad_scaler.is_enabled():
            self.grad_scaler.scale(comps.total).backward()
        else:
            comps.total.backward()

        return comps

    # ── Setup helpers ──────────────────────────────────────────────────

    def _build_optimizer(self) -> Optimizer:
        cfg = dict(self.cfg.optimizer_cfg or {})
        name = str(cfg.get("name", "adamw")).lower()
        lr = float(cfg.get("lr", 1e-4))
        weight_decay = float(cfg.get("weight_decay", 1e-4))
        if name == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(cfg.get("betas", (0.9, 0.999))),
                eps=float(cfg.get("eps", 1e-8)),
            )
        raise ValueError(f"Unsupported optimizer {name!r}")

    def _scheduler_step(self, val_metric: float) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if torch.isfinite(torch.tensor(val_metric)):
                self.scheduler.step(val_metric)
        else:
            try:
                self.scheduler.step()
            except Exception as exc:
                log.debug("scheduler.step() failed: {}", exc)

    def _effective_batch(self) -> int:
        return int(self.cfg.loop_cfg.get("batch_size", 32))

    def _epoch_date_indices(self, epoch: int) -> List[int]:
        """Indices into the train dataset for this epoch / rank."""
        total = len(self.train_ds)
        all_idx = list(range(total))
        if self.world_size <= 1:
            return all_idx
        # Distributed: each rank takes every world_size-th index starting at rank.
        return all_idx[self.rank :: self.world_size]

    def _maybe_resume(self) -> None:
        resume_path = self.ckpt.find_resume()
        if resume_path is None:
            return
        log.info("Resuming from {}", resume_path)
        ckpt = self.ckpt.load_into(resume_path, self._underlying_model(),
                                    optimizer=self.optimizer, scheduler=self.scheduler)
        self.state = TrainState(
            epoch=ckpt.epoch + 1,
            best_metric=ckpt.best_metric,
        )

    def _underlying_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _log_epoch(self, epoch: int, train_stats: Dict[str, float],
                    val_stats: Mapping[str, Any]) -> None:
        # Pull top-K metrics (already computed by the Validator with top_n=50)
        # so each epoch line tells us both the rank-correlation signal
        # (val_ic) AND the actual long-short outcome (hit_rate / spread).
        log.info(
            "epoch {:04d} | train_loss={:.5f} | "
            "val_ic={:.5f} | val_ic_ir={:.3f} | "
            "hit@50={:.3f} | spread@50={:+.5f}",
            epoch, train_stats["train_loss"],
            val_stats.get("ic_mean", float("nan")),
            val_stats.get("ic_ir", float("nan")),
            val_stats.get("hit_rate", float("nan")),
            val_stats.get("long_short_spread", float("nan")),
        )
        self.state.history.append({**train_stats,
                                    **{f"val_{k}": v for k, v in val_stats.items()
                                       if isinstance(v, (int, float))}})
        if self.wandb is not None:
            self.wandb.log({
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_stats,
                **{f"val_{k}": v for k, v in val_stats.items()
                   if isinstance(v, (int, float))},
            })


# ── Helpers ────────────────────────────────────────────────────────────────


def _batched_ranges(indices: List[int], batch_size: int):
    """Yield (start, end) tuples that group contiguous index ranges."""
    for i in range(0, len(indices), batch_size):
        chunk = indices[i : i + batch_size]
        if not chunk:
            continue
        yield chunk[0], chunk[-1] + 1


def _maybe_init_wandb(wandb_cfg: Mapping[str, Any], variant: str, rank: int,
                      config_dump: Mapping[str, Any]):
    cfg = dict(wandb_cfg or {})
    if not cfg.get("enabled", False) or rank != 0:
        return None
    try:
        import wandb
        wandb.init(
            project=str(cfg.get("project", "constellation_quant")),
            name=variant,
            config=dict(config_dump),
            mode="offline" if cfg.get("offline_on_hpc", True) else "online",
        )
        return wandb
    except Exception as exc:
        log.warning("wandb init failed: {}", exc)
        return None


def default_sample_adapter(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Minimal adapter for tests / unit runs that don't use the full graph.

    Expects a dict with keys {features, targets, mask, sectors}. Produces the
    shape the trainer consumes. Adds no edges — use a real adapter in
    production runs (lives in scripts/train.py).
    """
    return {
        "features": sample["features"].to(device),
        "targets":  sample["targets"].to(device),
        "mask":     sample["mask"].to(device),
        "sector_indices": sample.get("sectors", torch.zeros_like(sample["mask"], dtype=torch.long)).to(device),
        "edges": {},
    }
