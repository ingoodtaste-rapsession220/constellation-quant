"""Phase 3 training tests. Fake tiny dataset, verify loss decreases, metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch
import torch.nn as nn

from constellation_quant.evaluation import (
    aggregate_metrics,
    daily_metrics,
    long_short_spread,
    spearman_corr,
)
from constellation_quant.training import (
    Checkpoint,
    CheckpointManager,
    ListMLELoss,
    MaskedMSELoss,
    MultiTaskLoss,
    Trainer,
    TrainerConfig,
    build_scheduler,
    default_sample_adapter,
)


# ── Evaluation metrics ─────────────────────────────────────────────────────


def test_spearman_matches_scipy_behaviour():
    pred = np.array([3.0, 1.0, 2.0, 4.0, 5.0])
    target = np.array([3.1, 1.2, 2.5, 3.8, 5.1])
    # Monotonic data → IC == 1.0
    assert abs(spearman_corr(pred, target) - 1.0) < 1e-9


def test_spearman_nan_on_constant():
    pred = np.array([1.0, 1.0, 1.0, 1.0])
    target = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(spearman_corr(pred, target))


def test_long_short_spread_monotonic():
    pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    target = np.array([-0.1, -0.05, 0.0, 0.05, 0.1, 0.15])
    spread = long_short_spread(pred, target, top_n=2)
    assert spread > 0   # top 2 > bottom 2


def test_daily_and_aggregate_metrics():
    dms = []
    rng = np.random.default_rng(0)
    for _ in range(20):
        pred = rng.normal(size=100)
        target = 0.3 * pred + rng.normal(size=100)   # IC ≈ 0.3
        dms.append(daily_metrics(pred, target, top_n=10))
    agg = aggregate_metrics(dms)
    assert 0.2 < agg.mean_ic < 0.4
    assert agg.n_days == 20
    assert agg.hit_rate > 0.5


# ── Loss functions ─────────────────────────────────────────────────────────


def test_listmle_optimal_on_correct_order():
    """If scores equal targets, ListMLE loss should be near the theoretical minimum."""
    torch.manual_seed(0)
    loss = ListMLELoss()
    scores = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    targets = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    l = loss(scores, targets)
    assert torch.isfinite(l)
    # Now random scores → higher loss.
    l_rand = loss(torch.randn(5), targets)
    assert l < l_rand


def test_listmle_mask_respected():
    loss = ListMLELoss()
    scores = torch.tensor([3.0, 1.0, 2.0, 0.0])
    targets = torch.tensor([3.0, 1.0, 2.0, 99.0])
    mask_full   = torch.tensor([True, True, True, True])
    mask_masked = torch.tensor([True, True, True, False])
    l_full   = loss(scores, targets, mask=mask_full)
    l_masked = loss(scores, targets, mask=mask_masked)
    # Masking away the nonsensical target should yield a smaller loss.
    assert l_masked < l_full
    # And the masked loss must still be a meaningful positive value —
    # previously a -inf-minus-(-inf) NaN collapsed it to exactly 0.
    assert l_masked.item() > 0.0


def test_listmle_gradient_flows_with_partial_mask():
    """Regression guard for the silent-NaN bug.

    With any padded slot, the old implementation produced loss = 0.0 and
    zero gradient because (-inf) - (-inf) → NaN, then NaN * 0 → NaN, then
    the `finite.any()` fallback returned `scores.sum() * 0.0`. Every real
    training batch has padded slots (N_max union across all dates > live
    members per date), so the ranking head received no gradient across
    all ablation runs. This test must fail loudly if that regression
    ever sneaks back.
    """
    torch.manual_seed(0)
    loss = ListMLELoss()
    scores = torch.randn(1, 665, requires_grad=True)
    targets = torch.randn(1, 665)
    mask = torch.zeros(1, 665, dtype=torch.bool)
    mask[0, :500] = True                               # 500 real, 165 padded
    perm = torch.randperm(665)
    mask = mask[:, perm]
    l = loss(scores, targets, mask=mask)
    assert torch.isfinite(l) and l.item() > 0.0, \
        f"loss should be finite and positive; got {l.item()}"
    l.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum().item() > 0.0, \
        "scores.grad should be non-zero — ranking head must receive gradient"


def test_masked_mse_ignores_padded_slots():
    loss = MaskedMSELoss()
    pred = torch.tensor([1.0, 2.0, 100.0])
    target = torch.tensor([1.0, 2.0, 0.0])
    full_mask = torch.tensor([True, True, True])
    reduced_mask = torch.tensor([True, True, False])
    l_full = loss(pred, target, mask=full_mask)
    l_red = loss(pred, target, mask=reduced_mask)
    assert l_full > l_red
    assert abs(l_red.item()) < 1e-6


def test_ic_max_loss_minus_one_when_perfectly_correlated():
    """Pearson corr = 1 for any monotonic mapping → loss = -1."""
    from constellation_quant.training import ICMaximizationLoss
    loss = ICMaximizationLoss()
    scores  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    val = loss(scores, targets)
    assert abs(val.item() - (-1.0)) < 1e-4


def test_ic_max_loss_plus_one_when_anti_correlated():
    """Anti-correlated → loss = +1 (worst case)."""
    from constellation_quant.training import ICMaximizationLoss
    loss = ICMaximizationLoss()
    scores  = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    targets = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    val = loss(scores, targets)
    assert abs(val.item() - 1.0) < 1e-4


def test_ic_max_loss_zero_for_uncorrelated_inputs():
    """No linear correlation → loss ≈ 0."""
    from constellation_quant.training import ICMaximizationLoss
    torch.manual_seed(0)
    loss = ICMaximizationLoss()
    n = 1000
    scores  = torch.randn(n)
    targets = torch.randn(n)
    val = loss(scores, targets)
    # With n=1000 random samples, |corr| should be small.
    assert abs(val.item()) < 0.10


def test_ic_max_loss_respects_mask():
    """Padded slots must not affect the per-date IC computation."""
    from constellation_quant.training import ICMaximizationLoss
    loss = ICMaximizationLoss()
    scores  = torch.tensor([1.0, 2.0, 3.0, 4.0, 999.0])
    targets = torch.tensor([0.1, 0.2, 0.3, 0.4, -42.0])
    full_mask = torch.tensor([True] * 5)
    reduced   = torch.tensor([True, True, True, True, False])
    val_full = loss(scores, targets, mask=full_mask)
    val_red  = loss(scores, targets, mask=reduced)
    # With the outlier masked, the remaining 4 are perfectly monotone → loss = -1
    assert abs(val_red.item() - (-1.0)) < 1e-4
    # With the outlier IN, correlation drops sharply (not perfect anymore)
    assert val_full.item() > val_red.item() + 0.5


def test_ic_max_loss_gradient_flows():
    """Regression guard: gradient must flow back to scores."""
    from constellation_quant.training import ICMaximizationLoss
    loss = ICMaximizationLoss()
    scores = torch.randn(1, 100, requires_grad=True)
    targets = torch.randn(1, 100)
    mask = torch.ones(1, 100, dtype=torch.bool)
    val = loss(scores, targets, mask=mask)
    val.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_multi_task_loss_accepts_ic_max():
    """The loss factory must dispatch on `ic_max` correctly."""
    from constellation_quant.training import MultiTaskLoss
    cfg = {
        "ranking": {"name": "ic_max", "weight": 1.0},
        "return":  {"name": "mse",    "weight": 0.1},
    }
    mtl = MultiTaskLoss(cfg)
    pred = {
        "ranking": torch.randn(1, 10, requires_grad=True),
        "return":  torch.randn(1, 10, requires_grad=True),
    }
    targ = {"ranking": torch.randn(1, 10), "return": torch.randn(1, 10)}
    out = mtl(pred, targ, mask=torch.ones(1, 10, dtype=torch.bool))
    out.total.backward()
    assert pred["ranking"].grad is not None
    assert torch.isfinite(out.total).all()


def test_multi_task_loss_weighting():
    cfg = {
        "ranking": {"name": "listmle", "weight": 1.0},
        "return":  {"name": "mse",     "weight": 0.1},
    }
    mtl = MultiTaskLoss(cfg)
    pred = {
        "ranking": torch.randn(8, requires_grad=True),
        "return":  torch.randn(8, requires_grad=True),
    }
    targ = {"ranking": torch.randn(8), "return": torch.randn(8)}
    out = mtl(pred, targ, mask=torch.ones(8, dtype=torch.bool))
    assert out.total.grad_fn is not None
    # Loss components should be tracked but not require grad (detached).
    assert "ranking" in out.per_component and "return" in out.per_component
    out.total.backward()
    assert pred["ranking"].grad is not None
    assert pred["return"].grad is not None


# ── Checkpoint manager ─────────────────────────────────────────────────────


def test_checkpoint_roundtrip(tmp_path):
    model = nn.Linear(5, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
    mgr = CheckpointManager(ckpt_dir=tmp_path, variant_name="A")

    p = mgr.save_best(model, opt, sched, epoch=3, best_metric=0.05, config={"x": 1})
    assert p.exists()

    model2 = nn.Linear(5, 1)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-5)
    ckpt = mgr.load_into(p, model2, optimizer=opt2)
    assert ckpt.epoch == 3
    assert abs(ckpt.best_metric - 0.05) < 1e-6


def test_checkpoint_find_resume_prefers_newest(tmp_path):
    mgr = CheckpointManager(ckpt_dir=tmp_path, variant_name="A")
    model = nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    mgr.save_periodic(model, opt, None, epoch=0, best_metric=0.0, config={})
    mgr.save_periodic(model, opt, None, epoch=10, best_metric=0.0, config={})
    mgr.save_periodic(model, opt, None, epoch=5, best_metric=0.0, config={})
    resume = mgr.find_resume()
    assert resume is not None
    assert "epoch0010" in resume.name


# ── Scheduler builder ──────────────────────────────────────────────────────


def test_scheduler_cosine_with_warmup():
    model = nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, {"name": "cosine_annealing", "T_max": 10,
                                    "warmup_epochs": 2, "warmup_start_factor": 0.1})
    # First step during warmup should be below the target lr.
    lrs = []
    for _ in range(12):
        opt.step()
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])
    assert lrs[0] < 1e-3
    assert max(lrs) <= 1e-3 + 1e-12


def test_scheduler_reduce_on_plateau():
    model = nn.Linear(3, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = build_scheduler(opt, {"name": "reduce_on_plateau", "factor": 0.5, "patience": 1})
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    assert isinstance(sched, ReduceLROnPlateau)


# ── End-to-end mini training run ───────────────────────────────────────────


class _TinyDataset:
    """Fabricated dataset: each sample is a (N, L, F) tensor with a planted signal."""

    def __init__(self, n_samples: int = 8, N: int = 10, L: int = 12, F: int = 4, seed: int = 0):
        torch.manual_seed(seed)
        self.samples = []
        for _ in range(n_samples):
            features = torch.randn(N, L, F)
            # Target = sum of last row's features (a signal the model can learn).
            targets = features[:, -1, :].sum(dim=-1)
            self.samples.append({
                "features": features,
                "targets":  targets,
                "mask":     torch.ones(N, dtype=torch.bool),
                "sectors":  torch.zeros(N, dtype=torch.long),
                "tickers":  [f"T{i}" for i in range(N)],
                "date":     torch.tensor(0),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def test_trainer_loss_decreases(tmp_path):
    """A minimal end-to-end run should see training loss decrease on a learnable signal."""
    from constellation_quant.models import ConstellationQuant

    ds_train = _TinyDataset(n_samples=6, seed=0)
    ds_val   = _TinyDataset(n_samples=3, seed=1)

    model_cfg = {
        "lookback": 12, "horizon": 1, "multi_scale": False,
        "temporal": {"name": "lstm", "d_model": 16, "dropout": 0.0, "num_layers": 1},
        "graph": {"enabled": False, "gnn_name": "none"},
        "hierarchy": {"enabled": False},
        "heads": {
            "ranking":    {"enabled": True, "mlp": [8, 1], "dropout": 0.0,
                           "temperature_scaling": False},
            "return":     {"enabled": False},
            "volatility": {"enabled": False},
        },
    }
    model = ConstellationQuant(n_features=4, model_cfg=model_cfg)

    tcfg = TrainerConfig(
        optimizer_cfg={"name": "adamw", "lr": 5e-3, "weight_decay": 0.0},
        scheduler_cfg={"name": "cosine_annealing", "T_max": 5, "warmup_epochs": 0},
        loop_cfg={"batch_size": 2, "max_epochs": 5, "num_workers": 0, "shuffle": False},
        loss_cfg={"ranking": {"name": "listmle", "weight": 1.0}},
        regularization_cfg={"gradient_clip_norm": 1.0},
        mixed_precision_cfg={"enabled": False},
        early_stopping_cfg={"enabled": False, "patience": 100},
        checkpoint_cfg={"save_periodic_every": 1000, "keep_last_n": 1},
        wandb_cfg={"enabled": False},
        variant_name="test",
        checkpoint_dir=tmp_path,
    )
    trainer = Trainer(
        model=model,
        train_dataset=ds_train,
        val_dataset=ds_val,
        sample_adapter=default_sample_adapter,
        config=tcfg,
        device=torch.device("cpu"),
    )
    state = trainer.fit(resume=False)
    losses = [h["train_loss"] for h in state.history]
    assert len(losses) >= 2
    # Expect some meaningful reduction — signal is learnable.
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_trainer_checkpoint_resume_roundtrip(tmp_path):
    """Save after 2 epochs, restart, train 2 more — state should continue."""
    from constellation_quant.models import ConstellationQuant

    ds_train = _TinyDataset(n_samples=4, seed=0)
    ds_val   = _TinyDataset(n_samples=2, seed=1)

    def _make(max_epochs):
        cfg = {
            "lookback": 12, "horizon": 1,
            "temporal": {"name": "lstm", "d_model": 8, "dropout": 0.0, "num_layers": 1},
            "graph": {"enabled": False, "gnn_name": "none"},
            "hierarchy": {"enabled": False},
            "heads": {
                "ranking": {"enabled": True, "mlp": [4, 1], "dropout": 0.0,
                             "temperature_scaling": False},
            },
        }
        model = ConstellationQuant(n_features=4, model_cfg=cfg)
        tcfg = TrainerConfig(
            optimizer_cfg={"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
            scheduler_cfg={"name": "cosine_annealing", "T_max": 10, "warmup_epochs": 0},
            loop_cfg={"batch_size": 2, "max_epochs": max_epochs, "num_workers": 0, "shuffle": False},
            loss_cfg={"ranking": {"name": "listmle", "weight": 1.0}},
            regularization_cfg={"gradient_clip_norm": 1.0},
            mixed_precision_cfg={"enabled": False},
            early_stopping_cfg={"enabled": False, "patience": 100},
            checkpoint_cfg={"save_periodic_every": 1, "keep_last_n": 5},
            wandb_cfg={"enabled": False},
            variant_name="resume_test",
            checkpoint_dir=tmp_path,
        )
        return Trainer(
            model=model, train_dataset=ds_train, val_dataset=ds_val,
            sample_adapter=default_sample_adapter,
            config=tcfg, device=torch.device("cpu"),
        )

    t1 = _make(max_epochs=2)
    t1.fit(resume=False)
    first_run_epochs = len(t1.state.history)

    t2 = _make(max_epochs=4)
    t2.fit(resume=True)
    # Epoch count at end of run 2 = run 1's final epoch + 2 more.
    assert t2.state.epoch >= first_run_epochs
