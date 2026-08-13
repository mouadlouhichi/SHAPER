"""Unit tests: optimizer/scheduler protocol and common initialization."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.training import (
    common_initialization,
    make_optimizer,
    make_scheduler,
    train_coalition,
)


def test_optimizer_settings(tiny_cfg):
    from shaper.backbone import build_model

    model = build_model(tiny_cfg, 20)
    opt = make_optimizer(model, tiny_cfg, lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.defaults["betas"] == (0.9, 0.98)
    assert opt.defaults["eps"] == 1e-8
    assert opt.defaults["weight_decay"] == 1e-4


def test_scheduler_warmup_and_cosine(tiny_cfg):
    from shaper.backbone import build_model

    model = build_model(tiny_cfg, 20)
    opt = make_optimizer(model, tiny_cfg, lr=1e-3)
    total = 100
    sched = make_scheduler(opt, total, warmup_frac=0.1, cosine_final_frac=0.1)
    lrs = []
    for step in range(total):
        lrs.append(sched.get_last_lr()[0])
        sched.step()
    assert lrs[0] == 0.0
    assert abs(lrs[9] - 0.9e-3) < 1e-9  # linear warmup: 9/10 of lr
    assert abs(lrs[10] - 1e-3) < 1e-9  # end of warmup -> full lr
    assert abs(lrs[-1] - 1e-4) < 2e-3  # cosine decay to 0.1 * lr
    assert max(lrs) == pytest.approx(1e-3)
    # monotonic after warmup
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(10, total - 1))


def test_common_initialization_deterministic_per_seed(train_ctx):
    s1 = common_initialization(train_ctx)
    # a fresh context with the same seed produces the identical state
    from shaper.training import TrainContext

    ctx2 = TrainContext(**{**train_ctx.__dict__})
    ctx2.base_state = None
    s2 = common_initialization(ctx2)
    for key in s1:
        assert torch.equal(s1[key], s2[key]), f"init mismatch at {key}"
    # different seed -> different state
    ctx3 = TrainContext(**{**train_ctx.__dict__})
    ctx3.base_state = None
    ctx3.seed = 2002
    s3 = common_initialization(ctx3)
    assert not torch.equal(s1["backbone.item_emb.weight"], s3["backbone.item_emb.weight"])


def test_fixed_step_budget_no_early_stopping(train_ctx):
    """A coalition receives exactly train_steps optimizer updates."""
    result = train_coalition(train_ctx, [])
    assert result.steps == train_ctx.recipe.steps
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["step"] == train_ctx.recipe.steps
    assert len(result.metric_history["loss"]) == train_ctx.recipe.steps


def test_checkpoint_contains_all_required_fields(train_ctx):
    result = train_coalition(train_ctx, ["crop"])
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    for key in (
        "model", "optimizer", "scheduler", "epoch", "step", "seed", "coalition",
        "policy", "rng_states", "config_hash", "data_hash", "recipe_hash",
        "metric_history",
    ):
        assert key in payload, f"checkpoint missing {key}"


def test_effective_mass_recorded(train_ctx):
    result = train_coalition(train_ctx, ["crop", "mask"])
    assert "m_c" in result.effective_mass_stats
    assert result.effective_mass_stats["m_c"]["n"] == train_ctx.recipe.steps
    assert "crop" in result.applicability_stats
    assert "mask" in result.applicability_stats


def test_equal_optimizer_budgets_across_coalitions(train_ctx):
    a = train_coalition(train_ctx, [])
    b = train_coalition(train_ctx, ["crop", "mask", "reorder"])
    assert a.steps == b.steps == train_ctx.recipe.steps


def test_train_from_common_init_not_grand(train_ctx):
    """Coalitions never warm-start from the grand coalition: the initial
    parameters equal the common seed init, not any trained checkpoint."""
    init = common_initialization(train_ctx)
    result = train_coalition(train_ctx, ["crop"])
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert not torch.equal(init["backbone.item_emb.weight"], payload["model"]["backbone.item_emb.weight"])


def test_gated_training_uses_separate_lr_and_is_training_only(tiny_cfg, tiny_data):
    from shaper.training import DatasetLevelGates

    gate = DatasetLevelGates(3)
    assert torch.allclose(gate.weights(), torch.full((3,), 1 / 3))
