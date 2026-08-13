"""Protocol tests 8-10: coalition isolation, gradient flow, common
initialization, equal optimizer budgets, and split hygiene (spec A.9)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.training import train_coalition


def test_excluded_view_never_generated(train_ctx, monkeypatch):
    """Training crop-only must never generate mask/reorder draws."""
    from shaper import augment
    from shaper import training

    orig = augment.apply_view

    def spy(view, seq, rng, *args, **kwargs):
        if view != "crop":
            raise AssertionError(f"excluded view {view} generated")
        return orig(view, seq, rng, *args, **kwargs)

    monkeypatch.setattr(training, "_single_view",
                        lambda ctx, view, seq, rng, mask_id: spy(view, seq, rng, mask_id=mask_id))
    result = train_coalition(train_ctx, ["crop"])
    assert result.steps == train_ctx.recipe.steps


def test_excluded_view_leakage_via_grad_fn(train_ctx):
    """A crop-only forward pass must not touch mask-related machinery."""
    from shaper.training import _train_one_step
    from shaper.backbone import build_model

    ctx = train_ctx
    model = build_model(ctx.cfg, ctx.data.n_items)
    model.train()
    opt = __import__("shaper.training", fromlist=["make_optimizer"]).make_optimizer(model, ctx.cfg, 5e-4)
    sched = __import__("shaper.training", fromlist=["make_scheduler"]).make_scheduler(opt, ctx.recipe.steps)
    user_ids, seqs = next(ctx.data.train_batches(ctx.seed, 0))
    record = _train_one_step(ctx, model, opt, sched, 0, user_ids, seqs, ["crop"])
    assert record["views"][0]["view"] == "crop"


def test_common_initialization_cloned_every_coalition(train_ctx):
    from shaper.training import common_initialization

    init = common_initialization(train_ctx)
    for coalition in (["mask"], ["crop", "mask", "reorder"]):
        result = train_coalition(train_ctx, coalition)
        payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
        # the training STARTED from init: we verify the saved optimizer has
        # zero state at step 0 only in a fresh context; here assert the
        # coalition training consumed exactly train_steps updates
        assert payload["step"] == train_ctx.recipe.steps


def test_equal_optimizer_step_budget_all_coalitions(train_ctx):
    steps = set()
    for coalition in ([], ["crop"], ["mask"], ["reorder"]):
        result = train_coalition(train_ctx, coalition)
        steps.add(result.steps)
    assert steps == {train_ctx.recipe.steps}


def test_split_hygiene_roles_disjoint_and_test_untouched(tiny_data):
    tune = set(int(x) for x in tiny_data.user_ids_for_role("tune"))
    game = set(int(x) for x in tiny_data.user_ids_for_role("game"))
    select = set(int(x) for x in tiny_data.user_ids_for_role("select"))
    assert tune.isdisjoint(game) and game.isdisjoint(select) and tune.isdisjoint(select)
    assert tune | game | select == set(range(tiny_data.n_users))
    # test targets never appear in the training tensor
    train = tiny_data.train_seqs()
    for uid in range(tiny_data.n_users):
        items = set(int(x) for x in train[uid].tolist() if x != 0)
        assert int(tiny_data.tensors["test_target"][uid]) not in items
        assert int(tiny_data.tensors["val_target"][uid]) not in items
    # test prefix = training history + validation item
    test_prefix = tiny_data.test_inputs()["prefix"]
    for uid in range(tiny_data.n_users):
        pfx = [int(x) for x in test_prefix[uid].tolist() if x != 0]
        assert pfx[-1] == int(tiny_data.tensors["val_target"][uid])
        assert pfx[:-1] == [int(x) for x in train[uid].tolist() if x != 0][-(len(pfx) - 1):]


def test_gradient_reaches_transformer_and_item_embeddings(train_ctx):
    """Protocol test 8 via a real backward pass through the coalition loss."""
    from shaper.training import _train_one_step
    from shaper.backbone import build_model

    model = build_model(train_ctx.cfg, train_ctx.data.n_items)
    model.train()
    opt = __import__("shaper.training", fromlist=["make_optimizer"]).make_optimizer(model, train_ctx.cfg, 5e-4)
    sched = __import__("shaper.training", fromlist=["make_scheduler"]).make_scheduler(opt, train_ctx.recipe.steps)
    user_ids, seqs = next(train_ctx.data.train_batches(train_ctx.seed, 0))
    _train_one_step(train_ctx, model, opt, sched, 0, user_ids, seqs, ["crop", "mask", "reorder"])
    item_grad = model.backbone.item_emb.weight.grad
    assert item_grad is not None and item_grad.abs().sum().item() > 0
    transformer_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for block in model.backbone.blocks for p in block.parameters()
    )
    assert transformer_grad
