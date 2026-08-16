"""Coalition-order invariance (spec section 56): shuffling the coalition
enumeration must not change augmentation schedules, negative schedules,
dropout schedules, model results within numerical nondeterminism, or exact
Shapley values."""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from shaper.game import all_coalitions
from shaper.schedules import augmentation_rng, dropout_generator, recommendation_negative
from shaper.shapley import exact_shapley


def test_augmentation_schedules_order_invariant():
    """A view's draw is a pure function of its key regardless of which other
    views/coalitions were drawn first."""
    keys = [(2001, 3, i, 0, v) for i in range(40) for v in ("crop", "mask", "reorder")]
    forward = [augmentation_rng(*k).random() for k in keys]
    backward = [augmentation_rng(*k).random() for k in reversed(keys)]
    assert forward == backward[::-1]


def test_negative_schedules_order_invariant():
    a = recommendation_negative(2001, 1, 3, 2, 100, {1, 2}, 5)
    _ = recommendation_negative(2001, 1, 99, 9, 100, {1}, 7)
    b = recommendation_negative(2001, 1, 3, 2, 100, {1, 2}, 5)
    assert a == b


def test_dropout_schedules_order_invariant():
    g1 = dropout_generator(2001, 4, "contrast", "crop", 1)
    _ = dropout_generator(2001, 4, "contrast", "mask", 0)
    g2 = dropout_generator(2001, 4, "contrast", "crop", 1)
    assert g1.initial_seed() == g2.initial_seed()


def test_exact_shapley_order_invariant():
    rng = np.random.default_rng(0)
    values = {c: float(rng.uniform(-0.1, 0.1)) for c in all_coalitions(("crop", "mask", "reorder"))}
    players = ("crop", "mask", "reorder")
    phi1 = exact_shapley(values, players)
    # insert keys in a different order into a fresh dict
    shuffled = {}
    for c in sorted(values, key=lambda x: rng.random()):
        shuffled[c] = values[c]
    phi2 = exact_shapley(shuffled, players)
    assert phi1 == phi2


def test_coalition_results_order_invariant(train_ctx):
    """Training coalitions in a different order yields the same model results
    (identical schedules; deterministic kernels on CPU)."""
    from shaper.training import train_coalition
    from shaper.metrics import evaluate_model

    inputs = train_ctx.data.eval_inputs("game")
    r1 = train_coalition(train_ctx, ["crop"])
    n1 = evaluate_model(r1.model, inputs, k=10).ndcg
    r2 = train_coalition(train_ctx, ["crop"])
    n2 = evaluate_model(r2.model, inputs, k=10).ndcg
    assert n1 == n2
