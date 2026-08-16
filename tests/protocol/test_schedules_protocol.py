"""Protocol tests 7 (order invariance), 16 (epoch schedule), 17 (negatives),
and 18 (effective-mass logging)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import hashlib

import numpy as np
import pytest

from shaper.schedules import (
    augmentation_rng,
    epoch_permutation,
    recommendation_negatives,
)


def test_epoch_schedule_every_user_once_per_epoch(tiny_data):
    seen = set()
    for epoch in range(3):
        perm = tiny_data.epoch_permutation(2001, epoch)
        assert sorted(perm) == list(range(tiny_data.n_users))
        key = hashlib.blake2b(bytes(perm)).hexdigest()
        assert key not in seen  # a NEW keyed permutation per epoch
        seen.add(key)
    # identical across calls (deterministic)
    assert tiny_data.epoch_permutation(2001, 0).tolist() == tiny_data.epoch_permutation(2001, 0).tolist()


def test_epoch_schedule_identity_across_workers_and_batching(tiny_data):
    """Worker count / batch size must not change epoch or negative
    schedules: iterating with different batch sizes yields the same global
    order."""
    b1 = list(tiny_data.train_batches(2001, 0, batch_size=7))
    b2 = list(tiny_data.train_batches(2001, 0, batch_size=13))
    order1 = np.concatenate([u.numpy() for u, _ in b1])
    order2 = np.concatenate([u.numpy() for u, _ in b2])
    assert order1.tolist() == order2.tolist()


def test_recommendation_negatives_hash_identical_across_coalitions():
    """Same (seed, step, user, position) key -> identical negatives no matter
    which coalition context computes them."""
    n_items = 30
    a = recommendation_negatives(2001, 5, [7, 8], [2, 3], n_items, [{1}, {1}], [4, 5])
    b = recommendation_negatives(2001, 5, [7, 8], [2, 3], n_items, [{1}, {1}], [4, 5])
    assert a == b
    for neg, pos in zip(a, [4, 5]):
        assert 1 <= neg <= n_items and neg != pos and neg != 1


def test_augmentation_hash_identity_across_enumeration_orders():
    """Coalition enumeration order cannot change a view draw."""
    draws_forward = [augmentation_rng(2001, 3, i, 0, "crop").random() for i in range(50)]
    draws_backward = [augmentation_rng(2001, 3, i, 0, "crop").random() for i in range(49, -1, -1)]
    assert draws_forward == draws_backward[::-1]


def test_effective_mass_logging_identity():
    """Synthetic applicability tensors produce exactly sum(a)/|C| in Game A
    and sum(a)/K in Game B (protocol test 18)."""
    from shaper.augment import effective_mass

    a = [0.9, 0.8, 0.0]
    assert effective_mass(a, ["crop", "mask", "reorder"], "game_a") == pytest.approx(1.7 / 3)
    assert effective_mass(a, ["crop", "mask", "reorder"], "game_b", K=3) == pytest.approx(1.7 / 3)
    assert effective_mass(a, ["crop", "mask"], "game_a") == pytest.approx(1.7 / 2)
    assert effective_mass(a, ["crop", "mask"], "game_b", K=3) == pytest.approx(1.7 / 3)


def test_noop_budget_accounting_identity():
    """Replacing one view's entire batch by no-ops sets that view loss to zero
    without changing the |C| or K denominator (protocol test 14)."""
    from shaper.contrast import coalition_cl_loss
    import torch

    l = [torch.tensor(1.0), torch.tensor(0.0), torch.tensor(2.0)]
    coalition = ["crop", "mask", "reorder"]
    a = coalition_cl_loss(l, coalition, "game_a")
    b = coalition_cl_loss(l, coalition, "game_b", K=3)
    assert a.item() == pytest.approx(3.0 / 3)
    assert b.item() == pytest.approx(3.0 / 3)
