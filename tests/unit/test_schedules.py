"""Unit tests: keyed deterministic random schedules (spec A.4/A.5)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.schedules import (
    augmentation_rng,
    dropout_generator,
    epoch_permutation,
    keyed_forward,
    recommendation_negatives,
    recommendation_negative,
)


def test_augmentation_rng_deterministic_and_keyed():
    r1 = augmentation_rng(2001, 7, 42, 0, "crop")
    r2 = augmentation_rng(2001, 7, 42, 0, "crop")
    assert r1.getstate() == r2.getstate()
    # different keys -> different draws (almost surely)
    r3 = augmentation_rng(2001, 7, 43, 0, "crop")
    assert r1.random() != r3.random()


def test_negative_schedule_validity_and_keys():
    n_items = 20
    neg = recommendation_negative(
        2001, 3, global_user_id=5, target_position=2, n_items=n_items,
        exclude_set={1, 2, 3}, positive=4,
    )
    assert 1 <= neg <= n_items
    assert neg != 4
    assert neg not in {1, 2, 3}
    # identical for the same key, independent of call order/context
    neg2 = recommendation_negative(
        2001, 3, global_user_id=5, target_position=2, n_items=n_items,
        exclude_set={1, 2, 3}, positive=4,
    )
    assert neg == neg2


def test_negative_batch_identity():
    negs = recommendation_negatives(
        2001, 0, [0, 1, 2], [1, 1, 1], 10,
        [{1, 2}, {1, 2}, {1, 2}], [3, 3, 3],
    )
    assert len(negs) == 3
    assert all(1 <= n <= 10 and n != 3 for n in negs)


def test_epoch_permutation_every_user_once():
    perm = epoch_permutation(2001, 0, "hash123", 100)
    assert sorted(perm) == list(range(100))
    perm2 = epoch_permutation(2001, 0, "hash123", 100)
    assert perm == perm2
    perm3 = epoch_permutation(2001, 1, "hash123", 100)
    assert perm != perm3  # new keyed permutation per epoch (a.s.)


def test_epoch_permutation_depends_on_dataset_hash():
    a = epoch_permutation(2001, 0, "hash-a", 64)
    b = epoch_permutation(2001, 0, "hash-b", 64)
    assert a != b


def test_dropout_generator_keyed():
    g1 = dropout_generator(2001, 5, "rec", "rec", 0)
    g2 = dropout_generator(2001, 5, "rec", "rec", 0)
    assert g1.initial_seed() == g2.initial_seed()
    g3 = dropout_generator(2001, 5, "contrast", "crop", 1)
    assert g3.initial_seed() != g1.initial_seed()


def test_keyed_forward_is_pure_function_of_key():
    torch.manual_seed(123)
    layer = torch.nn.Dropout(0.5)
    x = torch.ones(64)
    layer.train()
    g = dropout_generator(2001, 0, "rec", "rec", 0)
    y1 = keyed_forward(lambda t: layer(t), g, x)
    y2 = keyed_forward(lambda t: layer(t), g, x)
    assert torch.equal(y1, y2)


def test_schedules_independent_of_call_order():
    """The same key yields the same draw regardless of which other draws
    happened in between (pure function of the key)."""
    before = augmentation_rng(2001, 1, 1, 0, "mask").random()
    _ = augmentation_rng(2001, 1, 999, 0, "crop").random()
    after = augmentation_rng(2001, 1, 1, 0, "mask").random()
    assert before == after
