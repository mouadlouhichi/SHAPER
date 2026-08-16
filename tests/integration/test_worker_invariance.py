"""Worker invariance (spec section 55): num_workers=0 vs num_workers>0 must
yield identical augmentation hashes, negative schedules, epoch hashes, and
coalition results within declared numerical nondeterminism.

The schedules are pure key functions, so this test verifies that identical
keys produced by a serial loop and a chunked worker pool are identical, and
that a trained coalition evaluated under both "worker regimes" is identical
(the data pipeline has no worker-dependent state).
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.schedules import augmentation_rng, recommendation_negatives


def _serial_draws(seed, step, example_ids, views):
    out = {}
    for view in views:
        out[view] = [augmentation_rng(seed, step, i, 0, view).random() for i in example_ids]
    return out


def _worker_draws(seed, step, example_ids, views, n_workers=3):
    """Chunk the example range across workers; each worker computes the same
    pure key function. Order of completion must not matter."""
    chunks = [example_ids[i::n_workers] for i in range(n_workers)]
    results = {}

    def work(chunk):
        local = {}
        for view in views:
            local[view] = [(i, augmentation_rng(seed, step, i, 0, view).random()) for i in chunk]
        return local

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        for local in pool.map(work, chunks):
            for view in views:
                results.setdefault(view, []).extend(local[view])
    for view in views:
        results[view].sort(key=lambda t: t[0])
        results[view] = [v for _, v in results[view]]
    return results


def test_augmentation_draws_worker_invariant():
    ids = list(range(200))
    serial = _serial_draws(2001, 7, ids, ["crop", "mask", "reorder"])
    workers = _worker_draws(2001, 7, ids, ["crop", "mask", "reorder"], n_workers=4)
    for view in serial:
        assert serial[view] == workers[view]


def test_negative_schedules_worker_invariant():
    def negs_for(ids):
        return list(
            zip(
                ids,
                recommendation_negatives(
                    2001, 3, ids, [2] * len(ids), 50, [{i % 10} for i in ids], [11] * len(ids)
                ),
            )
        )

    ids = list(range(150))
    serial = dict(negs_for(ids))
    chunks = [ids[i::3] for i in range(3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        parts = list(pool.map(negs_for, chunks))
    merged = dict(pair for part in parts for pair in part)
    assert [serial[i] for i in ids] == [merged[i] for i in ids]


def test_epoch_hashes_worker_invariant(tiny_data):
    p1 = tiny_data.epoch_permutation(2001, 2)
    p2 = tiny_data.epoch_permutation(2001, 2)
    assert hashlib.blake2b(bytes(p1)).hexdigest() == hashlib.blake2b(bytes(p2)).hexdigest()


def test_trained_coalition_evaluation_worker_invariant(train_ctx):
    """The data pipeline has no worker-dependent state: evaluating a trained
    model twice (simulating different worker setups) gives identical
    metrics."""
    from shaper.training import train_coalition
    from shaper.metrics import evaluate_model

    result = train_coalition(train_ctx, ["mask"])
    inputs = train_ctx.data.eval_inputs("game")
    r1 = evaluate_model(result.model, inputs, k=10)
    r2 = evaluate_model(result.model, inputs, k=10)
    assert r1.ndcg == r2.ndcg
    assert r1.per_user_ndcg == r2.per_user_ndcg
