"""Unit tests: augmentation views and their invariants (spec A.4)."""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.augment import (
    augment_crop,
    augment_mask,
    augment_reorder,
    effective_mass,
    view_diagnostics,
)


def test_crop_reference_semantics():
    seq = list(range(10))
    rng = random.Random(0)
    out, changed = augment_crop(seq, rng, eta=0.6)
    assert changed
    assert len(out) == 6
    assert out == seq[out[0] : out[0] + 6] or out[0] == seq[0]  # contiguous


def test_crop_keeps_at_least_two_and_at_most_n_minus_one():
    for n in range(3, 12):
        seq = list(range(n))
        for _ in range(50):
            from shaper.provenance import key_int

            rng = random.Random(key_int(n, _, salt="crop-bounds"))
            out, changed = augment_crop(seq, rng)
            if changed:
                assert 2 <= len(out) <= n - 1
            # contiguity
            start = out[0]
            assert out == list(range(start, start + len(out)))


def test_crop_noop_below_three():
    seq = [1, 2]
    rng = random.Random(0)
    out, changed = augment_crop(seq, rng)
    assert not changed and out == seq


def test_crop_start_support_uniform_over_inclusive_support():
    """Unit test: crop starts are uniform over the inclusive support
    0..n-keep (spec A.4)."""
    n, eta = 10, 0.5
    keep = min(n - 1, max(2, int(n * eta)))  # 5
    counts = {s: 0 for s in range(n - keep + 1)}
    for draw in range(12000):
        from shaper.provenance import key_int

        rng = random.Random(key_int(draw, "crop-start", salt="test"))
        out, changed = augment_crop(list(range(n)), rng, eta=eta)
        assert changed
        counts[out[0]] += 1
    total = sum(counts.values())
    for s, c in counts.items():
        frac = c / total
        assert 0.15 < frac < 0.185, f"start {s} not uniform: {frac:.3f}"


def test_mask_semantics_and_mask_id():
    seq = [3, 4, 5, 6, 7, 8]
    rng = random.Random(1)
    out, changed = augment_mask(seq, rng, mask_id=99, gamma=0.2)
    assert changed
    n_masked = sum(1 for x in out if x == 99)
    assert n_masked == 1  # round(6 * 0.2) = 1
    assert len(out) == len(seq)
    # mask_id required and valid
    with pytest.raises(ValueError):
        from shaper.augment import apply_view

        apply_view("mask", seq, rng, mask_id=None)


def test_mask_empty_sequence_noop():
    out, changed = augment_mask([], random.Random(0), mask_id=5)
    assert not changed and out == []


def test_reorder_nonidentity_and_min_span():
    seq = list(range(10))
    for draw in range(100):
        from shaper.provenance import key_int

        rng = random.Random(key_int(draw, 99, salt="reorder-test"))
        out, changed = augment_reorder(seq, rng, beta=0.2)
        if changed:
            assert out != seq
            # exactly one contiguous span of <= span differs
            span = min(10, max(2, int(round(10 * 0.2))))
            diffs = [i for i, (a, b) in enumerate(zip(out, seq)) if a != b]
            assert diffs == list(range(diffs[0], diffs[-1] + 1))
            assert len(diffs) <= span
            assert len(diffs) >= 2
            assert sorted(out) == sorted(seq)


def test_reorder_identity_retried_and_degenerate_noop():
    # a fully constant sequence cannot permute non-trivially (identity retried)
    seq = [7, 7, 7, 7, 7, 7]
    rng = random.Random(3)
    out, changed = augment_reorder(seq, rng, beta=0.5)
    assert not changed
    # a constant span keeps the sequence unchanged too
    out2, changed2 = augment_reorder([1, 1, 1, 1], random.Random(5), beta=0.5)
    assert not changed2
    # single item
    out, changed = augment_reorder([4], random.Random(0), beta=0.5)
    assert not changed


def test_padding_never_touched():
    """Views operate on the valid sequence; the training path strips padding
    before drawing and re-pads afterwards (see training._apply_views). Here
    we verify the view functions never mutate their input."""
    seq = [1, 2, 3, 4, 5]
    for fn in (
        lambda rng: augment_crop(seq, rng),
        lambda rng: augment_mask(seq, rng, mask_id=50),
        lambda rng: augment_reorder(seq, rng),
    ):
        rng = random.Random(7)
        out, _ = fn(rng)
        assert seq == [1, 2, 3, 4, 5]


def test_view_diagnostics_shape():
    seqs = [list(range(i)) for i in range(3, 12)]
    diag = view_diagnostics(seqs, n_draws=10, seed=0, mask_id=99)
    for view in ("crop", "mask", "reorder"):
        assert 0.0 <= diag["views"][view]["applicability_rate"] <= 1.0
        assert diag["views"][view]["draws"] == 90


def test_effective_mass_policies():
    a = [1.0, 0.5]
    assert effective_mass(a, ["crop", "mask"], "game_a") == pytest.approx(0.75)
    assert effective_mass(a, ["crop", "mask"], "game_b", K=3) == pytest.approx(0.5)
    assert effective_mass([], [], "game_a") == 0.0
    assert effective_mass([1.0], ["crop"], "game_b", K=3) == pytest.approx(1 / 3)
