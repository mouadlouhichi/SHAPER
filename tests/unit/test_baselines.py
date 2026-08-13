"""Unit tests: RQ4 fair controls."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.baselines import (
    control_budget_record,
    dirichlet_reference_candidates,
    drop_lowest_loo,
    loo_derived_weights,
    random_removal,
    simplex_design_15,
)


def test_simplex_design_15_points_deterministic():
    d1 = simplex_design_15()
    d2 = simplex_design_15()
    assert d1 == d2
    assert len(d1) == 15
    for w in d1:
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert all(v >= 0 for v in w.values())
    # first point is uniform (closest to the center)
    assert d1[0]["crop"] == pytest.approx(1 / 3)
    # all points distinct
    assert len({tuple(sorted(w.items())) for w in d1}) == 15


def test_dirichlet_candidates():
    cands = dirichlet_reference_candidates(seed=901)
    assert len(cands) == 10
    for w in cands:
        assert abs(sum(w.values()) - 1.0) < 1e-12
        assert all(v > 0 for v in w.values())
    # deterministic for the same seed
    assert cands == dirichlet_reference_candidates(seed=901)


def test_loo_derived_weights():
    loo = {"crop": 0.01, "mask": 0.02, "reorder": -0.005}
    w = loo_derived_weights(loo, alpha=0.5)
    assert abs(sum(w.values()) - 1.0) < 1e-12
    assert w["reorder"] == pytest.approx(0.5 / 3)  # negative LOO -> no extra credit


def test_drop_lowest_loo():
    out = drop_lowest_loo(["crop", "mask", "reorder"], {"crop": 0.01, "mask": 0.03, "reorder": -0.02})
    assert out["removed_view"] == "reorder"
    assert out["selected_coalition"] == ["crop", "mask"]


def test_random_removal_keyed():
    a = random_removal(["crop", "mask", "reorder"], key_seed=7)
    b = random_removal(["crop", "mask", "reorder"], key_seed=7)
    assert a == b
    assert a["removed_view"] in ("crop", "mask", "reorder")


def test_control_budget_record_fields():
    budget = control_budget_record()
    for key in ("validation_budget", "training_runs", "calibration_cost",
                "final_seed_count", "training_time_seconds"):
        assert key in budget
