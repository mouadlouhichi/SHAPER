"""Unit tests: frozen behavioural segments and the studentized permutation
tests (spec A.11)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from shaper.segments import (
    label_permutation_test,
    omnibus_profile_statistic,
    segment_means,
    segment_ses,
    studentized_mask_trend,
)


def _phi_by_seed(quartiles, n_seeds=3, effect=0.004, noise=0.02, seed=0):
    """Synthetic per-user Shapley vectors with an increasing mask trend."""
    rng = np.random.default_rng(seed)
    per_seed = []
    for _ in range(n_seeds):
        phi = {}
        for p in ("crop", "mask", "reorder"):
            base = rng.normal(0.0, noise, len(quartiles))
            if p == "mask":
                base = base + effect * (quartiles - 2.5)
            phi[p] = base
        per_seed.append(phi)
    return per_seed


def test_segment_means_and_ses():
    rng = np.random.default_rng(0)
    quartiles = rng.integers(1, 5, size=200)
    phi = {"mask": rng.normal(0, 0.1, 200)}
    mu = segment_means(phi, quartiles)
    se = segment_ses(phi, quartiles)
    assert set(mu.keys()) == {"1", "2", "3", "4"}
    assert all(se[str(m)]["mask"] > 0 for m in range(1, 5))


def test_studentized_mask_trend():
    mu = {str(m): {"mask": float(m), "crop": 0.0, "reorder": 0.0} for m in range(1, 5)}
    # T = sum (m-2.5)*m = (-1.5)(1)+(-0.5)(2)+(0.5)(3)+(1.5)(4) = -1.5-1+1.5+6 = 5
    assert studentized_mask_trend(mu) == pytest.approx(5.0)


def test_omnibus_statistic_positive():
    mu = {str(m): {"mask": float(m), "crop": 0.0, "reorder": 0.0} for m in range(1, 5)}
    se = {str(m): {"mask": 0.5, "crop": 0.5, "reorder": 0.5} for m in range(1, 5)}
    assert omnibus_profile_statistic(mu, se) > 0


def test_label_permutation_test_detects_trend():
    rng = np.random.default_rng(42)
    quartiles = rng.integers(1, 5, size=120)
    per_seed = _phi_by_seed(quartiles, n_seeds=3, effect=0.05, noise=0.01, seed=0)
    out = label_permutation_test(per_seed, quartiles, n_permutations=200, seed=0)
    assert out["n_permutations"] == 200
    assert out["T_mask_observed"] > 0
    assert out["p_mask_one_sided"] < 0.05  # strong injected trend
    assert 0.0 <= out["p_all"] <= 1.0


def test_label_permutation_test_null_uniform():
    rng = np.random.default_rng(43)
    quartiles = rng.integers(1, 5, size=120)
    per_seed = _phi_by_seed(quartiles, n_seeds=3, effect=0.0, noise=0.05, seed=1)
    out = label_permutation_test(per_seed, quartiles, n_permutations=200, seed=1)
    assert out["p_mask_one_sided"] >= 0.01  # no trend -> roughly uniform p


def test_same_permuted_map_across_seeds():
    """One permuted user map is applied to ALL seed records (the statistic is
    computed from the mean over seeds of per-user vectors)."""
    rng = np.random.default_rng(5)
    quartiles = rng.integers(1, 5, size=60)
    per_seed = _phi_by_seed(quartiles, n_seeds=3, effect=0.02, noise=0.05, seed=2)
    out1 = label_permutation_test(per_seed, quartiles, n_permutations=50, seed=99)
    out2 = label_permutation_test(per_seed, quartiles, n_permutations=50, seed=99)
    assert out1["T_mask_observed"] == out2["T_mask_observed"]
    assert out1["p_mask_one_sided"] == out2["p_mask_one_sided"]
