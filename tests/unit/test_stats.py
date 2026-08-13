"""Unit tests: seed-level statistics, Holm, descriptive user analyses, and
the Appendix-J planning formula."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
from scipy import stats as sps

from shaper.stats import (
    bootstrap_ci,
    cliffs_delta,
    half_width_of,
    hierarchical_bootstrap_ci,
    holm_adjust,
    paired_seed_effects,
    planning_half_width,
    rank_biserial,
    seed_summary,
    wilcoxon_paired,
)


def test_seed_summary_ci_matches_manual():
    vals = [0.010, 0.012, 0.011, 0.013, 0.009]
    out = seed_summary(vals)
    expected_half = sps.t.ppf(0.975, 4) * np.std(vals, ddof=1) / np.sqrt(5)
    assert out["ci"][0] == pytest.approx(np.mean(vals) - expected_half)
    assert out["ci"][1] == pytest.approx(np.mean(vals) + expected_half)
    assert out["n_seeds"] == 5


def test_paired_seed_effects_uses_seeds_as_unit():
    a = [0.10, 0.11, 0.12, 0.13, 0.14]
    b = [0.08, 0.08, 0.08, 0.08, 0.08]
    out = paired_seed_effects(a, b)
    assert out["n_seeds"] == 5
    assert out["mean_diff"] == pytest.approx(0.04)
    assert out["p_value"] < 0.05


def test_holm_adjust_known_example():
    # classic example: p = [0.01, 0.04, 0.03]
    adj = holm_adjust([0.01, 0.04, 0.03])
    # sorted: 0.01, 0.03, 0.04 -> x3, x2, x1 -> [0.03, 0.06, 0.04], made
    # monotone -> adjusted = [0.03, 0.06, 0.06]
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    assert all(0 <= p <= 1 for p in adj)


def test_wilcoxon_and_effect_sizes():
    rng = np.random.default_rng(0)
    a = rng.normal(0.5, 0.1, 500)
    b = rng.normal(0.45, 0.1, 500)
    w = wilcoxon_paired(a, b)
    assert w["p_value"] < 1e-3
    rb = rank_biserial(a, b)
    assert rb > 0
    cd = cliffs_delta(a, b)
    assert cd > 0.2


def test_bootstrap_ci_covers_mean():
    rng = np.random.default_rng(1)
    vals = rng.normal(0.0, 1.0, 200)
    out = bootstrap_ci(vals, n_bootstrap=500, seed=0)
    assert out["ci"][0] <= out["mean"] <= out["ci"][1]


def test_planning_half_width_formula():
    # h = t_(.975, S-1) * sqrt(sigma_seed^2/S + sigma_user^2/N_game)
    S, N = 5, 3600
    h = planning_half_width(S, 0.001, 0.05, N)
    t = sps.t.ppf(0.975, S - 1)
    expected = t * np.sqrt(0.001**2 / S + 0.05**2 / N)
    assert h == pytest.approx(expected)
    # the paper's provisional illustration: ~0.0026
    assert 0.0025 < h < 0.0027


def test_hierarchical_bootstrap():
    seed_values = [
        list(np.random.default_rng(i).normal(0.01, 0.05, 50)) for i in range(5)
    ]
    out = hierarchical_bootstrap_ci(seed_values, n_bootstrap=300, seed=0)
    assert out["ci"][0] <= out["ci"][1]


def test_half_width_of():
    assert half_width_of((0.0, 0.006)) == pytest.approx(0.003)
