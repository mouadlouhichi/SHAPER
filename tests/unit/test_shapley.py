"""Unit tests: exact Shapley allocation (efficiency, symmetry, dummy, known
games, the locked redundancy counterexample, LOO, per-user linearity)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from shaper.game import grand_loo, per_user_values
from shaper.interactions import perfect_substitutability_game, rejected_half_pair_removal
from shaper.shapley import (
    efficiency_residual,
    exact_shapley,
    per_user_shapley,
    shapley_of_mean_table,
    shapley_weight,
)

PLAYERS = ("crop", "mask", "reorder")


def random_complete_table(seed=0):
    rng = np.random.default_rng(seed)
    values = {}
    for r in range(4):
        for combo in __import__("itertools").combinations(PLAYERS, r):
            values[tuple(sorted(combo))] = float(rng.uniform(-0.1, 0.1))
    values[()] = 0.0
    return values


def test_shapley_weights_normalize():
    for K in (3, 4):
        total = sum(
            shapley_weight(s, K) * math.comb(K - 1, s) for s in range(K)
        )
        assert abs(total - 1.0) < 1e-12


def test_efficiency_on_random_tables():
    for seed in range(5):
        table = random_complete_table(seed)
        phi = exact_shapley(table, PLAYERS)
        residual = efficiency_residual(phi, table[tuple(PLAYERS)])
        assert residual < 1e-12, f"efficiency violated: residual {residual}"


def test_empty_coalition_zero_after_baseline():
    table = random_complete_table(0)
    assert table[()] == 0.0


def test_symmetry():
    # symmetric game: v(C) depends only on |C|
    values = {}
    for r in range(4):
        for combo in __import__("itertools").combinations(PLAYERS, r):
            values[tuple(sorted(combo))] = float(r * r)
    phi = exact_shapley(values, PLAYERS)
    assert len(set(phi.values())) == 1
    # known value for v(C)=|C|^2, K=3: phi_p = 3
    assert all(abs(v - 3.0) < 1e-12 for v in phi.values())


def test_dummy_player_gets_zero():
    players = PLAYERS + ("dummy",)
    base = random_complete_table(1)
    values = {}
    for combo, v in base.items():
        values[combo] = v
    for r in range(4):
        for combo in __import__("itertools").combinations(PLAYERS, r):
            c = tuple(sorted(combo))
            values[tuple(sorted(c + ("dummy",)))] = base[c]
    phi = exact_shapley(values, players)
    assert abs(phi["dummy"]) < 1e-12


def test_known_linear_game():
    # v(C) = sum of singleton utilities -> phi_p = u_p
    u = {"crop": 0.1, "mask": -0.05, "reorder": 0.2}
    values = {}
    for r in range(4):
        for combo in __import__("itertools").combinations(PLAYERS, r):
            values[tuple(sorted(combo))] = sum(u[p] for p in combo)
    phi = exact_shapley(values, PLAYERS)
    for p, expected in u.items():
        assert abs(phi[p] - expected) < 1e-12


def test_locked_redundancy_counterexample():
    """Paper Appendix A: perfect substitutes have zero grand-LOO, equal
    nonzero Shapley, and the half-pair-removal formula is wrong."""
    v = perfect_substitutability_game()
    players = ("p1", "p2", "p3")
    loo = grand_loo(v, players)
    assert abs(loo["p1"]) < 1e-12
    assert abs(loo["p2"]) < 1e-12
    phi = exact_shapley(v, players)
    assert abs(phi["p1"] - phi["p2"]) < 1e-12
    assert abs(phi["p1"] - 0.041666666666666664) < 1e-9
    assert abs(phi["p3"] - 0.016666666666666666) < 1e-9
    rejected = rejected_half_pair_removal(v, "p1", "p2", "p3")
    assert abs(rejected - 0.025) < 1e-12
    assert abs(rejected - phi["p1"]) > 1e-3  # the formula is INVALID here


def test_loo_never_equated_with_shapley():
    v = perfect_substitutability_game()
    loo = grand_loo(v, ("p1", "p2", "p3"))
    phi = exact_shapley(v, ("p1", "p2", "p3"))
    assert loo["p1"] != phi["p1"]  # zero vs nonzero in this locked game


def test_shapley_of_mean_equals_mean_of_shapley():
    tables = [random_complete_table(s) for s in range(3)]
    phi_of_mean = shapley_of_mean_table(tables, PLAYERS)
    mean_phi = {p: float(np.mean([exact_shapley(t, PLAYERS)[p] for t in tables])) for p in PLAYERS}
    for p in PLAYERS:
        assert abs(phi_of_mean[p] - mean_phi[p]) < 1e-12


def test_per_user_linear_decomposition():
    rng = np.random.default_rng(3)
    n_users = 17
    tables = []
    for _ in range(2):
        t = {}
        for r in range(4):
            for combo in __import__("itertools").combinations(PLAYERS, r):
                t[tuple(sorted(combo))] = rng.uniform(-0.05, 0.05, size=n_users)
        tables.append(t)
    phi_u = per_user_shapley(tables, PLAYERS)
    mean_phi = {p: float(phi_u[p].mean()) for p in PLAYERS}
    # per_user_shapley ACCUMULATES across the supplied tables, so the mean
    # over users equals the Shapley value of the summed (not averaged) table
    sum_table = {c: (tables[0][c] + tables[1][c]) for c in tables[0]}
    aggregate = exact_shapley({c: float(v.mean()) for c, v in sum_table.items()}, PLAYERS)
    for p in PLAYERS:
        assert abs(mean_phi[p] - aggregate[p]) < 1e-12


def test_incomplete_table_raises():
    table = random_complete_table(0)
    del table[("crop",)]
    with pytest.raises(ValueError, match="incomplete table"):
        exact_shapley(table, PLAYERS)
