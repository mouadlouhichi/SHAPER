"""Unit tests: Grabisch-Roubens interactions, substitutability diagnostics,
and the declaration rule."""

from __future__ import annotations

import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from shaper.game import epsilon_pq
from shaper.interactions import (
    all_pair_interactions,
    declare_interaction,
    grabisch_roubens_interaction,
    interaction_weight,
    perfect_substitutability_game,
)

PLAYERS = ("crop", "mask", "reorder")


def table(fn):
    out = {}
    for r in range(4):
        for combo in itertools.combinations(PLAYERS, r):
            out[tuple(sorted(combo))] = fn(combo)
    return out


def test_interaction_weights_normalize():
    K = 3
    # for a pair, S ranges over subsets of the remaining K-2 players
    total = sum(interaction_weight(s, K) * math_comb(K - 2, s) for s in range(K - 1))
    assert abs(total - 1.0) < 1e-12


def math_comb(n, k):
    import math

    return math.comb(n, k)


def test_additive_game_zero_interactions():
    v = table(lambda c: sum({"crop": 0.1, "mask": 0.2, "reorder": 0.3}[p] for p in c))
    for p, q in itertools.combinations(PLAYERS, 2):
        assert abs(grabisch_roubens_interaction(v, PLAYERS, p, q)) < 1e-12


def test_supermodular_game_positive_interaction():
    v = table(lambda c: len(c) ** 2)
    # K=3: contexts S in {empty, {third}}; Delta = 2 each; weight 1/2 -> I = 2.0
    for p, q in itertools.combinations(PLAYERS, 2):
        assert abs(grabisch_roubens_interaction(v, PLAYERS, p, q) - 2.0) < 1e-12


def test_substitutable_game_negative_interaction():
    v = table(lambda c: min(len(c), 1))
    # Delta(empty) = 1 - 1 - 1 + 0 = -1; weight 1/2 -> -0.5
    for p, q in itertools.combinations(PLAYERS, 2):
        assert abs(grabisch_roubens_interaction(v, PLAYERS, p, q) + 0.5) < 1e-12


def test_incomplete_k4_table_forbidden():
    """Exact K=4 interactions are FORBIDDEN from an incomplete MC table."""
    partial = {
        (): 0.0,
        ("a",): 0.1, ("b",): 0.1, ("c",): 0.0, ("d",): 0.05,
        ("a", "b", "c", "d"): 0.3,
    }
    with pytest.raises(ValueError, match="incomplete table"):
        grabisch_roubens_interaction(partial, ("a", "b", "c", "d"), "a", "b")


def test_declaration_rule():
    assert declare_interaction([-0.01, -0.011, -0.009], 0.003)["declaration"] == "directional"
    assert declare_interaction([-0.01, -0.011, -0.009], 0.003)["direction"].startswith("negative")
    # CI excludes zero but magnitude below threshold -> INCONCLUSIVE
    assert declare_interaction([-0.001, -0.0012, -0.0008], 0.003)["declaration"] == "INCONCLUSIVE"
    # CI includes zero -> INCONCLUSIVE even if mean is large
    assert declare_interaction([-0.05, 0.05], 0.003)["declaration"] == "INCONCLUSIVE"


def test_epsilon_pq_perfect_substitutes_zero():
    v = perfect_substitutability_game()
    eps = epsilon_pq(v, ("p1", "p2", "p3"), "p1", "p2")
    assert abs(eps) < 1e-12
    eps2 = epsilon_pq(v, ("p1", "p2", "p3"), "p1", "p3")
    assert eps2 > 0.0


def test_all_pair_interactions_keys():
    v = table(lambda c: len(c))
    out = all_pair_interactions(v, PLAYERS)
    assert set(out.keys()) == {("crop", "mask"), ("crop", "reorder"), ("mask", "reorder")}
    assert all(abs(x) < 1e-12 for x in out.values())
