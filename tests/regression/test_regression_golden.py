"""Regression tests: golden reference values for the locked mathematical
objects and for protocol configuration.

These pin exact numbers so any change to allocation arithmetic, the locked
counterexample, or the frozen protocol configs fails loudly.
"""

from __future__ import annotations

import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from shaper.game import grand_loo
from shaper.interactions import (
    grabisch_roubens_interaction,
    perfect_substitutability_game,
)
from shaper.shapley import exact_shapley


def test_golden_counterexample_shapley_values():
    v = perfect_substitutability_game()
    phi = exact_shapley(v, ("p1", "p2", "p3"))
    assert phi == pytest.approx({"p1": 1 / 24, "p2": 1 / 24, "p3": 1 / 60})
    # in the locked counterexample ALL grand-LOO values are zero
    assert grand_loo(v, ("p1", "p2", "p3")) == pytest.approx({"p1": 0.0, "p2": 0.0, "p3": 0.0})


def test_golden_random_game_hash():
    """Pin the Shapley vector of a fixed pseudo-random game."""
    rng = np.random.default_rng(20260813)
    players = ("crop", "mask", "reorder")
    import itertools

    values = {tuple(sorted(c)): float(rng.uniform(-0.1, 0.1)) for r in range(4) for c in itertools.combinations(players, r)}
    phi = exact_shapley(values, players)
    digest = hashlib.blake2b(
        repr(sorted((p, round(v, 12)) for p, v in phi.items())).encode()
    ).hexdigest()
    assert digest == "909e4e3eb27719790b0b45542f5a29cda30aff3247d7ba649675ace53749fbbd4290a8f513ac4606aa756ff5e85e946c413e7acb7ca2c789d619a5430ee4f3d4"


def test_golden_interaction_values():
    v = perfect_substitutability_game()
    # p1, p2 are perfect substitutes: Delta(empty) = -.10, Delta({p3}) = -.05,
    # weight 1/2 -> I = 0.5 * (-.15) = -.075
    assert grabisch_roubens_interaction(v, ("p1", "p2", "p3"), "p1", "p2") == pytest.approx(-0.075)
    # p1, p3: Delta(empty) = .1-.1-.05+0 = -.05 -> -0.025
    assert grabisch_roubens_interaction(v, ("p1", "p2", "p3"), "p1", "p3") == pytest.approx(-0.025)


def test_golden_config_hashes_stable():
    """Config hashes are part of frozen artifacts/caches; they must not
    drift without a deliberate, recorded amendment."""
    from shaper.config import load_run_config

    ml1m = load_run_config("ml1m")
    beauty = load_run_config("beauty")
    assert ml1m.config_hash() != beauty.config_hash()
    assert len(ml1m.config_hash()) == 32
    # lock the current canonical hashes
    assert ml1m.config_hash() == "fac5f791fddd25dddbc0a1efdb0c5460"
    assert beauty.config_hash() == "3eb6a105ba0770893b94a8cd2c0ee2a9"


def test_golden_seed_registry_hash():
    from shaper.config import load_yaml
    from shaper.provenance import stable_hash

    seeds = load_yaml(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "configs", "seeds.yaml"))
    assert stable_hash(seeds["registry"]) == "4ce219f72b89fcf77d5070e5df5588d4"
