"""Unit tests: the registered antithetic permutation-MC K=4 estimator."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest

from shaper import monte_carlo as mc
from shaper.monte_carlo import (
    MAX_UNIQUE_MODELS_PER_SEED,
    AntitheticPermutationMCShapley,
    K4SeedResult,
    path_marginals,
    prefix_path,
    required_coalitions,
    reverse_permutation,
    sample_permutation,
    telescoping_check,
)

PLAYERS = ("crop", "mask", "reorder", "dropout")


def complete_random_table(seed=0):
    import itertools

    rng = np.random.default_rng(seed)
    return {tuple(sorted(c)): float(rng.uniform(-0.1, 0.15)) for r in range(5) for c in itertools.combinations(PLAYERS, r)}


def test_permutation_sampling_deterministic():
    pi1 = sample_permutation(3001, PLAYERS)
    pi2 = sample_permutation(3001, PLAYERS)
    assert pi1 == pi2
    assert sorted(pi1) == sorted(PLAYERS)
    assert reverse_permutation(pi1) == tuple(reversed(pi1))


def test_required_coalitions_within_cap_and_complete_for_loo():
    for seed in (3001, 3002, 3003, 3004, 3005, 3101, 3102, 3103, 3104, 3105):
        pi = sample_permutation(seed, PLAYERS)
        registry = required_coalitions(pi, PLAYERS)
        assert len(registry) <= MAX_UNIQUE_MODELS_PER_SEED
        # empty + grand present
        assert () in registry.values()
        assert tuple(sorted(PLAYERS)) in registry.values()
        # every prefix of both paths present
        for path in (pi, reverse_permutation(pi)):
            for pref in prefix_path(path):
                assert tuple(sorted(pref)) in registry.values()
        # all four grand-LOO coalitions present
        for p in PLAYERS:
            assert tuple(sorted(q for q in PLAYERS if q != p)) in registry.values()


def test_fail_loudly_at_model_eleven(monkeypatch):
    monkeypatch.setattr(mc, "MAX_UNIQUE_MODELS_PER_SEED", 9)
    raised = False
    for seed in range(3001, 3100):
        pi = sample_permutation(seed, PLAYERS)
        try:
            required_coalitions(pi, PLAYERS)
        except RuntimeError as exc:
            assert "budget exceeded" in str(exc)
            raised = True
            break
    assert raised, "no seed exercised the loud model-budget failure"


def test_path_marginals_and_telescoping():
    table = complete_random_table(0)
    pi = sample_permutation(3001, PLAYERS)
    for path in (pi, reverse_permutation(pi)):
        marginals = path_marginals(path, table, PLAYERS)
        assert set(marginals.keys()) == set(PLAYERS)
        check = telescoping_check(path, table, PLAYERS)
        assert check["within_precision"]
        assert abs(check["residual"]) < 1e-9


def test_antithetic_estimate_is_mean_of_marginals():
    table = complete_random_table(1)
    pi = sample_permutation(3002, PLAYERS)
    est = AntitheticPermutationMCShapley(PLAYERS)
    result = est.new_seed_result(3002)
    assert result.permutation == pi
    est.set_values(result, table)
    m_pi = path_marginals(pi, table, PLAYERS)
    m_rev = path_marginals(reverse_permutation(pi), table, PLAYERS)
    for p in PLAYERS:
        assert abs(result.estimates[p] - 0.5 * (m_pi[p] + m_rev[p])) < 1e-12


def test_antithetic_preserves_telescoping_in_aggregate():
    est = AntitheticPermutationMCShapley(PLAYERS)
    for i, seed in enumerate((3001, 3002, 3003, 3004, 3005)):
        r = est.new_seed_result(seed)
        est.set_values(r, complete_random_table(i))
    agg = est.aggregate(delta_phi=0.003)
    # antithetic means telescope: sum of antithetic-averaged marginals equals
    # the mean of the path sums
    ests = {p: agg["per_player"][p]["estimate"] for p in PLAYERS}
    table_sums = [
        est.seed_results[i].values[tuple(sorted(PLAYERS))] - est.seed_results[i].values[()]
        for i in range(5)
    ]
    assert abs(sum(ests.values()) - float(np.mean(table_sums))) < 1e-9
    # every sampled path telescopes exactly
    assert all(t["within_precision"] for t in agg["telescoping"])


def test_aggregate_labels_and_inconclusive():
    est = AntitheticPermutationMCShapley(PLAYERS)
    for i, seed in enumerate((3001, 3002, 3003, 3004, 3005)):
        r = est.new_seed_result(seed)
        est.set_values(r, complete_random_table(i))
    agg = est.aggregate(delta_phi=0.003)
    assert agg["label"] == "Monte-Carlo approximate"
    assert agg["interactions"]["status"] == "NOT_COMPUTED"
    if agg["ordering"]["status"] == "interpretable":
        assert set(agg["ordering"]["order"]) == set(PLAYERS)
    else:
        assert agg["ordering"]["status"] == "INCONCLUSIVE"
    # forcing an unattainably tight threshold yields INCONCLUSIVE
    agg_tight = est.aggregate(delta_phi=0.0)
    assert agg_tight["ordering"]["status"] == "INCONCLUSIVE"


def test_k4_seed_result_persistence_preserves_permutation(tmp_path):
    est = AntitheticPermutationMCShapley(PLAYERS)
    r = est.new_seed_result(3003)
    path = str(tmp_path / "k4.json")
    r.save(path)
    loaded = K4SeedResult.load(path)
    assert loaded.permutation == r.permutation
    assert loaded.reverse_permutation == r.reverse_permutation
    assert loaded.coalition_registry == r.coalition_registry


def test_estimator_registered_for_k4_only():
    with pytest.raises(ValueError, match="K=4"):
        AntitheticPermutationMCShapley(("a", "b", "c"))
