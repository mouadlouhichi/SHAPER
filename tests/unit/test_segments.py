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


REPO_ROOT_FIXTURE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeRun:
    def __init__(self, root):
        self.root = str(root)


def test_per_user_table_storage_contract(tmp_path):
    """Regression: per-user utilities live in the coalition-table companion
    .npz (written by save_coalition_table, read by load_per_user_table AND by
    scripts/run_segments.py). The old raw/ location must not be required."""
    import numpy as np

    from shaper.game import CoalitionValueRecord, load_per_user_table, save_coalition_table

    records = [
        CoalitionValueRecord(
            dataset="synthetic", seed=2001, policy="game_a", coalition=(),
            metrics_by_role={"game": {"ndcg": 0.1, "nll": 1.0}},
            per_user_ndcg_game=[0.0, 0.0, 0.0],
        ),
        CoalitionValueRecord(
            dataset="synthetic", seed=2001, policy="game_a", coalition=("crop",),
            metrics_by_role={"game": {"ndcg": 0.12, "nll": 0.9}},
            per_user_ndcg_game=[0.01, 0.02, 0.03],
        ),
    ]
    table_dir = tmp_path / "coalition_tables"
    table_dir.mkdir()
    path = str(table_dir / "game_a_seed2001.json")
    save_coalition_table(records, path)
    assert os.path.exists(str(table_dir / "game_a_seed2001.npz"))
    table = load_per_user_table(path)
    assert table is not None
    assert () in table and ("crop",) in table
    assert list(table[("crop",)]) == [0.01, 0.02, 0.03]

    # scripts/run_segments.py discovers tables from the coalition_tables dir
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_segments", os.path.join(REPO_ROOT_FIXTURE, "scripts", "run_segments.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tables, found = mod.load_per_user_tables(_FakeRun(tmp_path), None, "game_a", [2001])
    assert found == [2001]
    assert len(tables) == 1
