"""Protocol tests: fixed seed registry, locked thresholds, Holm families,
Game-B scope, severity-diagnostic scope, and paper/code consistency."""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

from shaper.config import load_run_config, load_yaml  # noqa: E402


def _seeds():
    cfg = load_run_config("ml1m")
    return cfg.seeds


def test_seed_registry_locked():
    s = _seeds()
    assert s["recipe_alpha"] == [901, 902, 903]
    assert s["pilots"] == [1001, 1002]
    assert s["engineering"] == 1001
    assert s["confirmatory_game_a"] == [2001, 2002, 2003, 2004, 2005]
    assert s["extension_game_a"] == [2006, 2007, 2008, 2009, 2010]
    assert s["game_b"] == [2001, 2002, 2003]
    assert s["cosine_diagnostic"] == [2001, 2002, 2003]
    assert s["k4_beauty_mc"] == [3001, 3002, 3003, 3004, 3005]
    assert s["cached_adapter"] == 4001


def test_thresholds_locked_at_0003():
    stats = load_yaml(os.path.join(REPO_ROOT, "configs", "statistics.yaml"))
    assert stats["thresholds"]["delta_phi"] == 0.003
    assert stats["thresholds"]["delta_interaction"] == 0.003
    assert stats["thresholds"]["delta_action"] == 0.003


def test_holm_families_locked():
    stats = load_yaml(os.path.join(REPO_ROOT, "configs", "statistics.yaml"))
    inter = stats["holm"]["intervention_family"]
    assert inter == [
        "weight_vs_uniform", "select_vs_uniform", "weight_vs_loo",
        "select_vs_drop_lowest_loo", "weight_vs_gates", "weight_vs_direct_search",
    ]
    mask = stats["holm"]["mask_family"]
    assert len(mask) == 6
    assert sum(1 for x in mask if x.endswith("_ndcg")) == 3
    assert sum(1 for x in mask if x.endswith("_nll")) == 3
    assert stats["holm"]["interaction_family_per_dataset"] == [
        ["crop", "mask"], ["crop", "reorder"], ["mask", "reorder"]
    ]


def test_roles_locked_20_60_20():
    for dataset in ("ml1m", "beauty"):
        cfg = load_run_config(dataset)
        assert cfg.roles == {"tune": 0.20, "game": 0.60, "select": 0.20}


def test_game_b_scope_six_intermediate_only():
    from shaper.game import all_coalitions, intermediate_coalitions
    from shaper import MAIN_PLAYERS

    assert len(all_coalitions(MAIN_PLAYERS)) == 8
    intermediates = intermediate_coalitions(MAIN_PLAYERS)
    assert len(intermediates) == 6
    assert () not in intermediates
    assert tuple(MAIN_PLAYERS) not in intermediates


def test_game_b_never_expands_or_drives_rq4():
    """Structural: Game-B seed list equals the locked 2001-2003 and the
    Weight/Select entry points consume game_a_shapley.json only."""
    import inspect

    from scripts import run_interventions

    src = inspect.getsource(run_interventions.load_game_a)
    assert "game_a_shapley.json" in src
    assert "game_b" not in src


def test_severity_diagnostics_frozen_scope():
    """Severity calibration uses frozen checkpoints only and never trains a
    coalition Shapley sweep."""
    import inspect

    from scripts import run_game

    src = inspect.getsource(run_game.severity_calibration)
    assert "load_coalition_checkpoint" in src
    assert "no additional coalition Shapley sweep" in src
    # severity grids locked
    stats = load_yaml(os.path.join(REPO_ROOT, "configs", "statistics.yaml"))
    assert stats["severity"]["crop_eta_lows"] == [0.5, 0.6, 0.7, 0.8]
    assert stats["severity"]["mask_gammas"] == [0.1, 0.2, 0.3, 0.4]
    assert stats["severity"]["reorder_betas"] == [0.1, 0.2, 0.3, 0.4]
    assert stats["severity"]["match_tolerance"] == 0.10


def test_k4_scope_beauty_game_a_only():
    scope = load_yaml(os.path.join(REPO_ROOT, "configs", "scope.yaml"))
    k4 = next(i for i in scope["scope"] if i["item"] == "beauty_k4_game_a_mc")
    assert k4["datasets"] == ["beauty"]
    assert k4["max_unique_models_per_seed"] == 10
    assert k4["exact_interactions"] is False


def test_augmentation_parameters_locked():
    for dataset in ("ml1m", "beauty"):
        cfg = load_run_config(dataset)
        assert cfg.augmentation["min_contrastive_pairs"] == 8
        assert cfg.augmentation["crop"]["eta_low"] == 0.5
        assert cfg.augmentation["crop"]["eta_high"] == 1.0
        assert cfg.augmentation["mask"]["gamma"] == 0.2
        assert cfg.augmentation["mask"]["protect_last"] is False
        assert cfg.augmentation["reorder"]["beta"] == 0.2
        assert cfg.augmentation["reorder"]["min_span"] == 2
        assert cfg.augmentation["dropout"]["extra_forwards"] == 2


def test_training_protocol_locked():
    for dataset in ("ml1m", "beauty"):
        cfg = load_run_config(dataset)
        assert cfg.training["optimizer"] == "adamw"
        assert cfg.training["betas"] == [0.9, 0.98]
        assert cfg.training["eps"] == 1.0e-8
        assert cfg.training["weight_decay"] == 1.0e-4
        assert cfg.training["grad_clip"] == 1.0
        assert cfg.training["warmup_frac"] == 0.10
        assert cfg.training["cosine_final_frac"] == 0.1
    assert load_run_config("ml1m").batch_size == 128
    assert load_run_config("beauty").batch_size == 256
    assert load_run_config("ml1m").max_len == 200
    assert load_run_config("beauty").max_len == 50


def test_paper_consistency_document_exists():
    path = os.path.join(REPO_ROOT, "docs", "paper_implementation_consistency.md")
    assert os.path.isfile(path), "docs/paper_implementation_consistency.md missing"
    text = open(path, encoding="utf-8").read().lower()
    for needle in (
        "game a", "game b", "2001", "2002", "2003", "3001", "k=4", "antithetic",
        "0.003", "20%/60%/20%", "adamw", "128", "256", "10 unique",
    ):
        assert needle in text, f"consistency doc missing: {needle}"


def test_no_go_registry_present():
    scope = load_yaml(os.path.join(REPO_ROOT, "configs", "scope.yaml"))
    no_go = scope["no_go"]
    for item in ("identical_coalition_rankings", "efficiency_failure",
                 "per_user_consistency_failure", "split_leakage",
                 "excluded_view_leakage", "projection_only_surrogate_as_primary",
                 "warm_start_from_grand_checkpoint", "unequal_optimizer_budgets",
                 "test_affects_selection", "invalid_half_pair_removal_formula"):
        assert item in no_go
