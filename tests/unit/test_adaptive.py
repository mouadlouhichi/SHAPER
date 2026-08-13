"""Unit tests: SHAPER-Weight / SHAPER-Select and the locked activation rule."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.adaptive import (
    calibrate_alpha,
    mean_game_a_shapley,
    positive_transform,
    select_decision,
    shaper_weights,
    weight_activation_rule,
)

PHI_POS = {"crop": 0.012, "mask": 0.030, "reorder": 0.006}
PHI_NEG = {"crop": -0.01, "mask": -0.02, "reorder": -0.03}


def test_positive_transform_clips_negatives():
    q = positive_transform(PHI_POS)
    assert all(q[p] >= 0 for p in q)
    assert abs(sum(q.values()) - 1.0) < 1e-12
    assert q["mask"] == pytest.approx(0.030 / 0.048)


def test_positive_transform_all_nonpositive_gives_uniform():
    q = positive_transform(PHI_NEG)
    assert all(q[p] == pytest.approx(1 / 3) for p in q)


def test_positive_transform_eps_fallback():
    q = positive_transform({"crop": 0.0, "mask": 0.0, "reorder": 0.0}, eps=1e-12)
    assert all(v == pytest.approx(1 / 3) for v in q.values())


def test_shaper_weights_grid():
    w0 = shaper_weights(PHI_POS, alpha=0.0)
    assert all(v == pytest.approx(1 / 3) for v in w0.values())
    w1 = shaper_weights(PHI_POS, alpha=1.0)
    q = positive_transform(PHI_POS)
    assert w1 == pytest.approx(q)
    w = shaper_weights(PHI_POS, alpha=0.5)
    for p in PHI_POS:
        assert w[p] == pytest.approx(0.5 / 3 + 0.5 * q[p])
    assert abs(sum(w.values()) - 1.0) < 1e-12


def test_mean_game_a_shapley():
    per_seed = [PHI_POS, {k: v * 1.1 for k, v in PHI_POS.items()}]
    m = mean_game_a_shapley(per_seed)
    assert m["mask"] == pytest.approx(0.0315)


def test_activation_rule_pass_case():
    per_seed = [PHI_POS] * 5
    out = weight_activation_rule((0.001, 0.01), per_seed, PHI_POS)
    assert out["activated"] and out["status"] == "ACTIVATED"
    assert out["deployment_fallback"] == "weighted"


def test_activation_rule_fails_on_grand_uplift_ci():
    per_seed = [PHI_POS] * 5
    out = weight_activation_rule((-0.01, 0.01), per_seed, PHI_POS)
    assert not out["activated"]
    assert out["deployment_fallback"] == "rec-only"
    assert out["status"] == "NOT_ACTIVATED"


def test_activation_rule_fails_on_highest_weight_instability():
    per_seed = []
    for i in range(5):
        phi = dict(PHI_POS)
        if i >= 2:  # mask (highest weight) negative in 3/5 seeds
            phi["mask"] = -0.01
        per_seed.append(phi)
    out = weight_activation_rule((0.001, 0.01), per_seed, PHI_POS)
    assert not out["activated"]


def test_activation_rule_80pct_rule():
    per_seed = []
    for i in range(5):
        phi = dict(PHI_POS)
        if i >= 2:  # crop positive in only 2/5 = 40% < 80%
            phi["crop"] = -0.001
        per_seed.append(phi)
    out = weight_activation_rule((0.001, 0.01), per_seed, PHI_POS)
    assert not out["activated"]
    assert not out["checks"]["all_positive_views_80pct_stable"]["pass"]


def test_activation_rule_8_of_10_after_extension():
    per_seed = []
    for i in range(10):
        phi = dict(PHI_POS)
        if i >= 4:  # mask positive in 4/10 < 8/10 threshold
            phi["mask"] = -0.01
        per_seed.append(phi)
    out = weight_activation_rule((0.001, 0.01), per_seed, PHI_POS)
    assert not out["activated"]


def test_select_removes_lowest_when_gap_exceeds_delta():
    decision = select_decision(PHI_POS, delta_phi=0.003, activation_active=True)
    assert decision["status"] == "activated"
    assert decision["action"] == "remove_lowest"
    assert decision["removed_view"] == "reorder"
    assert decision["selected_coalition"] == ["crop", "mask"]


def test_select_no_action_within_delta():
    phi = {"crop": 0.010, "mask": 0.011, "reorder": 0.012}
    decision = select_decision(phi, delta_phi=0.003, activation_active=True)
    assert decision["action"] == "no_action"
    assert decision["selected_coalition"] == ["crop", "mask", "reorder"]


def test_select_tie_order():
    # fixed display tie order crop < mask < reorder < dropout: on equal values
    # crop is the lowest, so with gap == delta_phi (not <) the rule removes crop
    phi = {"crop": 0.01, "mask": 0.01, "reorder": 0.02}
    decision = select_decision(phi, delta_phi=0.0, activation_active=True)
    assert decision["action"] == "remove_lowest"
    assert decision["removed_view"] == "crop"
    # any positive delta_phi strictly above the gap -> no action
    phi2 = {"crop": 0.01, "mask": 0.01, "reorder": 0.01}
    d2 = select_decision(phi2, delta_phi=0.0001, activation_active=True)
    assert d2["action"] == "no_action"


def test_select_not_activated_falls_back_to_rec_only():
    decision = select_decision(PHI_POS, delta_phi=0.003, activation_active=False)
    assert decision["status"] == "not activated"
    assert decision["deployment"] == "rec-only"
    assert decision["selected_coalition"] == []


def test_alpha_calibration_tie_breaks_toward_smaller_alpha():
    out = calibrate_alpha([0.0, 0.25, 0.5, 0.75, 1.0], {0.0: 0.5, 0.25: 0.5, 0.5: 0.4, 0.75: 0.4, 1.0: 0.3})
    assert out["alpha"] == 0.0  # tie between 0.0 and 0.25 -> smaller alpha
    assert out["tie_break"].startswith("smaller alpha")
