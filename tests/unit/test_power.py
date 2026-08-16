"""Unit tests: Appendix-J precision tables and the locked direction-blind
seed-extension rules."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.power import (
    evaluate_game_a_extension_trigger,
    evaluate_intervention_extension_trigger,
    pilot_informed_table,
    variance_grid_table,
)
from shaper.stats import planning_half_width


def test_variance_grid_table_matches_formula():
    table = variance_grid_table(3600, [5, 10], [0.05, 0.10], [0.001, 0.003])
    assert len(table["rows"]) == 4
    for row in table["rows"]:
        for S in (5, 10):
            h = planning_half_width(S, row["sigma_seed"], row["sigma_user"], 3600)
            assert row[f"half_width_{S}_seeds"] == pytest.approx(h, abs=1e-6)


def test_pilot_informed_table_resolvability():
    table = pilot_informed_table(3600, [0.0008, 0.0012], 0.05, {"delta_phi": 0.003, "delta_action": 0.003})
    assert "pilot_note" in table
    assert set(table["resolvable"].keys()) == {"5seeds", "10seeds"}


def test_game_a_extension_trigger_direction_blind():
    wide_phi = {"crop": {"ci": (0.0, 0.010)}}  # half-width 0.005 > 0.003
    narrow_uplift = {"ci": (0.001, 0.005)}     # half-width 0.002 <= 0.003
    out = evaluate_game_a_extension_trigger(wide_phi, narrow_uplift, 0.003, 0.003)
    assert out["extend_game_a"] is True
    assert out["phi_half_width_trigger"] is True

    narrow_phi = {"crop": {"ci": (0.0, 0.004)}}  # half-width 0.002
    wide_uplift = {"ci": (-0.002, 0.006)}        # half-width 0.004 > 0.003
    out = evaluate_game_a_extension_trigger(narrow_phi, wide_uplift, 0.003, 0.003)
    assert out["extend_game_a"] is True
    assert out["grand_uplift_trigger"] is True

    out = evaluate_game_a_extension_trigger(narrow_phi, {"ci": (0.0, 0.004)}, 0.003, 0.003)
    assert out["extend_game_a"] is False


def test_intervention_extension_trigger():
    assert evaluate_intervention_extension_trigger({"ci": (0.0, 0.010)}, 0.003)["triggered"] is True
    assert evaluate_intervention_extension_trigger({"ci": (0.0, 0.004)}, 0.003)["triggered"] is False
