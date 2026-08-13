"""Pilot-informed precision tables (Appendix J) and the locked seed-extension
decision rules (spec A.12, "Precision and seed-count plan").

The conservative planning half-width is
    h = t_(.975, S-1) * sqrt(sigma_seed^2 / S + sigma_user^2 / N_game),
supplemented by a hierarchical-bootstrap estimate. Thresholds are fixed at
0.003 (NDCG@10) and are never scaled by observed grand uplift.

Seed extension rules (direction-blind, evaluated only after five primary
Game-A seeds):
  - Game A extends to seeds 2006-2010 iff ANY raw phi interval half-width
    exceeds delta_phi OR the grand-uplift CI half-width exceeds delta_action;
  - Weight-vs-uniform final intervention seeds extend iff that CI half-width
    exceeds delta_action;
  - Game B, NLL/cosine diagnostics, and K=4 MC NEVER expand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .stats import planning_half_width


def variance_grid_table(
    N_game: int, S_values: Sequence[int], sigma_user: Sequence[float], sigma_seed: Sequence[float],
    level: float = 0.95,
) -> Dict[str, Any]:
    """Appendix-J style half-width table over the prespecified variance grid."""
    rows = []
    for su in sigma_user:
        for ss in sigma_seed:
            row = {"sigma_user": su, "sigma_seed": ss}
            for S in S_values:
                row[f"half_width_{S}_seeds"] = round(planning_half_width(S, ss, su, N_game, level), 6)
            rows.append(row)
    return {
        "N_game": N_game,
        "S_values": list(S_values),
        "rows": rows,
        "formula": "h = t_(.975, S-1) * sqrt(sigma_seed^2/S + sigma_user^2/N_game)",
    }


def pilot_informed_table(
    N_game: int,
    pilot_seed_variances: Sequence[float],
    sigma_user_pilot: float,
    deltas: Dict[str, float],
) -> Dict[str, Any]:
    """Realized pilot-informed table: uses actual pilot variance estimates
    alongside the prespecified grid. Pilot seeds are excluded from
    confirmatory estimates; two seeds do NOT precisely identify seed variance.
    """
    ss_pilot = float(np.std(pilot_seed_variances, ddof=1)) if len(pilot_seed_variances) > 1 else float("nan")
    table = variance_grid_table(N_game, [5, 10], [sigma_user_pilot, 0.05, 0.10], [ss_pilot, 0.001, 0.003])
    table["pilot_seed_sd"] = ss_pilot
    table["pilot_note"] = (
        "two excluded pilot seeds provide variance estimates only; they do not "
        "precisely identify seed variance"
    )
    table["deltas"] = deltas
    table["resolvable"] = {
        f"{S}seeds": bool(
            all(r[f"half_width_{S}_seeds"] <= max(deltas.values()) for r in table["rows"])
        )
        for S in [5, 10]
    }
    return table


def evaluate_game_a_extension_trigger(
    phi_intervals: Dict[str, Dict[str, Any]],
    grand_uplift_interval: Dict[str, Any],
    delta_phi: float,
    delta_action: float,
) -> Dict[str, Any]:
    """Direction-blind rule evaluated AFTER the five primary Game-A seeds."""
    phi_triggered = any(
        (interval["ci"][1] - interval["ci"][0]) / 2 > delta_phi
        for interval in phi_intervals.values()
    )
    uplift_triggered = (grand_uplift_interval["ci"][1] - grand_uplift_interval["ci"][0]) / 2 > delta_action
    return {
        "phi_half_width_trigger": phi_triggered,
        "grand_uplift_trigger": uplift_triggered,
        "extend_game_a": phi_triggered or uplift_triggered,
        "rule": (
            "extend Game A to seeds 2006-2010 iff any raw phi interval half-width "
            "exceeds delta_phi OR the grand-uplift CI half-width exceeds delta_action"
        ),
        "note": "never extended because an effect appears promising",
    }


def evaluate_intervention_extension_trigger(
    weight_minus_uniform_ci: Dict[str, Any], delta_action: float
) -> Dict[str, Any]:
    triggered = (weight_minus_uniform_ci["ci"][1] - weight_minus_uniform_ci["ci"][0]) / 2 > delta_action
    return {
        "triggered": triggered,
        "rule": "extend final Weight/uniform seeds to 2006-2010 iff the Weight-vs-uniform CI half-width exceeds delta_action",
    }


__all__ = [
    "variance_grid_table",
    "pilot_informed_table",
    "evaluate_game_a_extension_trigger",
    "evaluate_intervention_extension_trigger",
]
