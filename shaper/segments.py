"""Behavioural segments and the RQ3 heterogeneity analysis (spec A.11).

Confirmatory segments are Q1-Q4 of pre-truncation training-history length,
frozen in the data artifact BEFORE any attribution is seen. Confirmatory
inference uses exact primary K=3 Game-A values on the fixed V_game users;
Game B and the K=4 MC extension are EXCLUDED from confirmatory segment
inference.

Tests (studentized, user-level label permutations; one permuted user map is
applied to ALL seed records):
    T_mask = sum_{m=1..4} (m - 2.5) * mu[m, mask]           (one-sided trend)
    T_all  = sum_{m,p} (mu[m,p] - mu[p])^2 / (SE[m,p]^2 + eps)   (omnibus)

Crop/reorder trends are exploratory; the omnibus test is confirmatory but
does not preregister rejection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import MAIN_PLAYERS
from .shapley import per_user_shapley


def segment_means(
    per_user_phi: Dict[str, np.ndarray], quartiles: np.ndarray, n_quartiles: int = 4
) -> Dict[str, Dict[str, float]]:
    """mu[m, p]: mean per-user Shapley credit per segment per player."""
    out: Dict[str, Dict[str, float]] = {}
    for m in range(1, n_quartiles + 1):
        mask = quartiles == m
        out[str(m)] = {
            p: float(phi[mask].mean()) if mask.any() else float("nan") for p, phi in per_user_phi.items()
        }
    return out


def segment_ses(
    per_user_phi: Dict[str, np.ndarray], quartiles: np.ndarray, n_quartiles: int = 4
) -> Dict[str, Dict[str, float]]:
    """SE[m, p] of the segment mean (user-level, within segment)."""
    out: Dict[str, Dict[str, float]] = {}
    for m in range(1, n_quartiles + 1):
        mask = quartiles == m
        out[str(m)] = {}
        for p, phi in per_user_phi.items():
            vals = phi[mask]
            if len(vals) > 1:
                out[str(m)][p] = float(vals.std(ddof=1) / np.sqrt(len(vals)))
            else:
                out[str(m)][p] = float("nan")
    return out


def studentized_mask_trend(mu: Dict[str, Dict[str, float]], player: str = "mask") -> float:
    """T_mask = sum_m (m - 2.5) * mu[m, mask]; larger = increasing credit."""
    total = 0.0
    for m in range(1, 5):
        total += (m - 2.5) * mu[str(m)][player]
    return float(total)


def omnibus_profile_statistic(
    mu: Dict[str, Dict[str, float]], se: Dict[str, Dict[str, float]], eps: float = 1e-8
) -> float:
    """T_all = sum_{m,p} (mu[m,p] - mu[p])^2 / (SE[m,p]^2 + eps)."""
    players = list(mu["1"].keys())
    grand = {p: float(np.nanmean([mu[str(m)][p] for m in range(1, 5)])) for p in players}
    total = 0.0
    for m in range(1, 5):
        for p in players:
            delta = mu[str(m)][p] - grand[p]
            total += delta**2 / (se[str(m)][p] ** 2 + eps)
    return float(total)


def label_permutation_test(
    per_user_phi_by_seed: Sequence[Dict[str, np.ndarray]],
    quartiles: np.ndarray,
    n_permutations: int = 10_000,
    seed: int = 0,
    player: str = "mask",
) -> Dict[str, Any]:
    """Studentized label-permutation test for the mask trend and the omnibus
    profile statistic. ONE permuted user-to-segment map is applied across ALL
    seed records (quartile labels are never permuted independently by seed).

    per_user_phi_by_seed[s][p] is the per-user Shapley array of seed s.
    """
    rng = np.random.default_rng(seed)
    n_users = len(quartiles)
    n_seeds = len(per_user_phi_by_seed)

    # Observed statistics from per-seed aggregation (linearity over seeds:
    # mean over seeds of per-user vectors, then segment means).
    phi_mean = {
        p: np.mean([s[p] for s in per_user_phi_by_seed], axis=0)
        for p in per_user_phi_by_seed[0]
    }
    mu_obs = segment_means(phi_mean, quartiles)
    se_obs = segment_ses(phi_mean, quartiles)
    t_mask_obs = studentized_mask_trend(mu_obs, player=player)
    t_all_obs = omnibus_profile_statistic(mu_obs, se_obs)

    # Segment-count standardization: each permutation must preserve the
    # segment sizes — relabel user segment IDs (a permutation of labels),
    # which is the frozen protocol ("permutations of frozen user-to-segment
    # labels").
    t_mask_null = np.empty(n_permutations)
    t_all_null = np.empty(n_permutations)
    for it in range(n_permutations):
        perm_quartiles = rng.permutation(quartiles)
        mu_p = segment_means(phi_mean, perm_quartiles)
        se_p = segment_ses(phi_mean, perm_quartiles)
        t_mask_null[it] = studentized_mask_trend(mu_p, player=player)
        t_all_null[it] = omnibus_profile_statistic(mu_p, se_p)

    p_mask = float((t_mask_null >= t_mask_obs).mean())
    p_all = float((t_all_null >= t_all_obs).mean())
    return {
        "n_permutations": n_permutations,
        "n_users": n_users,
        "n_seeds": n_seeds,
        "T_mask_observed": t_mask_obs,
        "p_mask_one_sided": p_mask,
        "T_all_observed": t_all_obs,
        "p_all": p_all,
        "mu_observed": mu_obs,
        "se_observed": se_obs,
        "note": "same permuted user map applied across all seed records",
    }


def per_user_tables_by_seed(
    per_user_tables: Sequence[Dict], players: Sequence[str] = MAIN_PLAYERS
) -> Sequence[Dict[str, np.ndarray]]:
    """Convenience: per-user Shapley vectors per seed."""
    return [per_user_shapley([t], players) for t in per_user_tables]


__all__ = [
    "segment_means",
    "segment_ses",
    "studentized_mask_trend",
    "omnibus_profile_statistic",
    "label_permutation_test",
    "per_user_tables_by_seed",
]
