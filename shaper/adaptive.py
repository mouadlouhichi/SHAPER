"""SHAPER-Weight and SHAPER-Select (spec A.10, paper 3.7/3.8).

Weight target: ONE dataset-level vector formed by averaging the canonical
fixed-nominal-budget Game-A seed-specific Shapley values on V_game.
NLL/cosine severity-derived weights are exploratory and can never replace
this target after outcomes are observed.

    positive = max(phi, 0);  q = positive/sum(positive)   (uniform if sum<=eps)
    w(alpha) = (1-alpha)/K + alpha*q
    L = L_rec + lambda_cl * sum_p w_p L_cl^p          (no extra denominator)

Select: ranked by mean raw Game-A V_game Shapley; fixed display tie order
crop < mask < reorder < dropout; if the two lowest differ by less than
delta_phi -> no action; else remove the lowest-Shapley view (reusing the
already trained coalition model). Test metrics are NEVER used to select.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import MAIN_PLAYERS, PLAYER_TIE_ORDER


def positive_transform(
    phi: Mapping[str, float], eps: float = 1e-12, K: Optional[int] = None
) -> Dict[str, float]:
    """Nonnegative credit distribution used ONLY for training weights.

    Raw signed Shapley values are reported unchanged; clipping breaks the
    efficiency interpretation, so these are Shapley-derived training weights,
    not an exact utility decomposition.
    """
    players = list(phi.keys())
    n = K or len(players)
    positive = np.maximum(np.asarray([phi[p] for p in players], dtype=np.float64), 0.0)
    if positive.sum() <= eps:
        q = np.full(len(players), 1.0 / len(players))
    else:
        q = positive / positive.sum()
    return {p: float(qi) for p, qi in zip(players, q)}


def shaper_weights(phi: Mapping[str, float], alpha: float, eps: float = 1e-12) -> Dict[str, float]:
    """w = (1-alpha)/K + alpha*q."""
    players = list(phi.keys())
    K = len(players)
    q = positive_transform(phi, eps=eps)
    return {p: (1.0 - alpha) / K + alpha * q[p] for p in players}


def mean_game_a_shapley(per_seed_phi: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    """Dataset-level attribution: mean of exact primary Game-A seed vectors."""
    players = list(per_seed_phi[0].keys())
    return {p: float(np.mean([s[p] for s in per_seed_phi])) for p in players}


def weight_activation_rule(
    grand_uplift_ci95: Tuple[float, float],
    per_seed_phi: Sequence[Mapping[str, float]],
    dataset_level_phi: Mapping[str, float],
) -> Dict[str, Any]:
    """Locked activation rule (spec A.12 / prompt section 38):

    1. Game-A grand-uplift 95% CI is entirely above zero;
    2. highest-weight view is positive in >= 4/5 seeds (8/10 after extension);
    3. every positively weighted view is positive in >= 80% of seeds.
    Otherwise NOT_ACTIVATED with rec-only deployment fallback.
    """
    q = positive_transform(dataset_level_phi)
    highest = max(q, key=q.get)
    n_seeds = len(per_seed_phi)
    majority_threshold = 8 if n_seeds >= 10 else 4
    checks: Dict[str, Dict[str, Any]] = {}

    ci_ok = grand_uplift_ci95[0] > 0
    checks["grand_uplift_ci_above_zero"] = {
        "pass": ci_ok,
        "detail": f"CI95 = ({grand_uplift_ci95[0]:.6f}, {grand_uplift_ci95[1]:.6f})",
    }

    highest_positive = sum(1 for s in per_seed_phi if s[highest] > 0)
    checks["highest_weight_sign_stability"] = {
        "pass": highest_positive >= majority_threshold,
        "detail": f"{highest} positive in {highest_positive}/{n_seeds} seeds (need >= {majority_threshold})",
    }

    positive_views = [p for p, qv in q.items() if qv > 0]
    unstable = []
    for p in positive_views:
        frac = sum(1 for s in per_seed_phi if s[p] > 0) / n_seeds
        if frac < 0.80:
            unstable.append((p, round(frac, 3)))
    checks["all_positive_views_80pct_stable"] = {
        "pass": not unstable,
        "detail": f"unstable views: {unstable}",
    }

    activated = ci_ok and checks["highest_weight_sign_stability"]["pass"] and checks["all_positive_views_80pct_stable"]["pass"]
    return {
        "activated": activated,
        "status": "ACTIVATED" if activated else "NOT_ACTIVATED",
        "deployment_fallback": "weighted" if activated else "rec-only",
        "checks": checks,
        "note": "activation never happens because results look promising",
    }


def select_decision(
    phi_mean: Mapping[str, float],
    delta_phi: float,
    players: Sequence[str] = MAIN_PLAYERS,
    activation_active: bool = True,
) -> Dict[str, Any]:
    """SHAPER-Select locked decision rule.

    Activation failure -> not activated, rec-only fallback. Otherwise:
    rank by mean raw Game-A Shapley (display tie order fixed); if the two
    lowest differ by < delta_phi -> no action (keep grand); else remove the
    lowest-Shapley view, REUSING the already trained coalition model.
    """
    if not activation_active:
        return {
            "status": "not activated",
            "action": "none",
            "deployment": "rec-only",
            "selected_coalition": [],
            "removed_view": None,
            "note": "activation rule failed; rec-only deployment fallback",
        }
    tie_order = {p: i for i, p in enumerate(PLAYER_TIE_ORDER)}
    ordered = sorted(players, key=lambda p: (phi_mean.get(p, 0.0), -tie_order[p]))
    # careful: display tie order means smaller index first on EQUAL values
    ordered = sorted(players, key=lambda p: (phi_mean.get(p, 0.0), tie_order[p]))
    lowest, second = ordered[0], ordered[1]
    gap = phi_mean[second] - phi_mean[lowest]
    if gap < delta_phi:
        return {
            "status": "activated",
            "action": "no_action",
            "deployment": "grand coalition (all views retained)",
            "selected_coalition": sorted(players),
            "removed_view": None,
            "gap_two_lowest": float(gap),
            "note": f"phi({second}) - phi({lowest}) = {gap:.6f} < delta_phi = {delta_phi}",
        }
    kept = sorted([p for p in players if p != lowest])
    return {
        "status": "activated",
        "action": "remove_lowest",
        "deployment": f"coalition {kept} (reused from the Game-A sweep)",
        "selected_coalition": kept,
        "removed_view": lowest,
        "gap_two_lowest": float(gap),
    }


def calibrate_alpha(
    alpha_grid: Sequence[float],
    alpha_metrics: Mapping[float, float],
    higher_is_better: bool = True,
) -> Dict[str, Any]:
    """Lock one alpha per dataset on V_select.

    Tie-break (documented engineering choice, absent from the spec): smaller
    alpha — the more conservative (closer-to-uniform) candidate.
    """
    best_value = max(alpha_metrics.values()) if higher_is_better else min(alpha_metrics.values())
    best = [a for a in alpha_grid if alpha_metrics[a] == best_value]
    alpha = min(best) if higher_is_better else max(best)
    return {
        "alpha": alpha,
        "best_metric": best_value,
        "tie_break": "smaller alpha (closer to uniform) on ties",
        "scores": dict(alpha_metrics),
    }


__all__ = [
    "positive_transform",
    "shaper_weights",
    "mean_game_a_shapley",
    "weight_activation_rule",
    "select_decision",
    "calibrate_alpha",
]
