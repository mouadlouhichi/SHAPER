"""Exact Grabisch-Roubens Shapley interaction indices and redundancy
diagnostics (spec A.8, paper 3.5).

    I_pq = sum_{S subset P\\{p,q}}  |S|! (K-|S|-2)! / (K-1)!  Delta_pq v(S)

    Delta_pq v(S) = v(S u {p,q}) - v(S u {p}) - v(S u {q}) + v(S).

Exact for complete K=3 tables ONLY. The K=4 value table is incomplete by
design (antithetic permutation MC): exact K=4 interactions are prohibited
and raise ValueError. A practically directional interaction requires a 95%
interval excluding zero AND |mean I_pq| >= delta_interaction; otherwise the
result is INCONCLUSIVE.

The locked three-player perfect-substitutability counterexample (paper,
Appendix A) is included verbatim: grand-LOO of both substitutes is zero,
Shapley gives equal nonzero credit, and the rejected half-pair-removal
formula (v(p1p2)-v(p3))/2 would give the wrong answer.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


def interaction_weight(size_s: int, K: int) -> float:
    """|S|! (K-|S|-2)! / (K-1)!"""
    return math.factorial(size_s) * math.factorial(K - size_s - 2) / math.factorial(K - 1)


def grabisch_roubens_interaction(
    values: Mapping[Tuple[str, ...], float], players: Sequence[str], p: str, q: str
) -> float:
    """Context-averaged Grabisch-Roubens interaction index for a COMPLETE
    table. Raises if the table is incomplete (never approximate)."""
    players = tuple(players)
    K = len(players)
    required = [tuple(c) for c in _all_subsets(players)]
    keys = {tuple(sorted(k)) for k in values}
    missing = [c for c in required if tuple(sorted(c)) not in keys]
    if missing:
        raise ValueError(
            f"incomplete table; exact interactions require all {2 ** K} coalitions "
            f"(missing: {missing}). K=4 MC tables are incomplete by design."
        )
    lookup = {tuple(sorted(k)): float(v) for k, v in values.items()}
    rest = [r for r in players if r not in (p, q)]
    total = 0.0
    for size_s in range(len(rest) + 1):
        for S in itertools.combinations(rest, size_s):
            S = tuple(sorted(S))
            Sp = tuple(sorted(S + (p,)))
            Sq = tuple(sorted(S + (q,)))
            Spq = tuple(sorted(S + (p, q)))
            delta = lookup[Spq] - lookup[Sp] - lookup[Sq] + lookup[S]
            total += interaction_weight(size_s, K) * delta
    return float(total)


def all_pair_interactions(
    values: Mapping[Tuple[str, ...], float], players: Sequence[str]
) -> Dict[Tuple[str, str], float]:
    return {
        (p, q): grabisch_roubens_interaction(values, players, p, q)
        for p, q in itertools.combinations(players, 2)
    }


def declare_interaction(
    interactions: Sequence[float],
    delta_interaction: float,
    ci95: Optional[Tuple[float, float]] = None,
) -> Dict[str, Any]:
    """Practical-direction rule: 95% CI excludes zero AND |mean| >= delta;
    otherwise INCONCLUSIVE."""
    mean = float(np.mean(interactions))
    if ci95 is None:
        se = float(np.std(interactions, ddof=1)) / math.sqrt(len(interactions)) if len(interactions) > 1 else math.inf
        from scipy.stats import t as _t

        half = _t.ppf(0.975, len(interactions) - 1) * se if len(interactions) > 1 else math.inf
        ci95 = (mean - half, mean + half)
    excludes_zero = ci95[0] > 0 or ci95[1] < 0
    if excludes_zero and abs(mean) >= delta_interaction:
        direction = "negative (consistent with substitutability)" if mean < 0 else "positive (consistent with complementarity)"
        return {"declaration": "directional", "direction": direction, "mean": mean, "ci95": ci95}
    return {"declaration": "INCONCLUSIVE", "direction": None, "mean": mean, "ci95": ci95}


def perfect_substitutability_game() -> Dict[Tuple[str, ...], float]:
    """The locked three-player counterexample (paper Appendix A).

    p1, p2 satisfy the exact substitutability property for every context:
        v(C u p1) = v(C u p2) = v(C u p1p2).
    Grand-LOO of both is ZERO; Shapley gives phi1 = phi2 = .0417,
    phi3 = .0167; the rejected half-pair-removal expression gives
    (v(p1p2) - v(p3)) / 2 = (.10 - .05)/2 = .025 != .0417.
    """
    return {
        (): 0.0,
        ("p1",): 0.10,
        ("p2",): 0.10,
        ("p1", "p2"): 0.10,
        ("p3",): 0.05,
        ("p1", "p3"): 0.10,
        ("p2", "p3"): 0.10,
        ("p1", "p2", "p3"): 0.10,
    }


def rejected_half_pair_removal(v: Mapping[Tuple[str, ...], float], p1: str, p2: str, p3: str) -> float:
    """The INVALID formula this project explicitly rejects without the extra
    context-independence assumption: (v(p1p2) - v(p3)) / 2."""
    return (v[tuple(sorted((p1, p2)))] - v[(p3,)]) / 2.0


def _all_subsets(players: Sequence[str]) -> List[Tuple[str, ...]]:
    out = []
    for size in range(len(players) + 1):
        for combo in itertools.combinations(players, size):
            out.append(combo)
    return out


__all__ = [
    "interaction_weight",
    "grabisch_roubens_interaction",
    "all_pair_interactions",
    "declare_interaction",
    "perfect_substitutability_game",
    "rejected_half_pair_removal",
]
