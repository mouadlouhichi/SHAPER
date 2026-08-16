"""Exact Shapley allocation over complete coalition tables (spec A.8).

    phi_p = sum_{C subset P\\{p}}  |C|! (K-|C|-1)! / K!  [v(C u {p}) - v(C)]

For K=3 this is exact over the complete 2^3 table; no approximation is used.
Properties verified in tests: efficiency (sum phi == v(P) within tolerance),
symmetry, dummy, additivity, per-user linear decomposition

    phi_p = (1/|U|) sum_u phi_{p,u}.

The mean seed-specific vector equals the Shapley vector of the mean observed
value table (linearity), without removing finite-seed uncertainty.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


def shapley_weight(size_c: int, K: int) -> float:
    """|C|! (K-|C|-1)! / K!"""
    return math.factorial(size_c) * math.factorial(K - size_c - 1) / math.factorial(K)


def exact_shapley(
    values: Mapping[Tuple[str, ...], float], players: Sequence[str]
) -> Dict[str, float]:
    """Exact Shapley values from a COMPLETE coalition table.

    Raises ValueError if any required coalition is missing (exactness is a
    precondition, never silently approximated).
    """
    players = tuple(players)
    K = len(players)
    required = [tuple(c) for c in _all_subsets(players)]
    missing = [c for c in required if tuple(sorted(c)) not in {tuple(sorted(k)) for k in values}]
    if missing:
        raise ValueError(f"incomplete table; missing coalitions: {missing}")
    lookup = {tuple(sorted(k)): float(v) for k, v in values.items()}
    phi: Dict[str, float] = {}
    for p in players:
        total = 0.0
        others = [q for q in players if q != p]
        for size_c in range(K):
            for C in itertools.combinations(others, size_c):
                C = tuple(sorted(C))
                Cp = tuple(sorted(C + (p,)))
                total += shapley_weight(size_c, K) * (lookup[Cp] - lookup[C])
        phi[p] = total
    return phi


def _all_subsets(players: Sequence[str]) -> List[Tuple[str, ...]]:
    out = []
    for size in range(len(players) + 1):
        for combo in itertools.combinations(players, size):
            out.append(combo)
    return out


def efficiency_residual(phi: Mapping[str, float], v_grand: float) -> float:
    """|sum_p phi_p - v(P)| — reported as the numerical audit quantity."""
    return abs(sum(phi.values()) - v_grand)


def shapley_from_table_records(
    value_tables: Sequence[Mapping[Tuple[str, ...], float]], players: Sequence[str]
) -> Dict[str, Any]:
    """Per-seed exact Shapley vectors, the mean vector, and the maximum
    efficiency residual across all seed-specific games."""
    per_seed = [exact_shapley(table, players) for table in value_tables]
    mean_phi = {
        p: float(np.mean([s[p] for s in per_seed])) for p in players
    }
    residuals = [
        efficiency_residual(s, max(table.values()) if table else 0.0)
        for s, table in zip(per_seed, value_tables)
    ]
    return {
        "per_seed": per_seed,
        "mean": mean_phi,
        "max_efficiency_residual": float(max(residuals)) if residuals else None,
    }


def shapley_of_mean_table(
    value_tables: Sequence[Mapping[Tuple[str, ...], float]], players: Sequence[str]
) -> Dict[str, float]:
    """Shapley of the mean observed table; equals the mean of seed-specific
    vectors by linearity (verified in tests)."""
    mean_table: Dict[Tuple[str, ...], float] = {}
    for coalition in _all_subsets(players):
        mean_table[tuple(sorted(coalition))] = float(
            np.mean([t[tuple(sorted(coalition))] for t in value_tables])
        )
    return exact_shapley(mean_table, players)


def per_user_shapley(
    per_user_tables: Sequence[Mapping[Tuple[str, ...], np.ndarray]],
    players: Sequence[str],
) -> Dict[str, np.ndarray]:
    """phi_{p,u} per user; mean over users recovers the aggregate vector."""
    n_users = len(next(iter(per_user_tables[0].values())))
    phi: Dict[str, np.ndarray] = {p: np.zeros(n_users) for p in players}
    for table in per_user_tables:
        for p in players:
            others = [q for q in players if q != p]
            K = len(players)
            for size_c in range(K):
                for C in itertools.combinations(others, size_c):
                    C = tuple(sorted(C))
                    Cp = tuple(sorted(C + (p,)))
                    phi[p] += shapley_weight(size_c, K) * (table[Cp] - table[C])
    return phi


def normalized_shares(phi: Mapping[str, float], v_grand: float) -> Dict[str, float]:
    """Signed shares phi_p / v(P). Only presented when v(P) is stably away
    from zero; callers must guard against unstable denominators."""
    if abs(v_grand) < 1e-12:
        return {p: math.nan for p in phi}
    return {p: v / v_grand for p, v in phi.items()}


__all__ = [
    "shapley_weight",
    "exact_shapley",
    "efficiency_residual",
    "shapley_from_table_records",
    "shapley_of_mean_table",
    "per_user_shapley",
    "normalized_shares",
]
