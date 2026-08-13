"""Antithetic permutation Monte Carlo for the Beauty K=4 Game-A extension.

This is the ONLY Monte Carlo estimator in the registered protocol and the
only estimator permitted for K=4. It is deliberately a dedicated class, not
a generic MC estimator. FastSHAP/KernelSHAP are NOT part of this study.

Per seed s in {3001..3005} (spec A.8, "Monte-Carlo K=4 Beauty extension"):

  1. sample one uniform player permutation pi_s keyed by the seed;
  2. use reverse(pi_s) as the antithetic permutation;
  3. train/cache every unique prefix coalition on both paths (empty/full
     shared): at most 8 path coalitions;
  4. train/cache all four P\\{p} coalitions required for exact grand-LOO,
     deduplicated against the paths: <= 10 unique models per seed;
  5. estimate each player's Shapley value as the mean of its marginal
     contribution on pi_s and reverse(pi_s);
  6. aggregate across the five seeds and report nested seed/permutation
     uncertainty; the estimator preserves telescoping:
         sum_p marginal_p = v(P) - v(empty).

Result labels:
  - allocation: "Monte-Carlo approximate" — never "exact"
  - ordering interpretation only if the 95% MC half-width <= delta_phi;
    otherwise INCONCLUSIVE (no models may be added to rescue it).
  - exact K=4 Grabisch-Roubens interactions are FORBIDDEN from this
    incomplete table (raises ValueError in shaper.interactions).

The permutation is part of the frozen run state: resuming after an
interruption must reuse pi_s, never resample it.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .provenance import key_int

MAX_UNIQUE_MODELS_PER_SEED = 10


def sample_permutation(seed: int, players: Sequence[str]) -> Tuple[str, ...]:
    """One uniform permutation keyed by the seed (frozen run state)."""
    players = list(players)
    rng = random.Random(key_int(seed, "k4-permutation", salt="shaper-mc-v1"))
    rng.shuffle(players)
    return tuple(players)


def reverse_permutation(pi: Sequence[str]) -> Tuple[str, ...]:
    return tuple(reversed(pi))


def prefix_path(pi: Sequence[str]) -> List[Tuple[str, ...]]:
    """All nonempty proper prefixes of a permutation path (in path order)."""
    return [tuple(pi[: i + 1]) for i in range(len(pi) - 1)]


def required_coalitions(pi: Sequence[str], players: Sequence[str]) -> Dict[str, Tuple[str, ...]]:
    """Deduplicated coalition registry for one K=4 seed.

    Returns {canonical_name: coalition} containing:
      - empty and grand (shared),
      - prefix coalitions of pi and reverse(pi),
      - all four P\\{p} grand-LOO coalitions.
    Never more than MAX_UNIQUE_MODELS_PER_SEED (10) unique coalitions.
    """
    players = tuple(players)
    K = len(players)
    registry: Dict[str, Tuple[str, ...]] = {}

    def _add(coalition: Tuple[str, ...]) -> None:
        name = "+".join(sorted(coalition)) if coalition else "empty"
        if name in registry:
            if registry[name] != tuple(sorted(coalition)):
                raise RuntimeError("coalition registry collision")
            return
        if len(registry) >= MAX_UNIQUE_MODELS_PER_SEED:
            raise RuntimeError(
                f"K=4 MC budget exceeded: attempting model #{len(registry) + 1} "
                f"(hard cap {MAX_UNIQUE_MODELS_PER_SEED} unique coalitions per seed). "
                "Do NOT add models to rescue an inconclusive result."
            )
        registry[name] = tuple(sorted(coalition))

    _add(())
    rev = reverse_permutation(pi)
    for path in (pi, rev):
        for coalition in prefix_path(path):
            _add(coalition)
    _add(tuple(players))
    for p in players:
        _add(tuple(q for q in players if q != p))
    if len(registry) > MAX_UNIQUE_MODELS_PER_SEED:
        raise RuntimeError(
            f"K=4 MC registry has {len(registry)} unique coalitions "
            f"(cap {MAX_UNIQUE_MODELS_PER_SEED}); this violates the locked scope."
        )
    return registry


def path_marginals(
    path: Sequence[str], values: Dict[Tuple[str, ...], float], players: Sequence[str]
) -> Dict[str, float]:
    """Marginal contribution of each player along one permutation path."""
    lookup = {tuple(sorted(k)): v for k, v in values.items()}
    out: Dict[str, float] = {}
    prev: Tuple[str, ...] = ()
    for p in path:
        curr = tuple(sorted(prev + (p,)))
        if curr not in lookup or prev not in lookup:
            raise ValueError(f"path coalition missing: {curr} or {prev}")
        out[p] = lookup[curr] - lookup[prev]
        prev = curr
    return out


def telescoping_check(
    path: Sequence[str], values: Dict[Tuple[str, ...], float], players: Sequence[str]
) -> Dict[str, Any]:
    """sum_p marginal_p == v(P) - v(empty) within numerical precision."""
    lookup = {tuple(sorted(k)): v for k, v in values.items()}
    marginals = path_marginals(path, values, players)
    total = sum(marginals.values())
    residual = total - (lookup[tuple(sorted(players))] - lookup[()])
    return {
        "sum_marginals": total,
        "v_grand_minus_empty": lookup[tuple(sorted(players))] - lookup[()],
        "residual": float(residual),
        "within_precision": abs(residual) <= 1e-9,
        "marginals": marginals,
    }


@dataclass
class K4SeedResult:
    seed: int
    permutation: Tuple[str, ...]
    reverse_permutation: Tuple[str, ...]
    coalition_registry: Dict[str, Tuple[str, ...]]
    values: Dict[Tuple[str, ...], float] = field(default_factory=dict)
    completed: List[str] = field(default_factory=list)
    estimates: Dict[str, float] = field(default_factory=dict)
    antithetic_marginals: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "permutation": list(self.permutation),
            "reverse_permutation": list(self.reverse_permutation),
            "coalition_registry": self.coalition_registry,
            "values": {("+".join(sorted(k)) if k else "empty"): v for k, v in self.values.items()},
            "completed": self.completed,
            "estimates": self.estimates,
            "antithetic_marginals": self.antithetic_marginals,
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)

    @classmethod
    def load(cls, path: str) -> "K4SeedResult":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        values = {}
        for name, v in d["values"].items():
            coalition = tuple(sorted(name.split("+"))) if name != "empty" else ()
            values[coalition] = v
        registry = {
            name: tuple(sorted(co)) if co else () for name, co in d["coalition_registry"].items()
        }
        return cls(
            seed=d["seed"],
            permutation=tuple(d["permutation"]),
            reverse_permutation=tuple(d["reverse_permutation"]),
            coalition_registry=registry,
            values=values,
            completed=d["completed"],
            estimates=d["estimates"],
            antithetic_marginals=d["antithetic_marginals"],
        )


class AntitheticPermutationMCShapley:
    """Dedicated antithetic permutation-MC estimator for the K=4 extension."""

    def __init__(self, players: Sequence[str]):
        self.players = tuple(players)
        self.K = len(players)
        if self.K != 4:
            raise ValueError("this estimator is registered for K=4 only")
        self.seed_results: List[K4SeedResult] = []

    # -- per-seed sampling ----------------------------------------------------
    def new_seed_result(self, seed: int) -> K4SeedResult:
        pi = sample_permutation(seed, self.players)
        registry = required_coalitions(pi, self.players)
        return K4SeedResult(
            seed=seed,
            permutation=pi,
            reverse_permutation=reverse_permutation(pi),
            coalition_registry=registry,
        )

    def set_values(self, result: K4SeedResult, values: Dict[Tuple[str, ...], float]) -> K4SeedResult:
        """Load realized coalition values and compute antithetic estimates."""
        for coalition in result.coalition_registry.values():
            if coalition not in {tuple(sorted(k)) for k in values}:
                raise ValueError(f"missing realized value for coalition {coalition}")
        lookup = {tuple(sorted(k)): float(v) for k, v in values.items()}
        result.values = {tuple(sorted(k)): float(v) for k, v in values.items()}
        marg_pi = path_marginals(result.permutation, lookup, self.players)
        marg_rev = path_marginals(result.reverse_permutation, lookup, self.players)
        result.antithetic_marginals = {"pi": marg_pi, "reverse_pi": marg_rev}
        result.estimates = {
            p: 0.5 * (marg_pi[p] + marg_rev[p]) for p in self.players
        }
        self.seed_results.append(result)
        return result

    # -- aggregation -----------------------------------------------------------
    def aggregate(
        self, delta_phi: float = 0.003, t_quantile: Optional[float] = None
    ) -> Dict[str, Any]:
        """Aggregate across seeds with nested seed/permutation uncertainty.

        For each player:
          seed_mean  = mean_s estimate_ps
          SE_seed    = SD(estimate_ps) / sqrt(S)          (seed-level)
          SE_perm   = sqrt( mean_s ((m_pi - m_rev)/2)^2 / S )  (permutation-level)
          nested SE  = sqrt(SE_seed^2 + SE_perm^2)
          95% MC half-width = t_{0.975, S-1} * nested SE
        Interpretation: ordering only if every relevant player's half-width
        <= delta_phi; otherwise INCONCLUSIVE. No models are added to rescue.
        """
        S = len(self.seed_results)
        if S == 0:
            raise ValueError("no seed results to aggregate")
        if t_quantile is None:
            from scipy.stats import t as _t

            t_quantile = float(_t.ppf(0.975, S - 1)) if S > 1 else math.inf

        out: Dict[str, Any] = {
            "label": "Monte-Carlo approximate",
            "n_seeds": S,
            "players": list(self.players),
            "per_player": {},
            "permutations": [
                {"seed": r.seed, "pi": list(r.permutation)} for r in self.seed_results
            ],
            "telescoping": [],
        }
        for p in self.players:
            ests = np.array([r.estimates[p] for r in self.seed_results])
            spread = np.array(
                [
                    r.antithetic_marginals["pi"][p] - r.antithetic_marginals["reverse_pi"][p]
                    for r in self.seed_results
                ]
            )
            seed_mean = float(ests.mean())
            se_seed = float(ests.std(ddof=1) / math.sqrt(S)) if S > 1 else math.nan
            se_perm = float(math.sqrt(((spread / 2.0) ** 2).mean() / S))
            se_nested = float(math.sqrt(se_seed**2 + se_perm**2)) if S > 1 else se_perm
            half_width = t_quantile * se_nested
            out["per_player"][p] = {
                "estimate": seed_mean,
                "per_seed_estimates": ests.tolist(),
                "se_seed": se_seed,
                "se_perm": se_perm,
                "se_nested": se_nested,
                "half_width_95": float(half_width),
                "interpretable": bool(half_width <= delta_phi),
            }
        for r in self.seed_results:
            out["telescoping"].append(telescoping_check(r.permutation, r.values, self.players))
            out["telescoping"].append(telescoping_check(r.reverse_permutation, r.values, self.players))
        all_interpretable = all(out["per_player"][p]["interpretable"] for p in self.players)
        if all_interpretable:
            order = sorted(self.players, key=lambda p: out["per_player"][p]["estimate"], reverse=True)
            out["ordering"] = {"status": "interpretable", "order": order}
        else:
            out["ordering"] = {
                "status": "INCONCLUSIVE",
                "reason": "MC half-width exceeds delta_phi for at least one player; "
                          "no additional models were trained",
            }
        out["efficiency_note"] = (
            "antithetic averaging preserves telescoping: every path satisfies "
            "sum_p marginal_p = v(P) - v(empty) within 1e-9"
        )
        out["interactions"] = {
            "status": "NOT_COMPUTED",
            "reason": "exact K=4 Grabisch-Roubens interactions are outside scope: "
                      "the K=4 value table is incomplete by design",
        }
        out["grand_loo"] = {
            p: r.values[tuple(sorted(self.players))]
            - r.values[tuple(sorted(q for q in self.players if q != p))]
            for p in self.players
            for r in self.seed_results
        } if S else {}
        return out


def grand_loo_from_seed(
    result: K4SeedResult, players: Sequence[str]
) -> Dict[str, float]:
    """Exact grand-LOO available because all four P\\{p} models were trained."""
    lookup = {tuple(sorted(k)): v for k, v in result.values.items()}
    P = tuple(players)
    return {p: lookup[P] - lookup[tuple(q for q in P if q != p)] for p in players}


__all__ = [
    "MAX_UNIQUE_MODELS_PER_SEED",
    "sample_permutation",
    "reverse_permutation",
    "prefix_path",
    "required_coalitions",
    "path_marginals",
    "telescoping_check",
    "K4SeedResult",
    "AntitheticPermutationMCShapley",
    "grand_loo_from_seed",
]
