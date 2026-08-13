"""The coalition games: characteristic function, value tables, per-user
utilities, grand-LOO and contextual marginals (spec A.8).

    v_s(C) = M_game(C, s) - M_game(empty, s),   v_s(empty) = 0

where M_game is full-catalog NDCG@10 on the fixed V_game users of a seed.
The rec-only empty model is shared by Games A/B within a seed. Per-user
utilities make the game linear over users (no extra model fits):

    v_s(C) = (1/|U|) sum_u v_{u,s}(C)

Coalition training is the expensive operation; all allocations below are
arithmetic over a realized value table.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import FrozenData
from .metrics import evaluate_model


# --------------------------------------------------------------------------
# Coalition enumeration
# --------------------------------------------------------------------------

def all_coalitions(players: Sequence[str]) -> List[Tuple[str, ...]]:
    """All 2^K subsets of the player set (deterministic order)."""
    players = tuple(players)
    out = []
    for size in range(len(players) + 1):
        for combo in itertools.combinations(players, size):
            out.append(tuple(combo))
    return out


def intermediate_coalitions(players: Sequence[str]) -> List[Tuple[str, ...]]:
    """The 2^K - 2 distinct intermediate (nonempty, non-grand) coalitions —
    used by secondary Game B; empty and grand are reused from Game A."""
    return [c for c in all_coalitions(players) if 0 < len(c) < len(players)]


def coalition_name(coalition: Sequence[str]) -> str:
    return "+".join(sorted(coalition)) if coalition else "empty"


# --------------------------------------------------------------------------
# Value records
# --------------------------------------------------------------------------

@dataclass
class CoalitionValueRecord:
    dataset: str
    seed: int
    policy: str
    coalition: Tuple[str, ...]
    metrics_by_role: Dict[str, Dict[str, float]] = field(default_factory=dict)
    per_user_ndcg_game: Optional[List[float]] = None
    per_user_nll_game: Optional[List[float]] = None
    value_ndcg: float = 0.0
    value_nll: float = 0.0
    repeated_target_rate_game: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "seed": self.seed,
            "policy": self.policy,
            "coalition": list(self.coalition),
            "metrics_by_role": self.metrics_by_role,
            "value_ndcg": self.value_ndcg,
            "value_nll": self.value_nll,
            "repeated_target_rate_game": self.repeated_target_rate_game,
        }


def evaluate_and_record(
    model: "Any",
    data: FrozenData,
    dataset: str,
    seed: int,
    policy: str,
    coalition: Sequence[str],
    k: int,
    roles: Sequence[str] = ("tune", "game", "select"),
    ordering: str = "primary",
) -> CoalitionValueRecord:
    """Evaluate one trained coalition model on the frozen role splits."""
    rec = CoalitionValueRecord(
        dataset=dataset,
        seed=seed,
        policy=policy,
        coalition=tuple(sorted(coalition)),
    )
    for role in roles:
        inputs = data.eval_inputs(role, ordering=ordering)
        res = evaluate_model(model, inputs, k)
        rec.metrics_by_role[role] = res.to_dict()
        if role == "game":
            rec.per_user_ndcg_game = res.per_user_ndcg
            rec.per_user_nll_game = res.per_user_nll
            rec.repeated_target_rate_game = res.repeated_target_rate
    return rec


def apply_baseline_subtraction(
    records: Sequence[CoalitionValueRecord],
    metric: str = "ndcg",
    role: str = "game",
) -> Dict[Tuple[str, ...], float]:
    """v(C) = M(C) - M(empty); v(empty) = 0 exactly."""
    by_coalition = {rec.coalition: rec for rec in records}
    empty = by_coalition.get(())
    if empty is None:
        raise ValueError("empty-coalition record required for baseline subtraction")
    metric_key = {"ndcg": "ndcg", "nll": "nll"}.get(metric, metric)
    m_empty = empty.metrics_by_role[role][metric_key]
    out: Dict[Tuple[str, ...], float] = {}
    for coalition, rec in by_coalition.items():
        if not coalition:
            out[coalition] = 0.0
            continue
        m = rec.metrics_by_role[role][metric_key]
        if metric == "nll":
            # smooth utility: v^NLL(C) = -NLL(C) + NLL(empty)
            out[coalition] = m_empty - m
        else:
            out[coalition] = m - m_empty
    return out


def subtract_baseline(records: Sequence[CoalitionValueRecord], metric: str = "ndcg", role: str = "game") -> Sequence[CoalitionValueRecord]:
    values = apply_baseline_subtraction(records, metric=metric, role=role)
    for rec in records:
        if metric == "ndcg":
            rec.value_ndcg = values[rec.coalition]
        elif metric == "nll":
            rec.value_nll = values[rec.coalition]
    return records


def per_user_values(
    records: Sequence[CoalitionValueRecord], metric: str = "ndcg"
) -> Dict[Tuple[str, ...], np.ndarray]:
    """v_u(C) = m_u(C) - m_u(empty) for every V_game user (linearity basis).

    Requires every record to carry per-user game-role utilities. Returns
    arrays aligned over the same user set.
    """
    by_coalition = {rec.coalition: rec for rec in records}
    empty = by_coalition.get(())
    if empty is None:
        raise ValueError("empty-coalition record required")
    attr = "per_user_ndcg_game" if metric == "ndcg" else "per_user_nll_game"
    base = getattr(empty, attr)
    if base is None:
        raise ValueError("per-user game utilities missing on empty record")
    base_arr = np.asarray(base, dtype=np.float64)
    out: Dict[Tuple[str, ...], np.ndarray] = {}
    for coalition, rec in by_coalition.items():
        vals = getattr(rec, attr)
        if vals is None:
            raise ValueError(f"per-user game utilities missing for coalition {coalition}")
        arr = np.asarray(vals, dtype=np.float64)
        if len(arr) != len(base_arr):
            raise ValueError(
                f"user-set mismatch for coalition {coalition}: {len(arr)} vs {len(base_arr)}"
            )
        out[coalition] = arr - base_arr
    return out


# --------------------------------------------------------------------------
# LOO, contextual marginals, monotonicity
# --------------------------------------------------------------------------

def grand_loo(values: Dict[Tuple[str, ...], float], players: Sequence[str]) -> Dict[str, float]:
    """LOO_p = v(P) - v(P \\ {p})."""
    P = tuple(players)
    out: Dict[str, float] = {}
    for p in players:
        without = tuple(q for q in P if q != p)
        if without not in values:
            raise ValueError(f"grand-LOO requires v(P\\{{{p}}})")
        out[p] = values[P] - values[without]
    return out


def contextual_marginals(
    values: Dict[Tuple[str, ...], float], players: Sequence[str]
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    """Every marginal v(C u {p}) - v(C) for C subset of P\\{p}."""
    out: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    for p in players:
        for size in range(len(players)):
            for C in itertools.combinations([q for q in players if q != p], size):
                Cp = tuple(sorted(C + (p,)))
                out[(p, C)] = values[Cp] - values[C]
    return out


def monotonicity_violations(
    values: Dict[Tuple[str, ...], float], players: Sequence[str]
) -> List[Dict[str, Any]]:
    """All v(C u {p}) < v(C) events (reported and diagnosed, not auto-dropped)."""
    viol: List[Dict[str, Any]] = []
    for p in players:
        for size in range(len(players)):
            for C in itertools.combinations([q for q in players if q != p], size):
                Cp = tuple(sorted(C + (p,)))
                delta = values[Cp] - values[C]
                if delta < 0:
                    viol.append(
                        {"player": p, "context": list(C), "coalition": list(Cp),
                         "marginal": float(delta)}
                    )
    return viol


def epsilon_pq(
    values: Dict[Tuple[str, ...], float], players: Sequence[str], p: str, q: str
) -> float:
    """Empirical proximity to perfect substitutability (spec A.8):

    eps_pq = max over contexts of |v(Cp)-v(Cq)|, |v(Cpq)-v(Cp)|, |v(Cpq)-v(Cq)|.
    """
    rest = [r for r in players if r not in (p, q)]
    max_val = 0.0
    for size in range(len(rest) + 1):
        for C in itertools.combinations(rest, size):
            Cp = tuple(sorted(C + (p,)))
            Cq = tuple(sorted(C + (q,)))
            Cpq = tuple(sorted(C + (p, q)))
            max_val = max(
                max_val,
                abs(values[Cp] - values[Cq]),
                abs(values[Cpq] - values[Cp]),
                abs(values[Cpq] - values[Cq]),
            )
    return float(max_val)


def coalition_spread(values: Dict[Tuple[str, ...], float]) -> Dict[str, float]:
    nonempty = [v for c, v in values.items() if c]
    if not nonempty:
        return {"min": 0.0, "max": 0.0, "range": 0.0, "sd": 0.0}
    arr = np.asarray(nonempty, dtype=np.float64)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "sd": float(arr.std()),
    }


def save_coalition_table(
    records: Sequence[CoalitionValueRecord], path: str
) -> None:
    """Persist the value table (JSON) plus the per-user game utilities (npz)
    so per-user Shapley decomposition needs no additional model fits."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in records], fh, indent=2, sort_keys=True)
    pu = {
        "+".join(rec.coalition) if rec.coalition else "empty": rec.per_user_ndcg_game
        for rec in records
    }
    if any(v is not None for v in pu.values()):
        import numpy as np

        np.savez_compressed(os.path.splitext(path)[0] + ".npz", **{
            k: v if v is not None else np.array([]) for k, v in pu.items()
        })


def load_per_user_table(path: str) -> Optional[Dict[Tuple[str, ...], "np.ndarray"]]:
    """Load the companion npz of a saved coalition table (if present)."""
    import numpy as np

    npz_path = os.path.splitext(path)[0] + ".npz"
    if not os.path.exists(npz_path):
        return None
    z = np.load(npz_path)
    return {
        tuple(sorted(k.split("+"))) if k != "empty" else (): z[k] for k in z.files
    }


def load_coalition_table(path: str) -> List[CoalitionValueRecord]:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    out = []
    for item in raw:
        out.append(
            CoalitionValueRecord(
                dataset=item["dataset"],
                seed=item["seed"],
                policy=item["policy"],
                coalition=tuple(item["coalition"]),
                metrics_by_role=item["metrics_by_role"],
                value_ndcg=item["value_ndcg"],
                value_nll=item["value_nll"],
                repeated_target_rate_game=item["repeated_target_rate_game"],
            )
        )
    return out


__all__ = [
    "all_coalitions",
    "intermediate_coalitions",
    "coalition_name",
    "CoalitionValueRecord",
    "evaluate_and_record",
    "apply_baseline_subtraction",
    "subtract_baseline",
    "grand_loo",
    "contextual_marginals",
    "monotonicity_violations",
    "epsilon_pq",
    "coalition_spread",
    "save_coalition_table",
    "load_coalition_table",
]
