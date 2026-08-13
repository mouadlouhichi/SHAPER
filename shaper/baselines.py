"""Fair control baselines for RQ4 (spec A.10, paper 4.6).

- rec-only: reuse the Game-A empty coalition model (no new training)
- uniform: reuse the Game-A grand coalition model (1/K weights == Game A
  grand objective for K=3)
- LOO-derived weights: positive-transform of grand-LOO, same clipping and
  coarse alpha grid as SHAPER-Weight
- learned dataset-level softmax gates: training-only, gate LR 1e-2, entropy
  coefficient selected from {0, 0.01, 0.1} on V_select, zero-init (uniform)
- drop-lowest-LOO: reuse the enumerated P\\{argmin LOO} coalition model
- random removal: keyed random view removal, reusing enumerated models
- 15-point simplex direct search: one declared calibration initialization,
  prespecified deterministic design, five-seed confirmation of the winner
- ten Dirichlet reference candidates on the same calibration initialization
  (a V_select reference distribution, not ten test-tuned methods)

No control may create an unfair test-tuned oracle; all budgets are declared
and recorded (validation budget, training runs, calibration cost, final seed
count, training time).
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adaptive import positive_transform
from .provenance import key_int


# --------------------------------------------------------------------------
# Weight-vector designs
# --------------------------------------------------------------------------

def simplex_design_15(n_views: int = 3, resolution: int = 6) -> List[Dict[str, float]]:
    """Prespecified 15-point simplex design (K=3): all compositions of
    `resolution` (6) into `n_views` nonnegative parts (step 1/6), sorted by
    Euclidean distance from uniform, first 15 points. Deterministic."""
    if n_views != 3:
        raise ValueError("the locked 15-point design is defined for K=3")
    points: List[Tuple[float, ...]] = []
    for a in range(resolution + 1):
        for b in range(resolution + 1 - a):
            c = resolution - a - b
            points.append((a / resolution, b / resolution, c / resolution))
    uniform = np.array([1 / n_views] * n_views)
    points.sort(key=lambda p: float(np.linalg.norm(np.array(p) - uniform)))
    selected = points[:15]
    return [{"crop": p[0], "mask": p[1], "reorder": p[2]} for p in selected]


def dirichlet_reference_candidates(
    n_candidates: int = 10, seed: int = 901, n_views: int = 3
) -> List[Dict[str, float]]:
    """Ten Dirichlet(1) reference weight vectors on the calibration seed."""
    rng = random.Random(key_int(seed, "dirichlet-reference", salt="shaper-controls"))
    views = ["crop", "mask", "reorder"][:n_views]
    out = []
    for _ in range(n_candidates):
        draw = np.random.default_rng(rng.randrange(2**32)).dirichlet(np.ones(n_views))
        out.append({v: float(draw[i]) for i, v in enumerate(views)})
    return out


def loo_derived_weights(loo: Dict[str, float], alpha: float) -> Dict[str, float]:
    """LOO-derived nonnegative weights with the same clipping and alpha
    shrinkage as SHAPER-Weight."""
    q = positive_transform(loo)
    K = len(loo)
    return {p: (1.0 - alpha) / K + alpha * q[p] for p in loo}


def drop_lowest_loo(players: Sequence[str], loo: Dict[str, float]) -> Dict[str, Any]:
    """Reuse-based control: remove the view with the lowest grand-LOO."""
    lowest = min(players, key=lambda p: loo.get(p, 0.0))
    return {
        "removed_view": lowest,
        "selected_coalition": sorted(p for p in players if p != lowest),
        "note": "reuses the already enumerated coalition model",
    }


def random_removal(players: Sequence[str], key_seed: int) -> Dict[str, Any]:
    """Keyed random view removal (reuses an enumerated coalition model)."""
    rng = random.Random(key_int(key_seed, "random-removal", salt="shaper-controls"))
    removed = rng.choice(list(players))
    return {
        "removed_view": removed,
        "selected_coalition": sorted(p for p in players if p != removed),
        "note": "keyed random control; reuses the enumerated coalition model",
    }


# --------------------------------------------------------------------------
# Control budget bookkeeping
# --------------------------------------------------------------------------

def control_budget_record() -> Dict[str, Any]:
    """All controls must record: validation budget, training runs,
    calibration cost, final seed count, training time (spec section 40)."""
    return {
        "validation_budget": {"role": "V_select", "note": "fixed users, never test"},
        "training_runs": [],
        "calibration_cost": {},
        "final_seed_count": None,
        "training_time_seconds": 0.0,
    }


__all__ = [
    "simplex_design_15",
    "dirichlet_reference_candidates",
    "loo_derived_weights",
    "drop_lowest_loo",
    "random_removal",
    "control_budget_record",
]
