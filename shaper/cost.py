"""Compute-budget estimator (spec: "BUDGET ESTIMATOR").

Before any expensive execution, estimate:
    coalition models, training steps, estimated GPU hours, checkpoint
    storage, and result storage for the declared scope.

All numbers here are PLANNING ESTIMATES derived from the per-dataset
nominal per-coalition cost ranges and fixed step counts. The paper must
report measured values only (see shaper.artifacts.ComputeTracker).

This is an engineering-detail module; it holds no scientific logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .config import RunConfig

# nominal per-checkpoint storage sizes (bytes) — planning only
BYTES_PER_CHECKPOINT = 8 * 1024 * 1024  # ~8 MB model+optimizer state
BYTES_PER_METRIC_ROW = 256


@dataclass
class CostBreakdown:
    items: List[Dict[str, object]] = field(default_factory=list)
    total_coalition_models: int = 0
    total_training_steps: int = 0
    estimated_gpu_hours: Dict[str, float] = field(default_factory=dict)
    estimated_checkpoint_gb: float = 0.0
    estimated_result_gb: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def add(self, label: str, models: int, steps_per_model: int, note: str = "") -> None:
        self.items.append(
            {
                "label": label,
                "coalition_models": models,
                "steps_per_model": steps_per_model,
                "total_steps": models * steps_per_model,
                "note": note,
            }
        )
        self.total_coalition_models += models
        self.total_training_steps += models * steps_per_model

    def summary(self) -> Dict[str, object]:
        return {
            "items": self.items,
            "total_coalition_models": self.total_coalition_models,
            "total_training_steps": self.total_training_steps,
            "estimated_gpu_hours_range": self.estimated_gpu_hours,
            "estimated_checkpoint_storage_gb": round(self.estimated_checkpoint_gb, 3),
            "estimated_result_storage_gb": round(self.estimated_result_gb, 3),
            "warnings": self.warnings,
            "planning_note": (
                "PLANNING ESTIMATES ONLY. The paper reports measured runtime, "
                "memory and energy; these ranges are never presented as results."
            ),
        }


def n_coalition_models_for_scope(cfg: RunConfig, frozen_steps: Optional[int] = None) -> CostBreakdown:
    """Estimate the full declared study for one dataset.

    Coalition-model counts follow the locked scope table:
        primary Game A:     8 x 5 seeds (+8 x 5 if the direction-blind
                            extension triggers)
        pilots:             8 x 2 (excluded)
        recipe calibration: 9 x 5000-step LR models + (3 seeds x 3 steps x 2
                            coalitions) empty/grand curve models + 9 + 2x2
                            lambda/tau grid models
        Game B:             6 x 3 (intermediate only; empty/grand reused)
        severity:           4 nonempty pilot models + forward-only scoring
        K=4 MC:             <= 10 x 5 (Beauty only)
        interventions:      alpha path, LOO, gates, direct search, Dirichlet,
                            final Weight seeds (upper bound)
    """
    steps = frozen_steps or (cfg.training["steps"] or 0)
    cb = CostBreakdown()

    S = len(cfg.seeds["confirmatory_game_a"])
    ext = len(cfg.seeds["extension_game_a"])
    pilots = len(cfg.seeds["pilots"])

    cb.add("recipe_calibration_step1_lr", 3, 5000, "empty coalition, seed 901, 3 LRs")
    cb.add(
        "recipe_calibration_step2_curve",
        3 * 3 * 2,
        max(steps, 10000) if steps else 10000,
        "seeds 901-903 x {2500,5000,10000} x empty+grand (checkpoints reused)",
    )
    cb.add("recipe_calibration_step3_lambda_tau", 9 + 2 * 2, steps, "seed 901 (9) + advance 2 pairs to 902,903")
    cb.add("pilot_game_a", 8 * pilots, steps, "excluded from confirmatory estimates")
    cb.add("primary_game_a", 8 * S, steps, "exact K=3, five confirmatory seeds")
    if cfg.dataset == "beauty":
        cb.add("k4_beauty_mc", 10 * len(cfg.seeds["k4_beauty_mc"]), steps, "<=10 unique models/seed, seeds 3001-3005")
    cb.add("game_b", 6 * len(cfg.seeds["game_b"]), steps, "intermediate coalitions only, seeds 2001-2003")
    cb.add("severity_diagnostics", 4, steps, "pilot 1002: 3 matched singletons + matched grand; frozen scoring otherwise")
    cb.add("singleton_lambda_stress", 3, steps, "seed 901, V_tune, {0.05,0.1,0.2}")
    cb.add(
        "interventions_upper_bound",
        2 * (5 * 1 + 5 * 1 + 3 + 1 + 15 + 10 + 2 * 5),
        steps,
        "Weight path + LOO path + gates entropy x3 + direct-search 15 + Dirichlet 10 "
        "+ final Weight/uniform 5-seed evaluation (upper bound; several reuse coalition models)",
    )
    if cfg.dataset == "beauty":
        cb.add("cached_adapter_appendix", 8, steps, "optional, seed 4001 (upper bound)")

    # Conditional extension estimate (direction-blind trigger only).
    ext_items = cb.items[-1:]
    _ = ext_items
    if ext:
        cb.add(
            "conditional_extension_game_a",
            8 * ext,
            steps,
            "ONLY if the locked direction-blind precision rule triggers (2006-2010)",
        )

    # GPU-hour estimate from per-coalition nominal cost range.
    lo, hi = cfg.cost_estimate.get("minutes_per_coalition", [15, 30])
    models = cb.total_coalition_models
    cb.estimated_gpu_hours = {
        "low_hours": round(models * lo / 60.0, 2),
        "high_hours": round(models * hi / 60.0, 2),
        "minutes_per_coalition_range": [lo, hi],
    }
    cb.estimated_checkpoint_gb = (
        models * (steps / max(cfg.training["checkpoint_interval"], 1)) * BYTES_PER_CHECKPOINT / 1e9
    )
    cb.estimated_result_gb = models * 64 * BYTES_PER_METRIC_ROW / 1e9
    if not steps:
        cb.warnings.append(
            "training steps not frozen yet — run RECIPE_CALIBRATION before relying on cost figures"
        )
    cb.warnings.append(
        "the project must never launch the entire study from one notebook cell; "
        "use scripts/run_all.py --stage <stage>"
    )
    return cb


def estimate_study_cost(cfgs: Dict[str, RunConfig]) -> Dict[str, object]:
    """Cost estimate across all declared datasets."""
    out: Dict[str, object] = {}
    for name, cfg in cfgs.items():
        out[name] = n_coalition_models_for_scope(cfg).summary()
    return {"per_dataset": out, "planning": "estimates; measured values replace these in the paper"}


__all__ = ["CostBreakdown", "n_coalition_models_for_scope", "estimate_study_cost"]
