"""SHAPER — SHAPley-guided PERturbation learning for contrastive sequential recommendation.

Implements the registered protocol in:
    specs/SHAPER_Implementation_Spec.md   (implementation source of truth)
    specs/SHAPER_Paper_Structure.md       (paper/interpretation source of truth)

Primary estimand: exact K=3 Game A (fixed nominal coefficient budget),
eight coalition models per confirmatory seed, exact Shapley allocation.
Secondary: exact K=3 Game B (fixed per-view dose) on seeds 2001-2003.
Extension: Beauty K=4 Game A, antithetic permutation Monte Carlo,
<= 10 unique coalition models per seed. Frozen NLL/cosine severity
diagnostics only; no additional coalition Shapley sweep.
"""

__version__ = "1.0.0"

PROTOCOL_SPEC = "specs/SHAPER_Implementation_Spec.md"
PAPER_SPEC = "specs/SHAPER_Paper_Structure.md"

# Main K=3 players, in fixed display/tie order.
MAIN_PLAYERS = ("crop", "mask", "reorder")
K4_PLAYERS = ("crop", "mask", "reorder", "dropout")
PLAYER_TIE_ORDER = ("crop", "mask", "reorder", "dropout")

STAGES = [
    "PREFLIGHT",
    "DATA_BUILD",
    "DATA_VALIDATE",
    "RECIPE_CALIBRATION",
    "PILOT_1001_1002",
    "PILOT_VALIDATION",
    "PILOT_AMENDMENT",
    "ARCHIVE_FREEZE",
    "PRIMARY_GAME_A",
    "SECONDARY_GAME_B",
    "SEVERITY_DIAGNOSTICS",
    "K4_BEAUTY_MC",
    "SHAPLEY",
    "LOO",
    "INTERACTIONS",
    "SEGMENTS",
    "POWER",
    "WEIGHT_CALIBRATION",
    "SELECT_CALIBRATION",
    "BASELINE_CONTROLS",
    "FINAL_INTERVENTION_TEST",
    "REPORT",
    "COMPLIANCE_AUDIT",
]

from . import config as config
from . import logging_utils as logging_utils

__all__ = [
    "__version__",
    "PROTOCOL_SPEC",
    "PAPER_SPEC",
    "MAIN_PLAYERS",
    "K4_PLAYERS",
    "PLAYER_TIE_ORDER",
    "STAGES",
]
