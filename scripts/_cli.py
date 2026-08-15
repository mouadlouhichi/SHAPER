"""Shared CLI helpers for the SHAPER scripts (engineering detail).

Adds the repo root to sys.path so scripts run directly from anywhere, and
provides common argument builders for dataset/run_id/recipe resolution.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# Scripts operate on absolute paths derived from the repository root; run
# them with the repo as the working directory so provenance (git commit/dirty
# state) and any relative reads are correct regardless of the launch cwd.
os.chdir(REPO_ROOT)

from shaper.config import load_run_config  # noqa: E402
from shaper.training import Recipe  # noqa: E402


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", choices=["ml1m", "beauty", "ml100k", "synthetic"], default="synthetic",
                   help="dataset to operate on (default synthetic for tests)")
    p.add_argument("--run-id", default=None, help="results/runs/<run_id> to use/create")
    p.add_argument("--device", default=None,
                   help="torch device (default: auto-detect cuda, then mps, then cpu)")
    return p


def resolve_device(args: argparse.Namespace) -> str:
    if args.device:
        return args.device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_recipe(cfg: Any, override: Optional[Dict[str, Any]] = None) -> Recipe:
    """Recipe from config overrides or the frozen manifest (freeze file).

    Synthetic runs pass training.lr/steps/lambda_cl/tau through overrides;
    real runs must read the frozen selected_recipe from
    configs/manifest_freeze.yaml after RECIPE_CALIBRATION.
    """
    t = dict(cfg.training)
    if override:
        t.update(override)
    missing = [k for k in ("learning_rate", "steps", "lambda_cl", "tau") if t.get(k) is None]
    if not missing:
        return Recipe(
            learning_rate=float(t["learning_rate"]),
            steps=int(t["steps"]),
            lambda_cl=float(t["lambda_cl"]),
            tau=float(t["tau"]),
        )
    # try the frozen manifest (verification datasets — synthetic, ml100k —
    # freeze into the results dir; the registered datasets freeze the
    # configs/manifest_freeze.yaml archive)
    from shaper.config import load_yaml

    if cfg.dataset in ("ml1m", "beauty"):
        freeze_path = os.path.join(cfg.paths["configs"], "manifest_freeze.yaml")
    else:
        freeze_path = os.path.join(cfg.paths["results"], f"freeze-synthetic-{cfg.dataset}.yaml")
    if not os.path.exists(freeze_path):
        raise RuntimeError(
            f"recipe not frozen for {cfg.dataset}; run RECIPE_CALIBRATION and "
            "ARCHIVE_FREEZE first"
        )
    freeze = load_yaml(freeze_path)
    sel = (freeze.get("selected_recipe") or {}).get(cfg.dataset) or {}
    vals = {
        "learning_rate": sel.get("learning_rate"),
        "steps": sel.get("steps"),
        "lambda_cl": sel.get("lambda_cl"),
        "tau": sel.get("tau"),
    }
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        raise RuntimeError(
            f"recipe not frozen for {cfg.dataset} (missing {missing}); "
            "run RECIPE_CALIBRATION first or pass --recipe overrides"
        )
    return Recipe(**vals)


def frozen_data(cfg: Any):
    from shaper.data import FrozenData

    data = FrozenData(cfg.dataset, processed_root=os.path.join(cfg.paths["data_processed"], cfg.dataset))
    data.load()
    return data
