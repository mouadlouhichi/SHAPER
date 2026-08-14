"""Configuration loading and validation.

Configs live under `configs/` (dataset configs, seed registry, scope table,
statistical protocol, archive-freeze manifest). This module merges them into
one validated `RunConfig` dataclass and computes the canonical config hash
used by caches and checkpoints.

This is an engineering-detail module (not present in the spec's module list);
it holds no scientific logic.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from .provenance import config_hash as _config_hash

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must be a mapping")
    return data


def configs_dir() -> str:
    return CONFIGS_DIR


@dataclass
class RunConfig:
    """Validated, merged scientific configuration for one dataset."""

    dataset: str
    raw: Dict[str, Any]
    model: Dict[str, Any]
    augmentation: Dict[str, Any]
    training: Dict[str, Any]
    evaluation: Dict[str, Any]
    roles: Dict[str, float]
    cost_estimate: Dict[str, Any]
    seeds: Dict[str, Any]
    statistics: Dict[str, Any]
    scope: Dict[str, Any]
    paths: Dict[str, str] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)

    # -- derived accessors -------------------------------------------------
    @property
    def max_len(self) -> int:
        return int(self.raw["dataset"]["max_len"])

    @property
    def batch_size(self) -> int:
        return int(self.raw["dataset"]["batch_size"])

    @property
    def n_players(self) -> int:
        return 3

    def hash(self) -> str:
        return self.config_hash()

    def config_hash(self, include_recipe: bool = False) -> str:
        payload = {
            "dataset": self.raw["dataset"],
            "model": self.model,
            "augmentation": self.augmentation,
            "training_base": {k: v for k, v in self.training.items()},
            "evaluation": self.evaluation,
            "roles": self.roles,
        }
        if include_recipe:
            payload["training_recipe"] = self.training
        return _config_hash(payload, salt=self.dataset)


def _validate(raw: Dict[str, Any], dataset: str) -> None:
    req_top = ["dataset", "model", "augmentation", "training", "evaluation", "roles"]
    for k in req_top:
        if k not in raw:
            raise ValueError(f"{dataset} config missing top-level key: {k}")
    if raw["dataset"]["name"] != dataset:
        raise ValueError(
            f"dataset name mismatch: config says {raw['dataset']['name']}, expected {dataset}"
        )
    roles = raw["roles"]
    if abs(roles["tune"] + roles["game"] + roles["select"] - 1.0) > 1e-9:
        raise ValueError("validation roles must sum to 1.0")
    if abs(roles["tune"] - 0.2) > 1e-9 or abs(roles["game"] - 0.6) > 1e-9:
        raise ValueError("validation roles must be the locked 20%/60%/20% split")
    aug = raw["augmentation"]
    if aug["min_contrastive_pairs"] != 8:
        raise ValueError("B_min must be fixed at 8")
    if aug["mask"]["protect_last"]:
        raise ValueError("main protocol must not protect the last two mask positions")
    if aug["crop"]["force_nonidentity"] is not True:
        raise ValueError("crop must force non-identity where possible")
    training = raw["training"]
    if training["optimizer"] != "adamw" or training["betas"] != [0.9, 0.98]:
        raise ValueError("optimizer must be AdamW(betas=(0.9, 0.98))")
    if abs(float(training["weight_decay"]) - 1e-4) > 1e-12:
        raise ValueError("weight_decay must be 1e-4")
    if abs(float(training["grad_clip"]) - 1.0) > 1e-12:
        raise ValueError("gradient clipping must be 1.0")


def load_run_config(
    dataset: str,
    repo_root: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> RunConfig:
    """Load and merge all config files for `dataset`.

    `overrides` is a nested dict applied on top of the dataset config (used
    by the synthetic end-to-end test to shrink the model/training budget —
    every override is recorded in the run's provenance).
    """
    root = repo_root or REPO_ROOT
    cfgdir = os.path.join(root, "configs")

    with open(os.path.join(cfgdir, f"{dataset}.yaml"), "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    _validate(raw, dataset)

    if overrides:
        _deep_update(raw, overrides)

    seeds_data = load_yaml(os.path.join(cfgdir, "seeds.yaml"))
    seeds = dict(seeds_data["registry"])
    # non-registry operational keys (not part of the frozen scientific registry)
    for key in ("nondeterminism_floor_repeat_seed", "mc_permutation_seed_offset"):
        if key in seeds_data:
            seeds[key] = seeds_data[key]
    # VERIFICATION-ONLY datasets (e.g. ml100k) declare their own seed set in
    # the dataset config; the frozen scientific registry in seeds.yaml is
    # never touched.
    if raw["dataset"].get("verification_only"):
        seeds = dict(raw.get("test_seeds", seeds))
    statistics = load_yaml(os.path.join(cfgdir, "statistics.yaml"))
    scope = load_yaml(os.path.join(cfgdir, "scope.yaml"))

    paths = {
        "repo_root": root,
        "configs": cfgdir,
        "data_raw": os.path.join(root, "data", "raw"),
        "data_processed": os.path.join(root, "data", "processed"),
        "data_manifests": os.path.join(root, "data", "manifests"),
        "results": os.path.join(root, "results", "runs"),
        "checkpoints": os.path.join(root, "checkpoints"),
    }
    return RunConfig(
        dataset=dataset,
        raw=raw,
        model=raw["model"],
        augmentation=raw["augmentation"],
        training=raw["training"],
        evaluation=raw["evaluation"],
        roles=raw["roles"],
        cost_estimate=raw.get("cost_estimate", {}),
        seeds=seeds,
        statistics=statistics,
        scope=scope,
        paths=paths,
        overrides=dict(overrides or {}),
    )


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def dataset_config_summary(cfg: RunConfig) -> Dict[str, Any]:
    """Compact summary of the frozen scientific settings (for manifests)."""
    return {
        "dataset": cfg.raw["dataset"],
        "model": cfg.model,
        "augmentation": cfg.augmentation,
        "training": cfg.training,
        "evaluation": cfg.evaluation,
        "roles": cfg.roles,
    }


__all__ = [
    "RunConfig",
    "load_run_config",
    "load_yaml",
    "configs_dir",
    "REPO_ROOT",
    "dataset_config_summary",
]
