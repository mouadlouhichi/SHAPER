"""Shared fixtures for the SHAPER test suite.

The tiny synthetic configuration keeps every test fast on CPU while
exercising the exact same code paths as the registered study.
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from shaper.config import load_run_config  # noqa: E402
from shaper.data import (  # noqa: E402
    FrozenData,
    build_dataset_artifact,
    build_synthetic_interactions,
)
from shaper.training import Recipe  # noqa: E402

torch.set_num_threads(2)


@pytest.fixture(scope="session")
def tiny_cfg():
    cfg = load_run_config(
        "synthetic",
        overrides={
            "training": {
                "steps": 24,
                "checkpoint_interval": 12,
                "lr": 5.0e-4,
                "lambda_cl": 0.1,
                "tau": 0.1,
            }
        },
    )
    return cfg


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory, tiny_cfg):
    root = tmp_path_factory.mktemp("synthetic-data")
    raw = build_synthetic_interactions(n_users=64, n_items=32, min_len=6, max_len=10, seed=7)
    build_dataset_artifact(tiny_cfg, raw, processed_root=str(root), force=True)
    return FrozenData("synthetic", processed_root=str(root), cfg=tiny_cfg).load()


@pytest.fixture(scope="session")
def tiny_recipe():
    return Recipe(learning_rate=5.0e-4, steps=24, lambda_cl=0.1, tau=0.1)


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory, tiny_cfg):
    from shaper.artifacts import RunDirectory

    root = tmp_path_factory.mktemp("synthetic-run")
    run = RunDirectory("tiny", results_root=str(root)).create(tiny_cfg)
    return run


@pytest.fixture(scope="session")
def train_ctx(tmp_path_factory, tiny_cfg, tiny_data, tiny_recipe):
    """A TrainContext writing checkpoints into a temp dir."""
    from shaper.checkpoints import CheckpointManifest
    from shaper.logging_utils import StructuredLogger
    from shaper.training import TrainContext

    root = tmp_path_factory.mktemp("synthetic-ckpt")
    logger = StructuredLogger(str(root), run_id="test")
    manifest = CheckpointManifest(str(root))
    return TrainContext(
        cfg=tiny_cfg,
        data=tiny_data,
        recipe=tiny_recipe,
        seed=2001,
        policy="game_a",
        run_dir=str(root),
        logger=logger,
        manifest=manifest,
        config_hash=tiny_cfg.config_hash(),
        device="cpu",
        log_interval=6,
        checkpoint_interval=12,
        base_state_path=os.path.join(str(root), "seed2001", "init.pt"),
    )
