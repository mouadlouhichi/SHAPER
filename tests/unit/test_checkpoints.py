"""Unit tests: atomic checkpointing, manifest bookkeeping, corruption
rejection, and cache-key validation."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.checkpoints import (
    CheckpointManifest,
    atomic_write_json,
    load_coalition_checkpoint,
    save_coalition_checkpoint,
    validate_cache_keys,
    validate_checkpoint_file,
)
from shaper.backbone import build_model
from shaper.config import load_run_config


@pytest.fixture()
def payload_dir(tmp_path):
    return str(tmp_path)


def _dummy_payload():
    """A checkpoint payload with a real model module (needed by
    save_coalition_checkpoint) plus a 'dataset' field for cache-key tests."""
    cfg = load_run_config("synthetic")
    model = build_model(cfg, 16)
    return {
        "model": model,
        "optimizer": None,
        "scheduler": None,
        "epoch": 0,
        "step": 10,
        "seed": 2001,
        "coalition": ["crop", "mask"],
        "policy": "game_a",
        "rng_states": {},
        "config_hash": "cfg",
        "data_hash": "data",
        "recipe_hash": "recipe",
        "metric_history": {},
    }


def test_atomic_save_and_validate(payload_dir):
    path = os.path.join(payload_dir, "ckpt.pt")
    payload = _dummy_payload()
    save_coalition_checkpoint(path, **payload)
    ok, reason = validate_checkpoint_file(path)
    assert ok, reason
    assert not any(f.startswith(".pt-tmp") for f in os.listdir(payload_dir))


def test_corrupted_checkpoint_rejected(payload_dir):
    path = os.path.join(payload_dir, "bad.pt")
    with open(path, "wb") as fh:
        fh.write(b"this is not a torch checkpoint")
    ok, reason = validate_checkpoint_file(path)
    assert not ok and "corrupted" in reason
    with pytest.raises(RuntimeError, match="rejected"):
        load_coalition_checkpoint(path)


def test_missing_keys_rejected(payload_dir):
    path = os.path.join(payload_dir, "partial.pt")
    payload = _dummy_payload()
    payload["model"] = payload["model"].state_dict()
    del payload["recipe_hash"]
    torch.save(payload, path)
    ok, reason = validate_checkpoint_file(path)
    assert not ok and "missing keys" in reason


def test_manifest_bookkeeping(payload_dir):
    manifest = CheckpointManifest(payload_dir)
    assert manifest.is_complete(dataset="synthetic", seed=2001, coalition=["crop"], policy="game_a") is False
    manifest.mark_coalition_complete(
        dataset="synthetic", seed=2001, coalition=["crop", "mask"], policy="game_a",
        step=24, checkpoint_path=os.path.join(payload_dir, "x.pt"),
        checkpoint_hash="h", config_hash="cfg", data_hash="data", recipe_hash="recipe",
    )
    assert manifest.is_complete(dataset="synthetic", seed=2001, coalition=["mask", "crop"], policy="game_a")
    entry = manifest.get(dataset="synthetic", seed=2001, coalition=["crop", "mask"], policy="game_a")
    assert entry["checkpoint_hash"] == "h"
    # reload from disk
    manifest2 = CheckpointManifest(payload_dir)
    assert manifest2.is_complete(dataset="synthetic", seed=2001, coalition=["crop", "mask"], policy="game_a")


def test_cache_key_validation_all_fields():
    payload = _dummy_payload()
    payload["dataset"] = "synthetic"
    payload["model"] = payload["model"].state_dict()
    kwargs = dict(dataset="synthetic", seed=2001, policy="game_a",
                  coalition=["crop", "mask"], config_hash="cfg",
                  data_hash="data", recipe_hash="recipe")
    ok, _ = validate_cache_keys(payload, **kwargs)
    assert ok
    for field, bad_value in [
        ("dataset", "beauty"), ("seed", 2002), ("policy", "game_b"),
        ("coalition", ["crop"]), ("config_hash", "other"), ("data_hash", "other"),
        ("recipe_hash", "other"),
    ]:
        bad = dict(kwargs)
        bad[field] = bad_value
        ok, reason = validate_cache_keys(payload, **bad)
        assert not ok, f"{field} mismatch not detected"
        assert f"{field} mismatch" in reason


def test_atomic_write_json(payload_dir):
    path = os.path.join(payload_dir, "m.json")
    atomic_write_json({"a": [1, 2], "b": {"c": 3}}, path)
    with open(path) as fh:
        assert "c" in fh.read()
