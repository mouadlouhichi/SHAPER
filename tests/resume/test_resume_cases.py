"""Resume-case tests (spec section 51): crashed seed, crashed mid-coalition,
crash between stages, partial K=4 (same permutation), corrupted checkpoint.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.checkpoints import CheckpointManifest, validate_checkpoint_file
from shaper.logging_utils import StructuredLogger
from shaper.training import Recipe, TrainContext, train_coalition


def _fresh_ctx(tmp_path, tiny_cfg, tiny_data, seed=2003, steps=24, interval=12):
    logger = StructuredLogger(str(tmp_path), run_id="resume")
    manifest = CheckpointManifest(str(tmp_path / "manifest"))
    return TrainContext(
        cfg=tiny_cfg, data=tiny_data,
        recipe=Recipe(5e-4, steps, 0.1, 0.1),
        seed=seed, policy="game_a", run_dir=str(tmp_path / "ckpt"),
        logger=logger, manifest=manifest, config_hash=tiny_cfg.config_hash(),
        device="cpu", log_interval=6, checkpoint_interval=interval,
        base_state_path=str(tmp_path / "ckpt" / f"seed{seed}" / "init.pt"),
    )


def test_case1_seed_crash_resume_continues_seed(tmp_path, tiny_cfg, tiny_data):
    """Game-A seed 2003 crashes partway; resume continues the same seed
    without retraining completed coalitions."""
    ctx = _fresh_ctx(tmp_path, tiny_cfg, tiny_data, seed=2003)
    done = []
    for coalition in (["crop"], ["mask"]):
        r = train_coalition(ctx, coalition)
        assert not r.cache_hit
        done.append(coalition)
    # crash -> resume: run the remaining coalitions + the completed ones
    hits = 0
    for coalition in (["crop"], ["mask"], ["reorder"], ["crop", "reorder"]):
        r = train_coalition(ctx, coalition)
        hits += int(r.cache_hit)
    assert hits == 2  # completed coalitions reused, never retrained


def test_case2_mid_coalition_crash_resume_matches_full_run(tmp_path, tiny_cfg, tiny_data):
    """Coalition crashes at step 12; resume restores the latest checkpoint and
    continues deterministically to the same final state as an uninterrupted
    run (identical recipe: 24 steps, snapshots every 12)."""
    full_ctx = _fresh_ctx(tmp_path / "full", tiny_cfg, tiny_data, steps=24, interval=12)
    full = train_coalition(full_ctx, ["crop", "mask"])

    # the interrupted run's step-12 snapshot is exactly what a crash leaves behind
    snapshot = os.path.join(
        full_ctx.run_dir, "seed2003", "game_a", "crop+mask", "step_12.pt"
    )
    assert os.path.exists(snapshot)
    resume_ctx = _fresh_ctx(tmp_path / "resume", tiny_cfg, tiny_data, steps=24, interval=12)
    resumed = train_coalition(resume_ctx, ["crop", "mask"], resume_from=snapshot)
    assert resumed.steps == 24
    assert len(resumed.metric_history["loss"]) == 24
    full_payload = torch.load(full.checkpoint_path, map_location="cpu", weights_only=False)
    resumed_payload = torch.load(resumed.checkpoint_path, map_location="cpu", weights_only=False)
    for key in full_payload["model"]:
        assert torch.equal(full_payload["model"][key], resumed_payload["model"][key]), key


def test_case3_game_b_never_retrains_game_a(tmp_path, tiny_cfg, tiny_data):
    """Game A completed -> Game B starts -> crash -> resume: Game A models
    are reused via cache and never retrained."""
    ctx = _fresh_ctx(tmp_path, tiny_cfg, tiny_data, seed=2001)
    for coalition in ([], ["crop", "mask", "reorder"], ["mask"]):
        train_coalition(ctx, coalition)  # policy game_a
    # Game B training starts (distinct policy entries), then "crashes"
    ctx_b = TrainContext(**{**ctx.__dict__, "policy": "game_b"})
    train_coalition(ctx_b, ["crop"])
    # resume: re-run both stages
    hits_a = [train_coalition(ctx, c).cache_hit for c in ([], ["crop", "mask", "reorder"], ["mask"])]
    assert all(hits_a)
    assert train_coalition(ctx_b, ["crop"]).cache_hit


def test_case4_k4_partial_resume_preserves_permutation(tmp_path):
    from shaper.monte_carlo import (
        K4SeedResult,
        required_coalitions,
        sample_permutation,
    )

    players = ("crop", "mask", "reorder", "dropout")
    pi = sample_permutation(3003, players)
    k4 = K4SeedResult(
        seed=3003, permutation=pi,
        reverse_permutation=tuple(reversed(pi)),
        coalition_registry=required_coalitions(pi, players),
    )
    k4.completed = ["empty", "+".join(sorted(pi[:2]))]
    path = str(tmp_path / "k4_seed3003.json")
    k4.save(path)
    # "crash" -> resume: load the frozen state and continue missing coalitions
    restored = K4SeedResult.load(path)
    assert restored.permutation == pi  # never resampled
    assert restored.reverse_permutation == tuple(reversed(pi))
    missing = [c for name, c in restored.coalition_registry.items() if name not in restored.completed]
    assert all(name in restored.coalition_registry for name in restored.completed)
    assert len(missing) + len(restored.completed) == len(restored.coalition_registry)
    # the same seed always maps to the same permutation (frozen run state)
    assert sample_permutation(3003, players) == pi


def test_case5_corrupted_checkpoint_rejected_and_previous_valid_restored(tmp_path, tiny_cfg, tiny_data):
    """A corrupted checkpoint is rejected; the manifest falls back to the
    previous valid snapshot."""
    ctx = _fresh_ctx(tmp_path, tiny_cfg, tiny_data, seed=2002, steps=24, interval=12)
    result = train_coalition(ctx, ["mask"])
    entry = ctx.manifest.get(dataset=tiny_cfg.dataset, seed=2002, coalition=["mask"], policy="game_a")
    # corrupt the final checkpoint
    with open(result.checkpoint_path, "wb") as fh:
        fh.write(b"corrupted bytes")
    ok, _ = validate_checkpoint_file(result.checkpoint_path)
    assert not ok
    fallback = ctx.manifest.find_previous_valid_checkpoint(
        dataset=tiny_cfg.dataset, seed=2002, coalition=["mask"], policy="game_a"
    )
    assert fallback is not None
    assert fallback != result.checkpoint_path
    ok2, _ = validate_checkpoint_file(fallback)
    assert ok2
    assert entry["checkpoint_hash"] != ""
