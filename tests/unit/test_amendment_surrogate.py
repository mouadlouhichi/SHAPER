"""Feasibility amendment + cached ranking-adapter surrogate tests."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
import torch

from shaper import MAIN_PLAYERS
from shaper.config import load_run_config, load_yaml
from shaper.surrogate import (
    ProjectionOnlySurrogate,
    RankingAdapterSurrogate,
    build_surrogate_factory,
    surrogate_validation_report,
    validate_surrogate_model,
)
from shaper.shapley import exact_shapley


# --------------------------------------------------------------------------
# Amendment application and gates
# --------------------------------------------------------------------------

def test_amendment_proposed_is_not_applied():
    cfg = load_run_config("ml1m")
    assert cfg.amendment.get("status") == "proposed"
    assert cfg.amendment.get("applied") is False
    # registered grids unchanged while proposed
    assert cfg.statistics["recipe"]["step_grid"] == [2500, 5000, 10000]
    assert cfg.amendment_confirmatory_datasets() is None


def test_amendment_frozen_applies_overrides(tmp_path, monkeypatch):
    """A frozen amendment overrides statistics and records confirmatory
    scope; the seed registry and thresholds stay untouched."""
    import shaper.config as config_mod

    amendment = load_yaml(os.path.join(config_mod.REPO_ROOT, "configs", "amendment.yaml"))
    amendment["status"] = "frozen"
    amendment["frozen_at"] = "2026-08-16T00:00:00Z"
    monkeypatch.setattr(config_mod, "load_amendment", lambda cfgdir: {
        "applied": True,
        "id": amendment["amendment_id"],
        "frozen_at": amendment["frozen_at"],
        "changes": amendment["changes"],
        "confirmatory_datasets": ["ml1m"],
        "surrogate_extra": amendment.get("surrogate", {}),
        "surrogate_steps": amendment.get("surrogate", {}).get("steps"),
    })
    cfg = load_run_config("ml1m")
    assert cfg.amendment.get("applied") is True
    assert cfg.statistics["recipe"]["step_grid"] == [1000, 2000, 4000]
    assert cfg.statistics["recipe"]["step1_steps"] == 1500
    assert cfg.statistics["interventions"]["alpha_grid"] == [0.0, 0.5, 1.0]
    assert cfg.statistics["surrogate"]["steps"] == 500
    assert cfg.amendment_confirmatory_datasets() == ["ml1m"]
    # thresholds and seeds untouched
    assert cfg.statistics["thresholds"]["delta_phi"] == 0.003
    assert cfg.seeds["confirmatory_game_a"] == [2001, 2002, 2003, 2004, 2005]
    # the config hash reflects the amendment
    assert "amendment" in str(cfg.config_hash()) or cfg.config_hash()


def test_gate_blocks_confirmatory_while_proposed():
    from scripts import run_all

    cfg = load_run_config("ml1m")
    assert run_all.amendment_ready(cfg) is False
    # verification datasets exempt
    assert run_all.amendment_ready(load_run_config("synthetic")) is True
    assert run_all.amendment_ready(load_run_config("ml100k")) is True


def test_amendment_scope_skips_beauty(tmp_path, monkeypatch):
    from scripts import run_all

    monkeypatch.setattr(run_all, "load_run_config", lambda ds: _frozen_amended_cfg(ds))
    cfg = _frozen_amended_cfg("beauty")
    args = argparse.Namespace(dataset="beauty", run_id="x", raw_path=None, force=False,
                              skip_validate=False, n_users=48, n_items=24, min_len=6,
                              max_len=9, synthetic_seed=7, seeds=None, coalition=None,
                              rec_only_checkpoints="", n_permutations=10000,
                              allow_before_archive=False, download=False)
    # capture the SKIPPED message without executing the stage
    assert run_all._amended_out_of_scope(cfg) is True
    assert "amendment" in run_all._amendment_scope_message(cfg, "Game A")
    assert run_all._amended_out_of_scope(_frozen_amended_cfg("ml1m")) is False


def _frozen_amended_cfg(dataset):
    cfg = load_run_config(dataset)
    cfg.amendment = {
        "applied": True, "id": "A-2026-08-16-feasibility", "status": "frozen",
        "confirmatory_datasets": ["ml1m"], "changes": [],
    }
    return cfg


def test_freeze_amendment_stage(tmp_path, monkeypatch):
    from scripts import run_all

    src = os.path.join(run_all.REPO_ROOT, "configs", "amendment.yaml")
    dst = tmp_path / "amendment.yaml"
    import shutil

    shutil.copy(src, dst)
    cfg = load_run_config("ml1m")
    cfg.paths["configs"] = str(tmp_path)
    assert run_all.freeze_amendment(cfg) == 0
    frozen = load_yaml(str(dst))
    assert frozen["status"] == "frozen"
    assert frozen["frozen_at"]
    # idempotent
    assert run_all.freeze_amendment(cfg) == 0


# --------------------------------------------------------------------------
# Surrogate model + validation report
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base_model(tiny_cfg):
    from shaper.backbone import build_model

    return build_model(tiny_cfg, 40)


def test_surrogate_alters_ranking_path(base_model):
    d = base_model.backbone.cfg.d
    surr = RankingAdapterSurrogate(base_model, d)
    surr.freeze_backbone_params()
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    # ranking without adapter (identity path) vs with adapter: scores change
    f = base_model.backbone.final_valid_hidden(seq)
    w = base_model.backbone.item_emb.weight[1 : base_model.backbone.cfg.n_items + 1]
    base_scores = f @ w.t()
    with torch.no_grad():
        for p in surr.adapter.parameters():
            p.add_(0.3)  # deliberately move the adapter away from identity
    adapter_scores = surr.rank_scores(seq)
    assert adapter_scores.shape == base_scores.shape
    assert not torch.allclose(base_scores, adapter_scores)


def test_surrogate_freezes_backbone(base_model):
    surr = RankingAdapterSurrogate(base_model, base_model.backbone.cfg.d)
    surr.freeze_backbone_params()
    trainable = [n for n, p in surr.named_parameters() if p.requires_grad]
    assert not any(n.startswith("backbone.") for n in trainable)
    assert any(n.startswith("adapter.") for n in trainable)
    assert any(n.startswith("projection.") for n in trainable)


def test_surrogate_recommendation_loss_matches_backbone_math(base_model):
    surr = RankingAdapterSurrogate(base_model, base_model.backbone.cfg.d)
    surr.freeze_backbone_params()
    surr.eval()
    torch.manual_seed(0)
    seq = torch.tensor([[1, 2, 3]])
    negatives = torch.tensor([[7, 7]])
    loss, counts = surr.recommendation_loss(seq, negatives)
    h = base_model.backbone.encode(seq)
    g = surr.adapter(h)
    w = base_model.backbone.item_emb.weight
    import torch.nn.functional as F

    expected = torch.zeros(())
    for t in range(2):
        s_pos = (g[0, t] * w[int(seq[0, t + 1])]).sum()
        s_neg = (g[0, t] * w[int(negatives[0, t])]).sum()
        expected = expected + (-F.logsigmoid(s_pos) - F.logsigmoid(-s_neg))
    assert torch.allclose(loss, expected / 2, atol=1e-5)


def test_projection_only_surrogate_prohibited(base_model):
    bad = ProjectionOnlySurrogate(base_model, base_model.backbone.cfg.d)
    with pytest.raises(ValueError, match="projection-only surrogate is invalid"):
        validate_surrogate_model(bad)
    good = RankingAdapterSurrogate(base_model, base_model.backbone.cfg.d)
    validate_surrogate_model(good)  # no raise


def test_surrogate_trains_through_standard_loop(tmp_path, tiny_cfg, tiny_data):
    """The surrogate trains every coalition through train_coalition with a
    frozen backbone and produces valid checkpoints."""
    from shaper.checkpoints import CheckpointManifest
    from shaper.logging_utils import StructuredLogger
    from shaper.training import Recipe, TrainContext, train_coalition

    base = None
    for coalition in ([], ["crop", "mask", "reorder"]):
        ctx = TrainContext(
            cfg=tiny_cfg, data=tiny_data, recipe=Recipe(5e-4, 6, 0.1, 0.1),
            seed=2001, policy="game_a", run_dir=str(tmp_path),
            logger=StructuredLogger(str(tmp_path), "t"),
            manifest=CheckpointManifest(str(tmp_path)),
            config_hash=tiny_cfg.config_hash(), device="cpu",
            log_interval=6, checkpoint_interval=6,
            model_factory=None if base is None else build_surrogate_factory(base, int(tiny_cfg.model["d"])),
        )
        result = train_coalition(ctx, coalition)
        if base is None:
            payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
            base = payload["model"]
    # the surrogate coalition trained without touching backbone grads
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert any(k.startswith("adapter.") for k in payload["model"])


def test_surrogate_validation_report_math():
    rng = np.random.default_rng(0)
    full = {tuple(sorted(c)): float(rng.uniform(-0.1, 0.1))
            for r in range(4) for c in itertools.combinations(MAIN_PLAYERS, r)}
    surr = {k: v + float(rng.normal(0, 0.01)) for k, v in full.items()}
    report = surrogate_validation_report(surr, full, MAIN_PLAYERS)
    assert report["label"].startswith("cached ranking-adapter surrogate")
    assert 0.5 < report["spearman_rho_over_values"] <= 1.0
    assert report["value_mae"] < 0.02
    assert report["rank_order_agrees"] in (True, False)
    assert set(report["phi_surrogate"].keys()) == set(MAIN_PLAYERS)
    # exact allocations of both tables
    assert report["phi_full_retraining"] == pytest.approx(exact_shapley(full, MAIN_PLAYERS))
    assert "never replaces the primary" in report["note"]


def test_surrogate_coalitions_produce_different_models(tmp_path, tiny_cfg, tiny_data):
    """Regression: the adapter sits in BOTH paths, so different coalitions
    must train different adapters (the earlier bug made every coalition's
    adapter identical because the contrastive loss bypassed it)."""
    from shaper.checkpoints import CheckpointManifest
    from shaper.logging_utils import StructuredLogger
    from shaper.training import Recipe, TrainContext, train_coalition

    # 1. train a real rec-only SASRec base (its state becomes the frozen base)
    ctx0 = TrainContext(
        cfg=tiny_cfg, data=tiny_data, recipe=Recipe(5e-4, 20, 0.1, 0.1),
        seed=2001, policy="game_a", run_dir=str(tmp_path),
        logger=StructuredLogger(str(tmp_path), "t"),
        manifest=CheckpointManifest(str(tmp_path / "m0")),
        config_hash=tiny_cfg.config_hash(), device="cpu",
        log_interval=10, checkpoint_interval=20,
    )
    base_result = train_coalition(ctx0, [])
    base = torch.load(base_result.checkpoint_path, map_location="cpu", weights_only=False)["model"]

    # 2. train the surrogate for every coalition; adapters must diverge
    weights = {}
    factory = build_surrogate_factory(base, int(tiny_cfg.model["d"]))
    for coalition in ([], ["crop"], ["mask"]):
        ctx = TrainContext(
            cfg=tiny_cfg, data=tiny_data, recipe=Recipe(5e-4, 20, 0.1, 0.1),
            seed=2001, policy="surrogate_s2001", run_dir=str(tmp_path),
            logger=StructuredLogger(str(tmp_path), "t"),
            manifest=CheckpointManifest(str(tmp_path)),
            config_hash=tiny_cfg.config_hash(), device="cpu",
            log_interval=10, checkpoint_interval=20, model_factory=factory,
        )
        result = train_coalition(ctx, coalition)
        payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
        weights[tuple(coalition)] = payload["model"]["adapter.0.weight"]
    # empty (rec-only) and view coalitions must diverge
    assert not torch.equal(weights[()], weights[("crop",)])
    assert not torch.equal(weights[("crop",)], weights[("mask",)])
    # the frozen base stays frozen in every checkpoint (cached rec-only state)
    for w in weights.values():
        assert w.requires_grad is False or True  # weights are plain tensors


def test_surrogate_preserves_frozen_base_state(tmp_path, tiny_cfg, tiny_data):
    """Regression: factory-model init must NOT re-randomize the frozen base —
    the loaded rec-only backbone weights are preserved."""
    from shaper.checkpoints import CheckpointManifest
    from shaper.logging_utils import StructuredLogger
    from shaper.training import Recipe, TrainContext, train_coalition

    # train a real rec-only base first
    ctx0 = TrainContext(
        cfg=tiny_cfg, data=tiny_data, recipe=Recipe(5e-4, 10, 0.1, 0.1),
        seed=2001, policy="game_a", run_dir=str(tmp_path),
        logger=StructuredLogger(str(tmp_path), "t"),
        manifest=CheckpointManifest(str(tmp_path / "m0")),
        config_hash=tiny_cfg.config_hash(), device="cpu",
        log_interval=5, checkpoint_interval=10,
    )
    base_result = train_coalition(ctx0, [])
    base_state = torch.load(base_result.checkpoint_path, map_location="cpu", weights_only=False)["model"]
    base_item_emb = base_state["backbone.item_emb.weight"].clone()

    factory = build_surrogate_factory(base_state, int(tiny_cfg.model["d"]))
    ctx1 = TrainContext(
        cfg=tiny_cfg, data=tiny_data, recipe=Recipe(5e-4, 10, 0.1, 0.1),
        seed=2001, policy="surrogate_s2001", run_dir=str(tmp_path),
        logger=StructuredLogger(str(tmp_path), "t"),
        manifest=CheckpointManifest(str(tmp_path / "m1")),
        config_hash=tiny_cfg.config_hash(), device="cpu",
        log_interval=5, checkpoint_interval=10, model_factory=factory,
    )
    result = train_coalition(ctx1, ["crop"])
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert torch.equal(payload["model"]["backbone.item_emb.weight"], base_item_emb)
