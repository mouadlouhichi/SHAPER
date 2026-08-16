"""Unit tests: recommendation baselines (GRU4Rec model + CL4SRec reference),
the exact-vs-MC audit, protect-last diagnostics, and the preregistration
archive."""

from __future__ import annotations

import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pytest
import torch

from shaper import MAIN_PLAYERS
from shaper.augment import augment_mask
from shaper.baseline_models import (
    BASELINE_REGISTRY,
    GRU4RecBaseline,
    build_gru4rec,
    cl4srec_reference,
)
from shaper.config import load_run_config
from shaper.monte_carlo import exact_vs_mc_audit, permutation_mc_estimate
from shaper.shapley import exact_shapley


# --------------------------------------------------------------------------
# GRU4Rec baseline
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def gru(tiny_cfg, tiny_data):
    return build_gru4rec(tiny_cfg, tiny_data.n_items)


def test_gru_forward_shapes(gru, tiny_data):
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    scores = gru.rank_scores(seq)
    assert scores.shape == (1, tiny_data.n_items)
    # evaluate_model compatibility: n_items derived from model.cfg
    assert gru.cfg.n_items == tiny_data.n_items


def test_gru_recommendation_loss_manual(gru):
    gru.eval()
    torch.manual_seed(0)
    seq = torch.tensor([[1, 2, 3]])
    negatives = torch.tensor([[7, 7]])
    loss, counts = gru.recommendation_loss(seq, negatives)
    h = gru.encode(seq)
    w = gru.item_emb.weight
    import torch.nn.functional as F

    expected = torch.zeros(())
    for t in range(2):
        s_pos = (h[0, t] * w[int(seq[0, t + 1])]).sum()
        s_neg = (h[0, t] * w[int(negatives[0, t])]).sum()
        expected = expected + (-F.logsigmoid(s_pos) - F.logsigmoid(-s_neg))
    assert torch.allclose(loss, expected / 2, atol=1e-5)
    assert counts[0].item() == 2


def test_gru_padding_safety(gru):
    gru.eval()
    seq = torch.tensor([[0, 0, 3, 7, 2], [5, 1, 4, 8, 9]])
    f = gru.final_valid_hidden(seq)
    assert f.shape == (2, gru.cfg.d)
    assert torch.isfinite(f).all()
    neg = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
    loss, counts = gru.recommendation_loss(seq, neg)
    assert torch.isfinite(loss)
    assert counts[0].item() == 3 and counts[1].item() == 4


def test_gru_trains_through_the_standard_loop(tmp_path, tiny_cfg, tiny_data):
    """The GRU4Rec baseline trains through train_coalition (rec-only) with
    the frozen recipe and produces a valid checkpoint."""
    from shaper.checkpoints import CheckpointManifest
    from shaper.logging_utils import StructuredLogger
    from shaper.training import Recipe, TrainContext, train_coalition

    ctx = TrainContext(
        cfg=tiny_cfg, data=tiny_data, recipe=Recipe(5e-4, 12, 0.1, 0.1),
        seed=2001, policy="gru4rec_baseline", run_dir=str(tmp_path),
        logger=StructuredLogger(str(tmp_path), "t"),
        manifest=CheckpointManifest(str(tmp_path)),
        config_hash=tiny_cfg.config_hash(), device="cpu",
        log_interval=6, checkpoint_interval=6, model_factory=build_gru4rec,
    )
    result = train_coalition(ctx, [])
    assert result.steps == 12
    payload = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert any(k.startswith("gru.") for k in payload["model"]), "GRU weights expected"
    assert "item_emb.weight" in payload["model"]


def test_baseline_registry_state():
    assert BASELINE_REGISTRY["gru4rec"]["status"] == "implemented"
    assert BASELINE_REGISTRY["cl4srec"]["status"].startswith("implemented")
    for pending in ("duorec", "coserec"):
        assert BASELINE_REGISTRY[pending]["status"] == "pending"


def test_cl4srec_reference_manifest(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "game_a_seed2001.json").write_text("[]")
    manifest = cl4srec_reference(str(tables), (2001,))
    assert manifest["baseline"] == "cl4srec"
    assert manifest["training_cost"] == "0 (reused coalition models)"
    assert manifest["reused_tables"] == [str(tables / "game_a_seed2001.json")]
    assert "protocol-compatible" in manifest["implementation"]


# --------------------------------------------------------------------------
# Exact-vs-MC audit
# --------------------------------------------------------------------------

def test_audit_matches_exact_with_increasing_samples():
    rng = np.random.default_rng(0)
    values = {tuple(sorted(c)): float(rng.uniform(-0.1, 0.1))
              for r in range(4) for c in itertools.combinations(MAIN_PLAYERS, r)}
    audit = exact_vs_mc_audit(values, MAIN_PLAYERS, n_samples=(4, 32), seed=0)
    exact = exact_shapley(values, MAIN_PLAYERS)
    for p in MAIN_PLAYERS:
        assert audit["exact"][p] == pytest.approx(exact[p])
    # every sampled path telescopes exactly (same table arithmetic)
    for n in (4, 32):
        est = audit["per_sample_size"][n]["estimate"]
        path_sum = sum(
            sum(permutation_mc_estimate(tuple(path), values, MAIN_PLAYERS).values())
            for path in audit["per_sample_size"][n]["paths"]
        ) / n
        assert path_sum == pytest.approx(values[tuple(MAIN_PLAYERS)] - values[()], abs=1e-12)
        assert abs(sum(est.values()) - (values[tuple(MAIN_PLAYERS)] - values[()])) < 1e-12
    assert set(audit["per_sample_size"].keys()) == {4, 32}


def test_audit_is_descriptive_not_an_estimator():
    """The audit samples from the exact table; it never replaces the exact
    allocation (label pinned)."""
    values = {tuple(sorted(c)): float(len(c)) for r in range(4) for c in itertools.combinations(MAIN_PLAYERS, r)}
    audit = exact_vs_mc_audit(values, MAIN_PLAYERS, n_samples=(8,), seed=1)
    assert "approximation audit" in audit["label"]
    assert audit["exact"] == exact_shapley(values, MAIN_PLAYERS)


# --------------------------------------------------------------------------
# Protect-last diagnostics
# --------------------------------------------------------------------------

def test_mask_protect_last_never_touches_tail():
    import random

    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    for draw in range(100):
        rng = random.Random(draw)
        out, changed = augment_mask(seq, rng, mask_id=99, gamma=0.5, protect_last=2)
        if changed:
            assert out[-1] != 99 and out[-2] != 99  # last two never masked
            assert any(x == 99 for x in out)         # others may be masked


def test_mask_protect_last_default_unchanged():
    import random

    # with protect_last=0 (main protocol), the last position CAN be masked
    masked_tail = 0
    for draw in range(200):
        rng = random.Random(draw)
        out, _ = augment_mask([1, 2, 3, 4, 5, 6, 7, 8], rng, mask_id=99, gamma=0.5)
        if out[-1] == 99 or out[-2] == 99:
            masked_tail += 1
    assert masked_tail > 0


# --------------------------------------------------------------------------
# Preregistration archive
# --------------------------------------------------------------------------

def test_preregistration_record_deterministic_parts():
    from scripts.archive_preregistration import generate_preregistration

    record = generate_preregistration("ml100k")
    assert record["kind"] == "staged preregistration archive"
    assert record["timestamp"]
    assert record["config_hash"] == load_run_config("ml100k").config_hash()
    # every frozen artifact is hashed
    for rel in ("specs/SHAPER_Implementation_Spec.md", "configs/seeds.yaml",
                "configs/statistics.yaml", "requirements.txt"):
        assert record["artifacts"][rel]["sha256"], rel
    assert len(record["archive_hash"]) == 32
    assert "BEFORE the two excluded pilot seeds" in record["note"]
