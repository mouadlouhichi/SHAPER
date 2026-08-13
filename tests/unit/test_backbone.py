"""Unit tests: backbone architecture, causality, padding safety, gradient
flow, and recommendation loss (spec A.5)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.backbone import SASRecWithProjection, backbone_config_from, build_model
from shaper.config import load_run_config


@pytest.fixture(scope="module")
def model():
    cfg = load_run_config("synthetic")
    m = build_model(cfg, 40)
    return m


def test_architecture_locked_for_ml1m():
    cfg = load_run_config("ml1m")
    bc = backbone_config_from(cfg, 4000)
    assert bc.d == 64 and bc.heads == 2 and bc.ff == 256 and bc.dropout == 0.2
    assert bc.n_blocks == 2
    assert bc.mask_id == 4001
    assert bc.vocab_size == 4002
    model = SASRecWithProjection(bc)
    assert model.backbone.item_emb.padding_idx == 0
    assert isinstance(model.projection[0], torch.nn.Linear)
    assert isinstance(model.projection[1], torch.nn.ReLU)
    assert isinstance(model.projection[2], torch.nn.Linear)


def test_causal_masking(model):
    model.eval()  # deterministic forward (dropout off)
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    h = model.backbone.encode(seq)
    # a later token must not change earlier hidden states
    seq2 = seq.clone()
    seq2[0, -1] = 9
    h2 = model.backbone.encode(seq2)
    assert torch.allclose(h[0, :3], h2[0, :3], atol=1e-5)


def test_padding_safety_and_final_valid_hidden(model):
    model.eval()
    seq = torch.tensor([[0, 0, 3, 7, 2], [5, 1, 4, 8, 9]])
    f = model.backbone.final_valid_hidden(seq)
    assert f.shape == (2, model.backbone.cfg.d)
    h = model.backbone.encode(seq)
    assert torch.allclose(f[0], h[0, 4], atol=1e-5)
    assert torch.allclose(f[1], h[1, 4], atol=1e-5)
    assert torch.isfinite(h).all()


def test_rank_scores_use_backbone_not_projection():
    # a LOCAL model: perturbing the projection must never change ranking
    # scores (and must not pollute other tests' shared fixtures)
    cfg = load_run_config("synthetic")
    model = build_model(cfg, 40)
    model.eval()
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    scores_before = model.rank_scores(seq).clone()
    with torch.no_grad():
        for p in model.projection.parameters():
            p.add_(0.5)
    scores_after = model.rank_scores(seq)
    assert torch.allclose(scores_before, scores_after, atol=1e-6)
    assert scores_before.shape == (1, model.backbone.cfg.n_items)


def test_gradient_reaches_transformer_and_item_embeddings(model):
    """Contrastive backward pass produces nonzero gradients in at least one
    Transformer parameter AND the item-embedding table (spec A.9 test 8)."""
    model.train()
    seq = torch.tensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
    z = model.contrastive_forward(seq)
    loss = (z * z).sum()
    loss.backward()
    item_grad_norm = model.backbone.item_emb.weight.grad.norm().item()
    transformer_grads = [
        p.grad.norm().item()
        for block in model.backbone.blocks
        for p in block.parameters()
        if p.grad is not None
    ]
    assert item_grad_norm > 0
    assert any(g > 0 for g in transformer_grads)
    assert len(transformer_grads) > 0


def test_recommendation_loss_masks_padding_positions(model):
    model.eval()
    seq = torch.tensor([[0, 0, 3, 7, 2], [5, 1, 4, 8, 9]])
    negatives = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2]])
    loss, counts = model.recommendation_loss(seq, negatives)
    assert torch.isfinite(loss)
    # user 0 has 3 real tokens -> 3 valid next-item positions (all real
    # positions t>=1, including the first real item after padding)
    assert counts[0].item() == 3
    assert counts[1].item() == 4


def test_recommendation_loss_manual_bce(model):
    model.eval()  # deterministic forward (dropout off)
    torch.manual_seed(0)
    seq = torch.tensor([[1, 2, 3]])
    negatives = torch.tensor([[7, 7]])
    loss, counts = model.recommendation_loss(seq, negatives)
    h = model.backbone.encode(seq)
    w = model.backbone.item_emb.weight
    import torch.nn.functional as F

    # per-position BCE, averaged over valid positions
    expected = torch.zeros(())
    for t in range(2):
        s_pos = (h[0, t] * w[int(seq[0, t + 1])]).sum()
        s_neg = (h[0, t] * w[int(negatives[0, t])]).sum()
        expected = expected + (-F.logsigmoid(s_pos) - F.logsigmoid(-s_neg))
    expected = expected / 2
    assert torch.allclose(loss, expected, atol=1e-5)


def test_dropout_player_two_forwards(model):
    from shaper.augment import contrastive_dropout_pair
    from shaper.schedules import dropout_generator

    model.train()
    seq = torch.tensor([[1, 2, 3, 4, 5]])
    gens = [dropout_generator(1, 0, "dropout", "dropout", i) for i in range(2)]
    print("DEBUG seeds:", [g.initial_seed() for g in gens], "train:", model.training)
    from shaper.schedules import keyed_forward
    h1 = keyed_forward(model.backbone.final_valid_hidden, gens[0], seq)
    h2 = keyed_forward(model.backbone.final_valid_hidden, gens[1], seq)
    print("DEBUG h equal:", torch.allclose(h1, h2), "h1 sum:", h1.sum().item())
    z1, z2 = contrastive_dropout_pair(model, seq, gens)
    print("DEBUG equal:", torch.allclose(z1, z2))
    import torch.nn.functional as F
    print("DEBUG z1 vs proj(h1):", torch.allclose(z1, F.normalize(model.projection(h1), dim=-1, eps=1e-12)))
    print("DEBUG proj h1==h2:", torch.allclose(model.projection(h1), model.projection(h2)))
    print("DEBUG h1-h2 norm:", (h1-h2).norm().item())
    print("DEBUG p1 vs p2 after norm:", torch.allclose(F.normalize(model.projection(h1), dim=-1, eps=1e-12), F.normalize(model.projection(h2), dim=-1, eps=1e-12)))
    print("DEBUG grad attached:", h1.requires_grad, h2.requires_grad, z1.requires_grad)
    print("DEBUG dropout modules:", [m.training for m in model.backbone.modules() if isinstance(m, torch.nn.Dropout)])
    print("DEBUG projection training:", [m.training for m in model.projection.modules() if isinstance(m, torch.nn.Linear)])
    assert z1.shape == (1, 64)
    assert not torch.allclose(z1, z2)  # stochastic forwards differ under dropout
    # keyed determinism: same key -> same embeddings
    gens2 = [dropout_generator(1, 0, "dropout", "dropout", i) for i in range(2)]
    z1b, z2b = contrastive_dropout_pair(model, seq, gens2)
    assert torch.allclose(z1, z1b, atol=1e-6)
    assert torch.allclose(z2, z2b, atol=1e-6)
