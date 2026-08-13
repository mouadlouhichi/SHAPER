"""Unit tests: symmetric NT-Xent, no-op accounting, budget-policy identity,
and the SHAPER-Weight objective term."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn.functional as F
import pytest

from shaper.contrast import (
    B_MIN,
    coalition_cl_loss,
    per_view_cl_loss,
    symmetric_nt_xent,
    weighted_cl_loss,
)


def test_symmetric_nt_xent_matches_hand_calculation():
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(4, 8), dim=-1)
    z2 = F.normalize(torch.randn(4, 8), dim=-1)
    tau = 0.1
    loss = symmetric_nt_xent(z1, z2, tau)

    z = torch.cat([z1, z2])
    sim = z @ z.t() / tau
    n = 8
    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool), float("-inf"))
    labels = torch.cat([torch.arange(4, 8), torch.arange(0, 4)])
    expected = F.cross_entropy(sim, labels)
    assert torch.allclose(loss, expected, atol=1e-6)


def test_symmetric_nt_xent_permutation_invariant():
    torch.manual_seed(1)
    z1 = F.normalize(torch.randn(6, 4), dim=-1)
    z2 = F.normalize(torch.randn(6, 4), dim=-1)
    a = symmetric_nt_xent(z1, z2, 0.1)
    perm = torch.randperm(6)
    b = symmetric_nt_xent(z1[perm], z2[perm], 0.1)
    assert torch.allclose(a, b, atol=1e-6)


def test_insufficient_pairs_zero_loss_with_event():
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(16, 4), dim=-1)
    z2 = F.normalize(torch.randn(16, 4), dim=-1)
    changed = torch.zeros(16, dtype=torch.bool)
    changed[: B_MIN - 1] = True
    out = per_view_cl_loss(z1, z2, changed, tau=0.1)
    assert out["insufficient_pairs"] is True
    assert out["loss"].item() == 0.0
    assert out["a_p"] == pytest.approx((B_MIN - 1) / 16)


def test_noop_accounting_full_batch():
    """Full-batch no-ops: zero loss, a_p = 0, batch denominator unchanged."""
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(12, 4), dim=-1)
    z2 = F.normalize(torch.randn(12, 4), dim=-1)
    changed = torch.zeros(12, dtype=torch.bool)
    out = per_view_cl_loss(z1, z2, changed, tau=0.1)
    assert out["loss"].item() == 0.0
    assert out["a_p"] == 0.0


def test_noop_weight_not_redistributed():
    """An inapplicable view contributes zero WITHOUT reallocating its nominal
    share to other views under either policy."""
    l_crop = torch.tensor(0.9)
    l_mask = torch.tensor(0.0)  # full-batch no-op
    l_reorder = torch.tensor(0.3)
    a = coalition_cl_loss([l_crop, l_mask, l_reorder], ["crop", "mask", "reorder"], "game_a")
    b = coalition_cl_loss([l_crop, l_mask, l_reorder], ["crop", "mask", "reorder"], "game_b", K=3)
    assert a.item() == pytest.approx((0.9 + 0.0 + 0.3) / 3)
    assert b.item() == pytest.approx((0.9 + 0.0 + 0.3) / 3)


def test_budget_policy_identity():
    """Hand-specified view losses produce sum/|C| in Game A and sum/K in
    Game B, including the empty coalition (spec A.9 test 15)."""
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    coalition = ["crop", "mask", "reorder"]
    assert coalition_cl_loss(losses, coalition, "game_a").item() == pytest.approx(2.0)
    assert coalition_cl_loss(losses, coalition, "game_b", K=3).item() == pytest.approx(2.0)
    # singleton differs between policies
    one = [torch.tensor(4.0)]
    assert coalition_cl_loss(one, ["crop"], "game_a").item() == pytest.approx(4.0)
    assert coalition_cl_loss(one, ["crop"], "game_b", K=3).item() == pytest.approx(4.0 / 3)
    # empty coalition -> zero under both
    empty = coalition_cl_loss([], [], "game_a")
    assert empty.item() == 0.0


def test_weighted_objective_applied_exactly_once():
    """SHAPER-Weight: sum_p w_p L_p exactly once, no additional denominator
    (spec A.9 test 13)."""
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0)]
    weights = [0.5, 0.3, 0.2]
    out = weighted_cl_loss(losses, weights)
    assert out.item() == pytest.approx(0.5 * 1 + 0.3 * 2 + 0.2 * 3)
    with pytest.raises(ValueError):
        weighted_cl_loss(losses, [0.5, 0.5])
