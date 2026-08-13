"""Contrastive objective and coalition loss assembly (spec A.6, A.7).

- Symmetric NT-Xent over the 2B representations of B valid (original,
  augmented) pairs: each representation uses its paired representation as
  the positive and the remaining 2B-2 as in-batch negatives. Batch
  normalized. Recommendation negatives never enter the contrastive
  denominator.
- No per-vector z-normalization beyond F.normalize(z, dim=-1, eps=1e-12)
  after the projection head.
- Coalition loss policies:
      Game A (primary, fixed nominal coefficient budget):
          L_cl^A(C) = |C|^{-1} * sum_{p in C} L_cl^p     (0 if C empty)
      Game B (secondary, fixed per-view dose):
          L_cl^B(C) = K^{-1} * sum_{p in C} L_cl^p
  with per-view term
      L_cl^p = (B_p / B) * mean(valid NT-Xent)  if B_p >= B_min(=8) else 0.
  No-op examples remain in the batch denominator; their nominal weight is
  NOT redistributed to other views.
- L = L_rec + lambda * L_cl(C).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

B_MIN = 8  # fixed before execution; near-degenerate NT-Xent guard


def symmetric_nt_xent(z_orig: torch.Tensor, z_aug: torch.Tensor, tau: float) -> torch.Tensor:
    """Symmetric NT-Xent for a batch of B valid (original, augmented) pairs.

    z_orig, z_aug: [B, d], already L2-normalized.
    """
    z = torch.cat([z_orig, z_aug], dim=0)  # [2B, d]
    sim = z @ z.t() / tau
    n = z.shape[0]
    sim = sim.masked_fill(torch.eye(n, dtype=torch.bool, device=z.device), float("-inf"))
    b = n // 2
    labels = torch.cat([torch.arange(b, n), torch.arange(0, b)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def per_view_cl_loss(
    z_orig: torch.Tensor,
    z_aug: torch.Tensor,
    changed: torch.Tensor,
    tau: float,
    min_pairs: int = B_MIN,
) -> Dict[str, Any]:
    """One view's batch-normalized contrastive term with no-op accounting.

    Returns {"loss": L_cl^p, "a_p": B_p/B, "insufficient_pairs": bool}.
    """
    b = changed.numel()
    b_p = int(changed.long().sum().item())
    a_p = b_p / b if b else 0.0
    if b_p < min_pairs:
        return {
            "loss": torch.zeros((), device=z_orig.device),
            "a_p": a_p,
            "insufficient_pairs": True,
            "n_pairs": b_p,
        }
    valid_mean = symmetric_nt_xent(z_orig[changed], z_aug[changed], tau)
    loss = changed.float().mean() * valid_mean
    return {"loss": loss, "a_p": a_p, "insufficient_pairs": False, "n_pairs": b_p}


def coalition_cl_loss(
    view_losses: Sequence[torch.Tensor],
    coalition: Sequence[str],
    policy: str,
    K: int = 3,
) -> torch.Tensor:
    """Combine per-view losses under the coalition budget policy.

    Game A denominator: |C|. Game B denominator: K. Empty coalition: zero.
    """
    if not coalition:
        return torch.zeros((), device=view_losses[0].device) if view_losses else torch.zeros(())
    summed = torch.stack(list(view_losses)).sum()
    if policy == "game_a":
        return summed / len(coalition)
    if policy == "game_b":
        return summed / K
    raise ValueError(f"unknown budget policy {policy}")


def weighted_cl_loss(
    view_losses: Sequence[torch.Tensor], weights: Sequence[float]
) -> torch.Tensor:
    """SHAPER-Weight objective term: sum_p w_p * L_cl^p — applied EXACTLY
    once, with nonnegative weights summing to one and NO additional 1/K or
    1/|C| denominator (spec A.10)."""
    if len(view_losses) != len(weights):
        raise ValueError("one weight per view required")
    if not view_losses:
        return torch.zeros(())
    total = sum(float(w) * loss for w, loss in zip(weights, view_losses))
    return total  # type: ignore[return-value]


def gate_entropy_regularized_cl_loss(
    view_losses: Sequence[torch.Tensor], weights: Sequence[torch.Tensor], entropy_coef: float
) -> torch.Tensor:
    """Learned dataset-level softmax-gate objective:
    sum_p w_p L_p - entropy_coef * H(w), H(w) = -sum w log w.
    Gates are training-only and are never used at inference."""
    w = torch.stack(list(weights))
    weighted = sum(w_i * loss for w_i, loss in zip(w, view_losses))
    if entropy_coef == 0.0:
        return weighted
    entropy = -(w * torch.log(w + 1e-12)).sum()
    return weighted - entropy_coef * entropy


def view_coefficient_weight(
    coalition: Sequence[str], policy: str, K: int = 3
) -> Dict[str, float]:
    """Nominal coefficient applied to each included view (before no-op mass)."""
    if policy == "game_a":
        denom = len(coalition) or 1
    elif policy == "game_b":
        denom = K
    else:
        raise ValueError(policy)
    return {p: 1.0 / denom for p in coalition}


def check_numerical_health(z: torch.Tensor, tau: float) -> Dict[str, Any]:
    """Log near-zero projection norms and NaN/Inf events (spec A.7)."""
    norm = z.norm(dim=-1)
    return {
        "nan_inf": bool(torch.isnan(z).any() or torch.isinf(z).any()),
        "min_norm": float(norm.min().item()),
        "near_zero_count": int((norm < 1e-6).sum().item()),
        "tau": tau,
    }


__all__ = [
    "B_MIN",
    "symmetric_nt_xent",
    "per_view_cl_loss",
    "coalition_cl_loss",
    "view_coefficient_weight",
    "check_numerical_health",
]
