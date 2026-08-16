"""Cached ranking-adapter surrogate (spec A.6, appendix experiment).

The registered protocol sanctions ONE cheap-attribution appendix: a cached
ranking-adapter surrogate that

  - starts from the rec-only (empty-coalition) checkpoint,
  - freezes the backbone,
  - fits a ranking-path adapter per coalition,
  - trains with the RECOMMENDATION loss plus the coalition contrastive loss,
  - ALTERS item rankings (a contrastive-head-only surrogate is prohibited).

This module implements it and the mandatory validation report: per-coalition
values, exact Shapley of the surrogate table, and the error of the surrogate
attribution against the FULL-RETRAINING Game-A table (Spearman correlation
over coalition values, per-player Shapley MAE, rank-order agreement).

The surrogate is a FEASIBILITY VALIDATION LAYER promoted by the feasibility
amendment (configs/amendment.yaml); it NEVER replaces the primary full-
retraining Game-A table in confirmatory reporting.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RankingAdapterSurrogate(nn.Module):
    """Frozen rec-only backbone + per-coalition ranking-path adapter.

    Ranking scores: g = adapter(final_valid_hidden(seq)); score = g . item_emb.
    The adapter is IN the ranking path, so coalition training genuinely
    changes rankings. The contrastive branch reuses the base projection MLP
    (freshly initialized per coalition) on frozen backbone states.
    """

    def __init__(self, base_model: nn.Module, d: int):
        super().__init__()
        self.backbone = base_model.backbone
        self.projection = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)
        )
        self.adapter = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)
        )

    def freeze_backbone_params(self) -> None:
        """Called by train_coalition before the optimizer is built: only the
        adapter + projection (ranking/contrastive paths) remain trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False

    @property
    def n_items(self) -> int:
        return self.backbone.cfg.n_items

    def rank_scores(self, seqs: torch.Tensor) -> torch.Tensor:
        f = self.backbone.final_valid_hidden(seqs)
        g = self.adapter(f)
        w = self.backbone.item_emb.weight[1 : self.n_items + 1]
        return g @ w.t()

    def item_scores_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        g = self.adapter(hidden)
        w = self.backbone.item_emb.weight[1 : self.n_items + 1]
        return g @ w.t()

    def contrastive_forward(self, seqs: torch.Tensor) -> torch.Tensor:
        # the adapter sits in the CONTRASTIVE path too, so the coalition
        # contrastive loss updates the ranking adapter (the contrastive head
        # outside the ranking path is prohibited; here it is inside it)
        f = self.backbone.final_valid_hidden(seqs)
        g = self.adapter(f)
        return F.normalize(self.projection(g), dim=-1, eps=1e-12)

    def recommendation_loss(
        self, seqs: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BCE next-item loss with adapter-transformed scores (ranking path
        is trained by the recommendation objective)."""
        h = self.backbone.encode(seqs)          # [B, T, d] (frozen)
        B, T = seqs.shape
        g = self.adapter(h)                     # [B, T, d] adapter in path
        valid = (seqs != 0)[:, 1:]
        w = self.backbone.item_emb.weight
        pos_emb = w[seqs[:, 1:]]
        neg_emb = w[negatives.long()]
        s_pos = torch.einsum("btd,btd->bt", g[:, :-1], pos_emb)
        s_neg = torch.einsum("btd,btd->bt", g[:, :-1], neg_emb)
        bce = -F.logsigmoid(s_pos) - F.logsigmoid(-s_neg)
        bce = bce.masked_fill(~valid, 0.0)
        counts = valid.sum(dim=1)
        per_user = bce.sum(dim=1) / counts.clamp(min=1)
        keep = counts > 0
        loss = per_user[keep].mean() if bool(keep.any()) else torch.zeros((), device=seqs.device)
        return loss, counts


class ProjectionOnlySurrogate(nn.Module):
    """FORBIDDEN design (spec: 'a projection-only surrogate is invalid').
    Exists only so the guard below can detect and reject it."""

    def __init__(self, base_model: nn.Module, d: int):
        super().__init__()
        self.backbone = base_model.backbone
        self.projection = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
        self.adapter = None

    def rank_scores(self, seqs: torch.Tensor) -> torch.Tensor:
        f = self.backbone.final_valid_hidden(seqs)
        w = self.backbone.item_emb.weight[1 : self.backbone.cfg.n_items + 1]
        return f @ w.t()


def validate_surrogate_model(model: nn.Module) -> None:
    """Guard: the surrogate must alter the RANKING path. A contrastive head
    outside the ranking path (projection-only surrogate) is prohibited."""
    if getattr(model, "adapter", None) is None:
        raise ValueError(
            "projection-only surrogate is invalid (spec A.6): the surrogate "
            "must train a ranking-path adapter with the recommendation loss"
        )


def build_surrogate_factory(base_state: Dict[str, torch.Tensor], d: int):
    """model_factory for train_coalition: returns a RankingAdapterSurrogate
    whose backbone loads the frozen rec-only base state."""

    def factory(cfg: Any, n_items: int) -> nn.Module:
        from .backbone import build_model

        base_model = build_model(cfg, n_items)
        base_model.load_state_dict(base_state)
        surrogate = RankingAdapterSurrogate(base_model, d)
        surrogate.freeze_backbone_params()
        return surrogate

    return factory


def surrogate_validation_report(
    surrogate_values: Dict[Tuple[str, ...], float],
    full_values: Dict[Tuple[str, ...], float],
    players: Tuple[str, ...],
) -> Dict[str, Any]:
    """Compare the surrogate table against the full-retraining table:

    - Spearman rank correlation over the complete coalition values,
    - MAE over coalition values,
    - per-player Shapley MAE (exact allocation of both tables),
    - rank-order agreement of the two Shapley vectors.
    """
    from scipy.stats import spearmanr

    from .shapley import exact_shapley

    coalitions = sorted(full_values.keys(), key=lambda c: (len(c), tuple(c)))
    surr_vec = np.array([surrogate_values.get(c, float("nan")) for c in coalitions])
    full_vec = np.array([full_values[c] for c in coalitions])
    mask = ~np.isnan(surr_vec)
    rho, p_value = spearmanr(surr_vec[mask], full_vec[mask])
    value_mae = float(np.abs(surr_vec[mask] - full_vec[mask]).mean())

    phi_surr = exact_shapley(surrogate_values, players)
    phi_full = exact_shapley(full_values, players)
    phi_mae = {p: float(abs(phi_surr[p] - phi_full[p])) for p in players}
    order_surr = sorted(players, key=lambda p: phi_surr[p], reverse=True)
    order_full = sorted(players, key=lambda p: phi_full[p], reverse=True)

    return {
        "label": "cached ranking-adapter surrogate vs full-retraining Game A",
        "spearman_rho_over_values": float(rho),
        "spearman_p": float(p_value),
        "value_mae": value_mae,
        "phi_surrogate": phi_surr,
        "phi_full_retraining": phi_full,
        "phi_mae": phi_mae,
        "rank_order_surrogate": order_surr,
        "rank_order_full_retraining": order_full,
        "rank_order_agrees": order_surr == order_full,
        "note": (
            "the surrogate starts from the rec-only checkpoint, freezes the "
            "backbone, and trains a ranking-path adapter with BOTH the "
            "recommendation loss and the coalition contrastive loss (the "
            "adapter sits in both paths), so coalition membership genuinely "
            "changes rankings; it is reported with its error against full "
            "retraining and never replaces the primary Game-A table."
        ),
    }


__all__ = [
    "RankingAdapterSurrogate",
    "ProjectionOnlySurrogate",
    "validate_surrogate_model",
    "build_surrogate_factory",
    "surrogate_validation_report",
]
