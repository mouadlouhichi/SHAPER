"""Recommendation baselines (paper 4.2, "Recommendation baselines").

The paper requires the baseline implementations, versions, tuning budgets,
deterministic flags and hardware to be archived BEFORE confirmatory
training. This module provides:

  - GRU4RecBaseline: GRU-based sequential recommender (rec-only; no
    contrastive branch) trained with the SAME frozen recipe (optimizer,
    warmup/decay, fixed step budget, keyed negatives, deterministic epoch
    permutations) as the SHAPER coalitions, via the `model_factory` hook of
    TrainContext. Declared validation budget: V_tune/V_game/V_select roles
    as usual; final evaluation on test only after every decision is locked.
  - cl4srec_reference(): CL4SRec (crop/mask/reorder contrastive views,
    uniformly weighted) is exactly the Game-A GRAND coalition under this
    study's protocol; the helper reuses the already-trained grand-coalition
    models and records them under the CL4SRec label (zero new training).
    The original implementation's exact hyperparameters differ; the paper
    reports ours as a protocol-compatible re-implementation with the frozen
    recipe (this caveat is recorded in the baseline manifest).

DuoRec/CoSeRec are tracked as remaining pre-confirmatory baseline work
(they require semantic-augmentation machinery and a literature-freeze
decision); the registry in this module makes that gap explicit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GRUConfig:
    n_items: int
    d: int = 64
    layers: int = 1
    dropout: float = 0.2
    max_len: int = 200

    @property
    def mask_id(self) -> int:
        return self.n_items + 1

    @property
    def vocab_size(self) -> int:
        return self.n_items + 2


class GRU4RecBaseline(nn.Module):
    """GRU4Rec-style sequential recommender with the SHAPER training/eval
    interface (recommendation_loss + rank_scores). Rec-only: it has no
    contrastive branch and is trained with the empty coalition."""

    def __init__(self, cfg: GRUConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.vocab_size, cfg.d, padding_idx=0)
        self.gru = nn.GRU(
            input_size=cfg.d,
            hidden_size=cfg.d,
            num_layers=cfg.layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        # NOTE: no self.backbone alias — assigning a module to itself
        # recurses infinitely; evaluate_model derives n_items from
        # model.backbone.cfg (SASRec) or model.cfg (baselines).

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """GRU hidden states for every position (input = item embeddings)."""
        x = self.item_emb(seqs)
        x = self.dropout(x)
        h, _ = self.gru(x)
        return h

    def final_valid_hidden(self, seqs: torch.Tensor) -> torch.Tensor:
        h = self.encode(seqs)
        B, T = seqs.shape
        lens = (seqs != 0).sum(dim=1)
        idx = torch.where(
            lens > 0,
            torch.full((B,), T - 1, dtype=torch.long, device=seqs.device),
            torch.zeros(B, dtype=torch.long, device=seqs.device),
        )
        return h[torch.arange(B, device=seqs.device), idx]

    def rank_scores(self, seqs: torch.Tensor) -> torch.Tensor:
        """Full-catalog dot-product ranking from the final valid state."""
        f = self.final_valid_hidden(seqs)
        w = self.item_emb.weight[1 : self.cfg.n_items + 1]
        return f @ w.t()

    def recommendation_loss(
        self, seqs: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BCE next-item loss at all valid positions (same averaging scheme
        as the SASRec backbone: per-user mean, then batch mean)."""
        h = self.encode(seqs)  # [B, T, d]; h[:, t] consumes seq[:, t]
        B, T = seqs.shape
        valid = (seqs != 0)[:, 1:]
        w = self.item_emb.weight
        pos_emb = w[seqs[:, 1:]]
        neg_emb = w[negatives.long()]
        prefix_h = h[:, :-1]  # state BEFORE the target item
        s_pos = torch.einsum("btd,btd->bt", prefix_h, pos_emb)
        s_neg = torch.einsum("btd,btd->bt", prefix_h, neg_emb)
        bce = -F.logsigmoid(s_pos) - F.logsigmoid(-s_neg)
        bce = bce.masked_fill(~valid, 0.0)
        counts = valid.sum(dim=1)
        per_user = bce.sum(dim=1) / counts.clamp(min=1)
        keep = counts > 0
        loss = per_user[keep].mean() if bool(keep.any()) else torch.zeros((), device=seqs.device)
        return loss, counts


def build_gru4rec(cfg: Any, n_items: int, **kw: Any) -> GRU4RecBaseline:
    """model_factory compatible constructor for the GRU4Rec baseline."""
    model_cfg = cfg.model if isinstance(cfg.model, dict) else vars(cfg.model)
    return GRU4RecBaseline(
        GRUConfig(
            n_items=n_items,
            d=int(kw.get("d", model_cfg.get("d", 64))),
            layers=int(kw.get("layers", 1)),
            dropout=float(model_cfg.get("dropout", 0.2)),
            max_len=cfg.max_len,
        )
    )


def cl4srec_reference(coalition_tables_dir: str, seeds: Tuple[int, ...]) -> Dict[str, Any]:
    """CL4SRec reference under this study's protocol = the Game-A grand
    coalition (uniform crop/mask/reorder contrastive views on the frozen
    recipe). Reuses the enumerated models; records the protocol-compatibility
    caveat. Returns a manifest referencing the reused table files."""
    reused = []
    for seed in seeds:
        path = os.path.join(coalition_tables_dir, f"game_a_seed{seed}.json")
        if os.path.exists(path):
            reused.append(path)
    return {
        "baseline": "cl4srec",
        "implementation": "protocol-compatible re-implementation (Game-A grand coalition reuse)",
        "note": (
            "CL4SRec uses crop/mask/reorder contrastive views under uniform weights, "
            "which equals this study's Game-A grand coalition objective; the original "
            "implementation's hyperparameters differ and ours follow the frozen recipe."
        ),
        "training_cost": "0 (reused coalition models)",
        "reused_tables": reused,
        "seeds": list(seeds),
    }


BASELINE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "gru4rec": {
        "status": "implemented",
        "implementation": "shaper/baseline_models.py::GRU4RecBaseline",
        "training": "rec-only, frozen recipe, fixed step budget",
        "validation_budget": "V_tune/V_game/V_select roles (same as coalitions)",
    },
    "cl4srec": {
        "status": "implemented (grand-coalition reuse)",
        "implementation": "shaper/baseline_models.py::cl4srec_reference",
        "training": "reuses Game-A grand coalition",
    },
    "duorec": {
        "status": "pending",
        "note": "semantic (supervised) augmentation machinery + literature-freeze decision required",
    },
    "coserec": {
        "status": "pending",
        "note": "substitution-aware view machinery + literature-freeze decision required",
    },
}


__all__ = [
    "GRUConfig",
    "GRU4RecBaseline",
    "build_gru4rec",
    "cl4srec_reference",
    "BASELINE_REGISTRY",
]
