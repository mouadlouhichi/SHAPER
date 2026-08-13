"""SASRec-style ranking backbone with contrastive projection head (spec A.5).

Architecture (locked):
  - item embedding: nn.Embedding(n_items + 2, d, padding_idx=0); [MASK] = n_items + 1
  - learnable positional embeddings; table covers max_len + 1 positions so the
    test prefix (training history + validation item) is representable
  - two causal Transformer blocks, d = 64, 2 heads, FFN = 256, dropout = 0.2
  - sequence representation: final VALID hidden state
  - contrastive projection MLP: Linear(64,64) -> ReLU -> Linear(64,64)
    (trained jointly in every nonempty coalition, discarded for ranking)

Recommendation:
  - BCE next-item training at all valid positions with one uniformly sampled
    unseen negative per positive (keyed negative schedule)
  - full-catalog dot-product ranking from the backbone representation and the
    item-embedding table; the projection MLP is never used for ranking.

Engineering note (documented): the positional table covers max_len + 1
positions. Training histories are truncated to max_len; evaluation prefixes
are truncated to max_len + 1 (train history + validation item for the test
role), per the locked evaluation protocol.

Attention masking (documented): queries attend causally; a REAL query never
attends to padding keys. Padding queries (whose outputs are never used) keep
a valid softmax to avoid NaN gradients propagating through values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RunConfig


@dataclass
class BackboneConfig:
    n_items: int
    d: int = 64
    heads: int = 2
    ff: int = 256
    dropout: float = 0.2
    n_blocks: int = 2
    max_len: int = 200

    @property
    def mask_id(self) -> int:
        return self.n_items + 1

    @property
    def vocab_size(self) -> int:
        return self.n_items + 2  # padding 0, items 1..n_items, MASK n_items+1


def backbone_config_from(cfg: RunConfig, n_items: int, max_len: Optional[int] = None) -> BackboneConfig:
    m = cfg.model
    return BackboneConfig(
        n_items=n_items,
        d=int(m["d"]),
        heads=int(m["heads"]),
        ff=int(m["ff"]),
        dropout=float(m["dropout"]),
        n_blocks=int(m["blocks"]),
        max_len=max_len or cfg.max_len,
    )


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        assert d % heads == 0
        self.heads = heads
        self.d_k = d // heads
        self.w_q = nn.Linear(d, d)
        self.w_k = nn.Linear(d, d)
        self.w_v = nn.Linear(d, d)
        self.w_o = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, additive_mask: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        q = self.w_q(x).view(B, T, self.heads, self.d_k).transpose(1, 2)  # [B,H,T,d_k]
        k = self.w_k(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.heads, self.d_k).transpose(1, 2)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,T,T]
        logits = logits + additive_mask
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.w_o(out)


class TransformerBlock(nn.Module):
    def __init__(self, d: int, heads: int, ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, heads, dropout)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SASRecBackbone(nn.Module):
    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.vocab_size, cfg.d, padding_idx=0)
        # max_len + 1 positions: training histories use <= max_len; the test
        # prefix (training history + validation item) uses max_len + 1.
        self.pos_emb = nn.Embedding(cfg.max_len + 1, cfg.d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.d, cfg.heads, cfg.ff, cfg.dropout) for _ in range(cfg.n_blocks)]
        )

    def attention_mask(self, seqs: torch.Tensor) -> torch.Tensor:
        """Additive [B,1,T,T] mask: causal, and real queries never attend to
        padding keys. Padding queries keep a valid softmax (their outputs are
        unused) to avoid NaN gradients."""
        B, T = seqs.shape
        device = seqs.device
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
        is_pad = seqs == 0  # [B, T]
        excluded = causal[None, :, :] | (is_pad[:, None, :] & ~is_pad[:, :, None])
        mask = torch.zeros(B, 1, T, T, device=device)
        mask.masked_fill_(excluded.unsqueeze(1), float("-inf"))
        return mask

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """[B, T, d] hidden states for every position."""
        B, T = seqs.shape
        x = self.item_emb(seqs) + self.pos_emb(torch.arange(T, device=seqs.device))[None]
        mask = self.attention_mask(seqs)
        for block in self.blocks:
            x = block(x, mask)
        return x

    def final_valid_hidden(self, seqs: torch.Tensor) -> torch.Tensor:
        """Final VALID hidden state per sequence.

        Sequences are LEFT-padded, so the most recent item is always the last
        position; the final valid hidden state is that position (position 0
        for an all-padding sequence, which never occurs in evaluation).
        """
        h = self.encode(seqs)
        B, T = seqs.shape
        lens = (seqs != 0).sum(dim=1)
        idx = torch.where(lens > 0, torch.full((B,), T - 1, dtype=torch.long, device=seqs.device),
                          torch.zeros(B, dtype=torch.long, device=seqs.device))
        return h[torch.arange(B, device=seqs.device), idx]

    def rank_scores(self, seqs: torch.Tensor) -> torch.Tensor:
        """Full-catalog dot-product scores [B, n_items] from the backbone
        representation and the item-embedding table (projection discarded)."""
        f = self.final_valid_hidden(seqs)
        w = self.item_emb.weight[1 : self.cfg.n_items + 1]
        return f @ w.t()

    def item_scores_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        w = self.item_emb.weight[1 : self.cfg.n_items + 1]
        return hidden @ w.t()


class SASRecWithProjection(nn.Module):
    """Backbone + contrastive projection MLP used for coalition training."""

    def __init__(self, cfg: BackboneConfig):
        super().__init__()
        self.backbone = SASRecBackbone(cfg)
        self.projection = nn.Sequential(
            nn.Linear(cfg.d, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    # -- contrastive path ---------------------------------------------------
    def contrastive_forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """L2-normalized projection of the final valid hidden state."""
        f = self.backbone.final_valid_hidden(seqs)
        z = self.projection(f)
        return F.normalize(z, dim=-1, eps=1e-12)

    # -- ranking path (projection discarded) --------------------------------
    def rank_scores(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.backbone.rank_scores(seqs)

    # -- recommendation loss -------------------------------------------------
    def recommendation_loss(
        self, seqs: torch.Tensor, negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """BCE next-item loss at all valid positions.

        For each valid position t (seq[t] is a real item), score the positive
        item seq[t] and the keyed negative against the hidden state at t-1.
        Averages over valid positions per user, then over users.

        Returns (loss, per_user_valid_position_counts).
        """
        h = self.backbone.encode(seqs)  # [B, T, d]
        B, T = seqs.shape
        valid = (seqs != 0)[:, 1:]  # positions 1..T-1 that hold real items
        pos_items = seqs[:, 1:]
        w = self.backbone.item_emb.weight

        pos_emb = w[pos_items]  # [B, T-1, d]
        neg_emb = w[negatives.long()]  # [B, T-1, d]
        prefix_h = h[:, :-1]  # hidden state at t-1

        s_pos = torch.einsum("btd,btd->bt", prefix_h, pos_emb)
        s_neg = torch.einsum("btd,btd->bt", prefix_h, neg_emb)

        bce = -F.logsigmoid(s_pos) - F.logsigmoid(-s_neg)  # [B, T-1]
        bce = bce.masked_fill(~valid, 0.0)
        counts = valid.sum(dim=1)  # [B]
        per_user = bce.sum(dim=1) / counts.clamp(min=1)
        keep = counts > 0
        if bool(keep.any()):
            loss = per_user[keep].mean()
        else:
            loss = torch.zeros((), device=seqs.device)
        return loss, counts


def build_model(cfg: RunConfig, n_items: int, max_len: Optional[int] = None) -> SASRecWithProjection:
    return SASRecWithProjection(backbone_config_from(cfg, n_items, max_len=max_len))


__all__ = [
    "BackboneConfig",
    "SASRecBackbone",
    "SASRecWithProjection",
    "backbone_config_from",
    "build_model",
]
