"""Perturbation views (players): crop, mask, reorder (+ dropout procedure).

Reference implementations follow spec A.4 exactly. All views are pure input
transformations used only in the contrastive branch; recommendation loss is
always computed on the original sequence. The `rng` is Python `random.Random`
seeded from deterministic keyed schedules (shaper.schedules), so worker count
and coalition enumeration order cannot change a view draw.

No-op semantics: unchanged examples contribute ZERO view loss but remain in
the batch denominator. Game A denominator |C|, Game B denominator K.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

MASK_PAD = 0  # padding id; [MASK] token id = n_items + 1 (backbone convention)


def augment_crop(seq: Sequence[int], rng: random.Random, eta: Optional[float] = None) -> Tuple[List[int], bool]:
    """Contiguous subsequence; keep ratio eta ~ U[0.5, 1.0]; keep at least 2
    items and, for n >= 3, at most n-1 (forced non-identity where possible)."""
    seq = list(seq)
    n = len(seq)
    if n < 3:
        return seq, False
    eta = rng.uniform(0.5, 1.0) if eta is None else eta
    keep = min(n - 1, max(2, int(n * eta)))
    # rng is Python random.Random; randrange upper bound is exclusive.
    start = rng.randrange(n - keep + 1)
    out = seq[start : start + keep]
    return out, out != seq


def augment_mask(
    seq: Sequence[int], rng: random.Random, mask_id: int, gamma: float = 0.2,
    protect_last: int = 0,
) -> Tuple[List[int], bool]:
    """Replace gamma of valid positions by [MASK]. The main protocol does NOT
    protect the last two positions (protect_last=0).

    `protect_last>0` is used ONLY by the frozen rec-only diagnostics: masking
    candidates are restricted to positions[:-protect_last], so the last
    `protect_last` valid positions are never masked.
    """
    seq = list(seq)
    n = len(seq)
    if n == 0:
        return seq, False
    candidates = range(max(0, n - protect_last))
    if not candidates:
        return seq, False
    k = min(len(candidates), max(1, int(round(n * gamma))))
    idx = rng.sample(candidates, k)
    out = seq.copy()
    for i in idx:
        out[i] = mask_id
    return out, out != seq


def augment_reorder(
    seq: Sequence[int], rng: random.Random, beta: float = 0.2, max_attempts: int = 10
) -> Tuple[List[int], bool]:
    """Local non-identity permutation of a span of at least 2 (beta fraction);
    identity permutations are retried."""
    seq = list(seq)
    n = len(seq)
    if n < 2:
        return seq, False
    span = min(n, max(2, int(round(n * beta))))
    for _ in range(max_attempts):
        start = rng.randrange(n - span + 1)
        original = seq[start : start + span]
        if len(set(original)) < 2:
            continue
        permuted = original.copy()
        rng.shuffle(permuted)
        if permuted != original:
            out = seq[:start] + permuted + seq[start + span :]
            return out, True
    return seq, False


VIEW_FUNCTIONS = {"crop": augment_crop, "mask": augment_mask, "reorder": augment_reorder}


def apply_view(
    view: str,
    seq: Sequence[int],
    rng: random.Random,
    mask_id: Optional[int] = None,
    gamma: float = 0.2,
    beta: float = 0.2,
    eta: Optional[float] = None,
    protect_last: int = 0,
) -> Tuple[List[int], bool]:
    """Apply one named view to one sequence."""
    if view == "crop":
        return augment_crop(seq, rng, eta=eta)
    if view == "mask":
        if mask_id is None:
            raise ValueError("mask_id required for mask view")
        return augment_mask(seq, rng, mask_id=mask_id, gamma=gamma)
    if view == "reorder":
        return augment_reorder(seq, rng, beta=beta)
    raise ValueError(f"unknown view {view}")


# --------------------------------------------------------------------------
# Diagnostics (spec A.4 "Required diagnostics")
# --------------------------------------------------------------------------

def view_diagnostics(
    seqs: Sequence[Sequence[int]],
    n_draws: int = 5,
    seed: int = 0,
    mask_id: Optional[int] = None,
    gamma: float = 0.2,
    beta: float = 0.2,
) -> Dict[str, Any]:
    """Aggregate no-op/applicability diagnostics by view over a set of
    sequences using keyed canonical draws.

    Returns per view:
      - applicability_rate: fraction of draws that change the sequence
      - edit_fraction: mean fraction of positions changed
      - crop_suffix_deletion_rate / mask_final_position_rate / reorder_identity_retry_rate
    """
    diag: Dict[str, Any] = {"views": {}}
    for view in ("crop", "mask", "reorder"):
        changed = 0
        edit_fracs: List[float] = []
        special = 0
        total = 0
        for si, seq in enumerate(seqs):
            for draw in range(n_draws):
                from .provenance import key_int

                rng = random.Random(key_int(seed, si, draw, view, salt="view-diag"))
                out, is_changed = apply_view(view, seq, rng, mask_id=mask_id, gamma=gamma, beta=beta)
                total += 1
                if is_changed:
                    changed += 1
                    diffs = sum(1 for a, b in zip(out, seq) if a != b)
                    edit_fracs.append(diffs / max(len(seq), 1))
                if view == "crop" and is_changed:
                    # suffix deletion: the crop end is before the original end
                    if len(out) < len(seq) and list(seq)[-1] != out[-1]:
                        special += 1
                if view == "mask" and is_changed:
                    if len(out) and out[-1] == mask_id:
                        special += 1
                if view == "reorder" and not is_changed and len(seq) >= 2:
                    special += 1  # identity after retries
        diag["views"][view] = {
            "applicability_rate": changed / total if total else 0.0,
            "mean_edit_fraction": float(sum(edit_fracs) / len(edit_fracs)) if edit_fracs else 0.0,
            "special_event_rate": special / total if total else 0.0,
            "draws": total,
        }
    return diag


def effective_mass(a_p: Sequence[float], coalition: Sequence[str], policy: str, K: int = 3) -> float:
    """Applicability-adjusted effective coefficient mass (spec A.4).

    Game A: m_C^A = sum(a_p) / |C|   (0 for the empty coalition)
    Game B: m_C^B = sum(a_p) / K
    """
    a = list(a_p)
    if not coalition:
        return 0.0  # empty coalition: zero mass under every policy
    if policy == "game_a":
        return sum(a) / len(coalition)
    if policy == "game_b":
        return sum(a) / K
    raise ValueError(f"unknown budget policy {policy}")


# --------------------------------------------------------------------------
# Dropout player (Beauty K=4 extension)
# --------------------------------------------------------------------------

def contrastive_dropout_pair(
    model: "torch.nn.Module",
    batch_seqs: torch.Tensor,
    generators: Sequence[Optional[torch.Generator]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SimCSE-style dropout view: two additional keyed stochastic forwards of
    the backbone+projection form the positive pair. Ordinary backbone dropout
    stays active in every model; the player does not toggle it on/off.

    `generators` supplies one torch.Generator per forward, keyed by
    (seed, optimizer_step, purpose="dropout", view="dropout", pass_index), so
    worker count and coalition order cannot shift these masks.
    """
    zs = []
    for gen in generators:
        with torch.random.fork_rng(devices=[]):
            if gen is not None:
                torch.manual_seed(gen.initial_seed())
            zs.append(model.contrastive_forward(batch_seqs))
    return zs[0], zs[1]


__all__ = [
    "MASK_PAD",
    "augment_crop",
    "augment_mask",
    "augment_reorder",
    "apply_view",
    "VIEW_FUNCTIONS",
    "view_diagnostics",
    "effective_mass",
    "contrastive_dropout_pair",
]
