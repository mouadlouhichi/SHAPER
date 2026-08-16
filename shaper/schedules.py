"""Deterministic keyed random schedules (spec A.4/A.5).

All randomness in the study is counter-keyed:

    augmentation key:  (seed, optimizer_step, global_example_id, occurrence, view)
    recommendation negative key: (seed, optimizer_step, global_user_id, target_position)
    epoch key:         (seed, epoch, dataset_hash)
    dropout key:       (seed, optimizer_step, purpose, view, pass_index)

Changing coalition enumeration order, worker count, or process schedule must
not change any of these schedules. Each draw is a pure function of its key,
implemented with Python `random.Random` (augmentations) or a keyed
`torch.Generator` (dropout forwards).
"""

from __future__ import annotations

import random
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import torch

from .provenance import key_int

SALT = "shaper-schedules-v1"


# --------------------------------------------------------------------------
# Augmentation schedules
# --------------------------------------------------------------------------

def augmentation_rng(
    seed: int,
    optimizer_step: int,
    global_example_id: int,
    occurrence: int,
    view: str,
) -> random.Random:
    """A view receives the same random draw whenever it occurs in different
    coalitions (identical (seed, step, example, occurrence, view) key)."""
    key = key_int(
        seed, optimizer_step, global_example_id, occurrence, view, salt=f"{SALT}:aug"
    )
    return random.Random(key)


def view_draws(
    seed: int,
    optimizer_step: int,
    global_example_ids: Sequence[int],
    view: str,
    occurrence: int = 0,
) -> List[random.Random]:
    """One keyed RNG per example for `view` at `optimizer_step`."""
    return [
        augmentation_rng(seed, optimizer_step, int(gid), occurrence, view)
        for gid in global_example_ids
    ]


# --------------------------------------------------------------------------
# Recommendation negative schedule
# --------------------------------------------------------------------------

def recommendation_negatives(
    seed: int,
    optimizer_step: int,
    global_user_ids: Sequence[int],
    target_positions: Sequence[int],
    n_items: int,
    exclude_sets: Sequence[Iterable[int]],
    positives: Sequence[int],
) -> List[int]:
    """One uniformly sampled negative per (user, position), keyed by
    (seed, optimizer_step, global_user_id, target_position).

    The negative is drawn uniformly from retained items that are absent from
    the user's TRAINING history and differ from the positive item, and is
    hash-identical across coalitions for the same key.
    """
    out: List[int] = []
    for uid, pos, exclude, pos_item in zip(
        global_user_ids, target_positions, exclude_sets, positives
    ):
        rng = random.Random(
            key_int(seed, optimizer_step, int(uid), int(pos), salt=f"{SALT}:neg")
        )
        excluded = set(int(e) for e in exclude) | {int(pos_item)}
        out.append(int(rng.choice([i for i in range(1, n_items + 1) if i not in excluded])))
    return out


def recommendation_negative(
    seed: int, optimizer_step: int, global_user_id: int, target_position: int,
    n_items: int, exclude_set: Iterable[int], positive: int,
) -> int:
    """Single keyed negative (used in tests and non-batched paths)."""
    return recommendation_negatives(
        seed,
        optimizer_step,
        [global_user_id],
        [target_position],
        n_items,
        [exclude_set],
        [positive],
    )[0]


# --------------------------------------------------------------------------
# Epoch schedule
# --------------------------------------------------------------------------

def epoch_key(seed: int, epoch: int, dataset_hash: str) -> int:
    return key_int(seed, epoch, dataset_hash, salt=f"{SALT}:epoch")


def epoch_permutation(seed: int, epoch: int, dataset_hash: str, n_users: int) -> List[int]:
    """New keyed permutation per (seed, epoch, dataset_hash); every eligible
    user appears exactly once."""
    rng = random.Random(epoch_key(seed, epoch, dataset_hash))
    perm = list(range(n_users))
    rng.shuffle(perm)
    return perm


# --------------------------------------------------------------------------
# Dropout schedule
# --------------------------------------------------------------------------

def dropout_generator(
    seed: int, optimizer_step: int, purpose: str, view: str, pass_index: int
) -> torch.Generator:
    """Keyed torch generator for a stochastic forward."""
    key = key_int(seed, optimizer_step, purpose, view, pass_index, salt=f"{SALT}:dropout")
    # torch.Generator accepts int64 seeds; the 128-bit key is reduced mod 2^63-1
    # (documented; collision probability is negligible)
    key64 = key % (2**63 - 1)
    gen = torch.Generator()
    gen.manual_seed(key64)
    return gen


def keyed_forward(fn: "callable", generator: torch.Generator, *args: Any, **kwargs: Any) -> Any:
    """Run `fn` under a forked RNG seeded by the keyed generator so the
    stochastic forward is a pure function of its key."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(generator.initial_seed())
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------
# Determinism setup
# --------------------------------------------------------------------------

def set_deterministic_rng(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - GPU environments
        torch.cuda.manual_seed_all(seed)


def configure_determinism() -> dict:
    """Enable deterministic kernels per spec A.2.

    Returns {deterministic_mode: bool, exception: str|null, detail}.
    If an operation prevents deterministic kernels the exception is recorded
    and NEVER silently suppressed; data/RNG schedules remain deterministic.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True)
        return {"deterministic_mode": True, "exception": None, "detail": "all kernels deterministic"}
    except Exception as exc:  # noqa: BLE001 - must be recorded, not swallowed
        return {
            "deterministic_mode": False,
            "exception": f"{type(exc).__name__}: {exc}",
            "detail": "deterministic data/RNG schedules retained; kernel execution marked nondeterministic",
        }


__all__ = [
    "SALT",
    "augmentation_rng",
    "view_draws",
    "recommendation_negatives",
    "recommendation_negative",
    "epoch_key",
    "epoch_permutation",
    "dropout_generator",
    "keyed_forward",
    "set_deterministic_rng",
    "configure_determinism",
]
