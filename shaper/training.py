"""Coalition training protocol (spec A.5, A.6) plus the weighted/gated
objective machinery used ONLY by the RQ4 interventions (spec A.10).

Within each seed:
  - one common initial `state_dict` is cloned into every coalition model
  - independent optimizers (never shared, never warm-started from another
    coalition — in particular never from the grand coalition)
  - shared sequence/batch order (keyed epoch permutations)
  - shared recommendation-negative schedule (keyed by seed/step/user/position)
  - keyed augmentation draws for each view (identical across coalitions)
  - every stochastic forward uses a keyed dropout RNG so adding a view cannot
    shift recommendation or incumbent-view dropout masks

Fixed budget: exactly `train_steps` optimizer updates; NO per-coalition early
stopping. AdamW(betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4), global
gradient-norm clipping at 1.0, linear warm-up over the first 10% of locked
steps, cosine decay to 0.1 * learning_rate.

Objective modes:
  coalition (game_a | game_b): L = L_rec + lambda * L_cl(C)
  weighted  (SHAPER-Weight, LOO weights, direct search, Dirichlet controls):
            L = L_rec + lambda * sum_p w_p L_cl^p   (no extra denominator)
  gated     (learned dataset-level softmax gates, training-only):
            L = L_rec + lambda * (sum_p softmax_p L_p - c_ent * H(softmax))

Every step logs the per-view applicability a_p = B_p / B and the
applicability-adjusted effective coefficient mass m_C (Game A: sum(a_p)/|C|;
Game B: sum(a_p)/K). Effective coefficient mass is never reported as total
gradient magnitude.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .augment import contrastive_dropout_pair, effective_mass
from .backbone import SASRecWithProjection, build_model
from .checkpoints import (
    CheckpointManifest,
    load_coalition_checkpoint,
    save_coalition_checkpoint,
    validate_cache_keys,
)
from .contrast import (
    coalition_cl_loss,
    gate_entropy_regularized_cl_loss,
    per_view_cl_loss,
    weighted_cl_loss,
)
from .data import FrozenData
from .logging_utils import StructuredLogger
from .schedules import (
    augmentation_rng,
    dropout_generator,
    keyed_forward,
    recommendation_negatives,
    set_deterministic_rng,
)


@dataclass
class Recipe:
    """Frozen training recipe (selected once on V_tune)."""

    learning_rate: float
    steps: int
    lambda_cl: float
    tau: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "steps": self.steps,
            "lambda_cl": self.lambda_cl,
            "tau": self.tau,
        }

    def hash(self) -> str:
        from .provenance import recipe_hash

        return recipe_hash(self.to_dict())


@dataclass
class TrainContext:
    cfg: Any
    data: FrozenData
    recipe: Recipe
    seed: int
    policy: str
    run_dir: str
    logger: StructuredLogger
    manifest: CheckpointManifest
    config_hash: str
    device: str = "cpu"
    log_interval: int = 100
    checkpoint_interval: Optional[int] = None
    base_state: Optional[Dict[str, torch.Tensor]] = None
    base_state_path: Optional[str] = None
    model_factory: Optional[Any] = None  # (cfg, n_items) -> nn.Module; default SASRec

    @property
    def data_hash(self) -> str:
        return self.data.data_hash

    @property
    def recipe_hash(self) -> str:
        return self.recipe.hash()


@dataclass
class CoalitionTrainResult:
    coalition: List[str]
    policy: str
    seed: int
    dataset: str
    model: Optional[SASRecWithProjection]
    steps: int
    epoch: int
    checkpoint_path: str
    checkpoint_hash: str
    config_hash: str
    data_hash: str
    recipe_hash: str
    metric_history: Dict[str, List[float]] = field(default_factory=dict)
    effective_mass_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    applicability_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    final_grad_norm: float = 0.0
    cache_hit: bool = False
    wall_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "coalition": self.coalition,
            "policy": self.policy,
            "seed": self.seed,
            "dataset": self.dataset,
            "steps": self.steps,
            "epoch": self.epoch,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self.checkpoint_hash,
            "config_hash": self.config_hash,
            "data_hash": self.data_hash,
            "recipe_hash": self.recipe_hash,
            "effective_mass_stats": self.effective_mass_stats,
            "applicability_stats": self.applicability_stats,
            "final_grad_norm": self.final_grad_norm,
            "cache_hit": self.cache_hit,
            "wall_seconds": self.wall_seconds,
        }
        for key, values in self.metric_history.items():
            d[f"metric_history_{key}"] = {
                "mean": float(np.mean(values)) if values else None,
                "last": float(values[-1]) if values else None,
            }
        return d


def make_optimizer(model: nn.Module, cfg: Any, lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=tuple(cfg.training["betas"]),
        eps=float(cfg.training["eps"]),
        weight_decay=float(cfg.training["weight_decay"]),
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    train_steps: int,
    warmup_frac: float = 0.1,
    cosine_final_frac: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up (0 -> lr) over the first 10% of locked steps, cosine
    decay to 0.1 * learning_rate afterwards."""
    warmup = max(1, int(train_steps * warmup_frac))
    total = max(train_steps, warmup + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (total - warmup)
        return cosine_final_frac + 0.5 * (1.0 - cosine_final_frac) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def common_initialization(
    ctx: TrainContext, model: Optional[SASRecWithProjection] = None
) -> Dict[str, torch.Tensor]:
    """One saved initial model state per seed, cloned into every coalition."""
    if ctx.base_state is not None:
        return {k: v.clone() for k, v in ctx.base_state.items()}
    n_items = ctx.data.n_items
    model = model or build_model(ctx.cfg, n_items)
    set_deterministic_rng(ctx.seed)
    model.apply(_init_weights)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ctx.base_state_path:
        os.makedirs(os.path.dirname(ctx.base_state_path), exist_ok=True)
        torch.save({"state_dict": state, "seed": ctx.seed}, ctx.base_state_path)
    ctx.base_state = state
    return {k: v.clone() for k, v in state.items()}


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class DatasetLevelGates(nn.Module):
    """Dataset-level three-logit softmax gates shared by all users/batches.

    Initialized at zero (uniform). Never used at inference. Trained with a
    separate gate learning rate (1e-2, locked)."""

    def __init__(self, n_views: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_views))

    def weights(self) -> torch.Tensor:
        import torch.nn.functional as F

        return F.softmax(self.logits, dim=-1)


def _view_batches(
    ctx: TrainContext, seed: int, epoch: int
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Keyed epoch permutation over ALL training users (every user once)."""
    for user_ids, seqs in ctx.data.train_batches(seed, epoch):
        yield user_ids, seqs


def _negative_matrix(ctx: TrainContext, step: int, user_ids: torch.Tensor, seqs: torch.Tensor) -> torch.Tensor:
    """One keyed unseen negative per (user, target_position), drawn uniformly
    from retained items absent from the user's frozen training history and
    different from the positive item. Hash-identical across coalitions."""
    B, T = seqs.shape
    negatives = torch.zeros((B, T - 1), dtype=torch.int64)
    valid = (seqs != 0)[:, 1:]
    for b in range(B):
        uid = int(user_ids[b])
        exclude = set(int(x) for x in seqs[b].tolist() if x != 0)
        for t in range(T - 1):
            if not bool(valid[b, t]):
                continue
            pos = int(seqs[b, t + 1])
            negatives[b, t] = recommendation_negatives(
                ctx.seed, step, [uid], [t + 1], ctx.data.n_items, [exclude], [pos]
            )[0]
    return negatives


def _apply_views(
    ctx: TrainContext, step: int, user_ids: torch.Tensor, seqs: torch.Tensor, views: Sequence[str]
) -> Dict[str, Dict[str, Any]]:
    """Keyed augmentation draws for every included view."""
    out: Dict[str, Dict[str, Any]] = {}
    mask_id = ctx.data.n_items + 1
    for view in views:
        if view == "dropout":
            continue  # model-level procedure, handled separately
        aug_seqs = []
        changed = torch.zeros(seqs.shape[0], dtype=torch.bool)
        for i in range(seqs.shape[0]):
            rng = augmentation_rng(ctx.seed, step, int(user_ids[i]), 0, view)
            seq_list = [int(x) for x in seqs[i].tolist() if x != 0]
            aug, is_changed = _single_view(ctx, view, seq_list, rng, mask_id)
            padded = [0] * (seqs.shape[1] - len(aug)) + aug
            aug_seqs.append(padded)
            changed[i] = bool(is_changed)
        out[view] = {
            "aug_seqs": torch.tensor(aug_seqs, dtype=torch.int64, device=seqs.device),
            "changed": changed.to(seqs.device),
        }
    return out


def _single_view(ctx: TrainContext, view: str, seq: List[int], rng: Any, mask_id: int):
    from .augment import apply_view

    aug_cfg = ctx.cfg.augmentation
    if view == "crop":
        return apply_view("crop", seq, rng)
    if view == "mask":
        return apply_view("mask", seq, rng, mask_id=mask_id, gamma=aug_cfg["mask"]["gamma"])
    if view == "reorder":
        return apply_view("reorder", seq, rng, beta=aug_cfg["reorder"]["beta"])
    raise ValueError(view)


def _train_one_step(
    ctx: TrainContext,
    model: SASRecWithProjection,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    user_ids: torch.Tensor,
    seqs: torch.Tensor,
    views: Sequence[str],
    weights: Optional[Dict[str, float]] = None,
    gate: Optional[DatasetLevelGates] = None,
    gate_optimizer: Optional[torch.optim.Optimizer] = None,
    entropy_coef: float = 0.0,
) -> Dict[str, Any]:
    """One optimizer step under one of the three objective modes."""
    device = next(model.parameters()).device
    aug_cfg = ctx.cfg.augmentation
    min_pairs = int(aug_cfg["min_contrastive_pairs"])

    negatives = _negative_matrix(ctx, step, user_ids, seqs).to(device)
    rec_gen = dropout_generator(ctx.seed, step, "rec", "rec", 0)
    rec_loss, counts = keyed_forward(model.recommendation_loss, rec_gen, seqs, negatives)

    view_losses: List[torch.Tensor] = []
    a_p: List[float] = []
    log_records: List[Dict[str, Any]] = []
    for view in views:
        if view == "dropout":
            gen0 = dropout_generator(ctx.seed, step, "dropout", "dropout", 0)
            gen1 = dropout_generator(ctx.seed, step, "dropout", "dropout", 1)
            z1, z2 = contrastive_dropout_pair(model, seqs, [gen0, gen1])
            changed = torch.ones(seqs.shape[0], dtype=torch.bool, device=device)
            out = per_view_cl_loss(z1, z2, changed, ctx.recipe.tau, min_pairs)
            a_p.append(1.0)  # model-level player applies to every example
        else:
            draws = _apply_views(ctx, step, user_ids, seqs, [view])[view]
            aug_seqs = draws["aug_seqs"]
            changed = draws["changed"]
            anchor_gen = dropout_generator(ctx.seed, step, "contrast", view, 0)
            aug_gen = dropout_generator(ctx.seed, step, "contrast", view, 1)
            z_orig = keyed_forward(model.contrastive_forward, anchor_gen, seqs)
            z_aug = keyed_forward(model.contrastive_forward, aug_gen, aug_seqs)
            out = per_view_cl_loss(z_orig, z_aug, changed, ctx.recipe.tau, min_pairs)
            a_p.append(out["a_p"])
        view_losses.append(out["loss"])
        log_records.append(
            {"view": view, "a_p": out["a_p"], "n_pairs": out["n_pairs"],
             "insufficient_pairs": out["insufficient_pairs"]}
        )

    if gate is not None:
        w = gate.weights()
        cl_loss = gate_entropy_regularized_cl_loss(view_losses, list(w), entropy_coef)
        w_dict = {v: float(wi.detach()) for v, wi in zip(views, w)}
    elif weights is not None:
        w_list = [float(weights[v]) for v in views]
        cl_loss = weighted_cl_loss(view_losses, w_list)
        w_dict = dict(weights)
    else:
        cl_loss = coalition_cl_loss(view_losses, list(views), ctx.policy, K=ctx.cfg.n_players)
        w_dict = {}

    loss = rec_loss + ctx.recipe.lambda_cl * cl_loss

    optimizer.zero_grad(set_to_none=True)
    if gate_optimizer is not None:
        gate_optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm_pre = float(_total_grad_norm(model))
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(ctx.cfg.training["grad_clip"]))
    grad_norm_post = float(_total_grad_norm(model))
    optimizer.step()
    scheduler.step()
    if gate_optimizer is not None:
        gate_optimizer.step()

    if gate is not None or weights is not None:
        mass = float(sum(float(w_dict.get(v, 0.0)) * ap for v, ap in zip(views, a_p))) if a_p else 0.0
    else:
        mass = effective_mass(a_p if a_p else [0.0], list(views), ctx.policy, K=ctx.cfg.n_players)
    return {
        "loss": float(loss.detach().item()),
        "rec_loss": float(rec_loss.detach().item()),
        "cl_loss": float(cl_loss.detach().item()),
        "a_p": a_p,
        "m_c": mass,
        "grad_norm_pre": grad_norm_pre,
        "grad_norm_post": grad_norm_post,
        "views": log_records,
        "gate_weights": w_dict,
    }


def _total_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum().item())
    return math.sqrt(total)


def coalition_dir(ctx: TrainContext, coalition: Sequence[str]) -> str:
    cname = "+".join(sorted(coalition)) if coalition else "empty"
    d = coalition_dir_for(ctx, coalition, policy=ctx.policy)
    os.makedirs(d, exist_ok=True)
    return d


def _mass_summary(
    metric_history: Dict[str, List[float]],
    a_p_history: Dict[str, List[float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    def _summ(values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0.0, "sd": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "n": 0}
        arr = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "sd": float(arr.std()),
            "p10": float(np.percentile(arr, 10)),
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "n": len(arr),
        }

    eff = {
        "m_c": _summ(metric_history.get("m_c", [])),
        "note": "effective coefficient mass — never called total gradient magnitude",
    }
    app = {view: _summ(vals) for view, vals in a_p_history.items()}
    return eff, app


def _result_payload(
    ctx: TrainContext, coalition: List[str], policy: str, model: SASRecWithProjection,
    step: int, epoch: int, final_path: str, final_hash: str,
    metric_history: Dict[str, List[float]], a_p_history: Dict[str, List[float]],
    cache_hit: bool, wall_seconds: float,
) -> CoalitionTrainResult:
    eff, app = _mass_summary(metric_history, a_p_history)
    return CoalitionTrainResult(
        coalition=coalition,
        policy=policy,
        seed=ctx.seed,
        dataset=ctx.cfg.dataset,
        model=model,
        steps=step,
        epoch=epoch,
        checkpoint_path=final_path,
        checkpoint_hash=final_hash,
        config_hash=ctx.config_hash,
        data_hash=ctx.data_hash,
        recipe_hash=ctx.recipe_hash,
        metric_history=metric_history,
        effective_mass_stats=eff,
        applicability_stats=app,
        final_grad_norm=metric_history["grad_norm_pre"][-1] if metric_history["grad_norm_pre"] else 0.0,
        cache_hit=cache_hit,
        wall_seconds=wall_seconds,
    )


def train_coalition(
    ctx: TrainContext,
    coalition: Sequence[str],
    resume_from: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    gate: Optional[DatasetLevelGates] = None,
    entropy_coef: float = 0.0,
    label: Optional[str] = None,
) -> CoalitionTrainResult:
    """Train one coalition (or weighted/gated variant) from the common
    seed-specific initialization for exactly `train_steps` optimizer updates.

    - Completed coalitions with matching cache keys are reused, never retrained.
    - `resume_from` continues an interrupted training (restoring model,
      optimizer, scheduler, step and RNG states).
    - weights / gate switch the objective to the RQ4 modes; the coalition
      label then embeds a hash of the weight vector so caches never collide.
    """
    t0 = time.time()
    coalition = sorted(coalition)
    if weights is not None:
        from .provenance import stable_hash

        h8 = stable_hash({k: round(v, 9) for k, v in weights.items()}, salt="w")[:8]
        coalition = [f"weighted-{h8}"]
        policy = "weighted"
        views = [v for v, w in sorted(weights.items(), key=lambda kv: kv[0])]
    elif gate is not None:
        coalition = [f"gated-ent{entropy_coef}"]
        policy = "gated"
        views = list(coalition_views(ctx))
    else:
        policy = ctx.policy
        views = list(coalition)
    cname = "+".join(coalition) if coalition else "empty"
    cdir = coalition_dir_for(ctx, coalition, policy=policy)
    final_path = os.path.join(cdir, "final.pt")
    latest_path = os.path.join(cdir, "latest.pt")

    # Cache: reuse a completed coalition when every key matches.
    cached = ctx.manifest.get(
        dataset=ctx.cfg.dataset, seed=ctx.seed, coalition=coalition, policy=policy
    )
    if cached is not None and os.path.exists(cached["checkpoint_path"]):
        payload = load_coalition_checkpoint(cached["checkpoint_path"])
        ok, reason = validate_cache_keys(
            payload,
            dataset=ctx.cfg.dataset,
            seed=ctx.seed,
            policy=policy,
            coalition=coalition,
            config_hash=ctx.config_hash,
            data_hash=ctx.data_hash,
            recipe_hash=ctx.recipe_hash,
        )
        if ok and payload.get("step", -1) >= ctx.recipe.steps:
            ctx.logger.info(
                "PRIMARY_GAME_A", "cache_hit", seed=ctx.seed, coalition=coalition,
                policy=policy, dataset=ctx.cfg.dataset, checkpoint=cached["checkpoint_path"],
            )
            factory = ctx.model_factory or build_model
            model = factory(ctx.cfg, ctx.data.n_items)
            model.load_state_dict(payload["model"])
            model.to(ctx.device)
            return _result_payload(
                ctx, coalition, policy, model, payload["step"], payload["epoch"],
                cached["checkpoint_path"], cached["checkpoint_hash"],
                payload.get("metric_history", {}),
                payload.get("extra", {}).get("a_p_history", {}),
                cache_hit=True, wall_seconds=0.0,
            )
        ctx.logger.warning(
            "PRIMARY_GAME_A", "stale_cache_rejected", seed=ctx.seed,
            coalition=coalition, policy=policy, reason=reason,
        )

    factory = ctx.model_factory or build_model
    model = factory(ctx.cfg, ctx.data.n_items)
    if ctx.model_factory is None:
        # registered protocol: every coalition clones the common seed init
        init_state = common_initialization(ctx)
        model.load_state_dict({k: v.clone() for k, v in init_state.items()})
    else:
        # baseline models have their own seed-specific initialization
        # (same RNG discipline: seeded by the seed, xavier embeddings/linear)
        set_deterministic_rng(ctx.seed)
        model.apply(_init_weights)
    model.to(ctx.device)
    optimizer = make_optimizer(model, ctx.cfg, ctx.recipe.learning_rate)
    scheduler = make_scheduler(optimizer, ctx.recipe.steps)

    gate_optimizer = None
    if gate is not None:
        gate.to(ctx.device)
        gate_lr = float(ctx.cfg.statistics["interventions"]["gate_learning_rate"])
        gate_optimizer = torch.optim.AdamW(gate.parameters(), lr=gate_lr, weight_decay=0.0)

    start_step = 0
    epoch = 0
    skip_batches = 0  # batch position within the current epoch (resume fidelity)
    metric_history: Dict[str, List[float]] = {
        "loss": [], "rec_loss": [], "cl_loss": [], "m_c": [], "grad_norm_pre": [], "grad_norm_post": []
    }
    a_p_history: Dict[str, List[float]] = {}
    if resume_from is not None:
        payload = load_coalition_checkpoint(resume_from)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_step = int(payload["step"])
        epoch = int(payload["epoch"])
        skip_batches = int(payload.get("extra", {}).get("batches_consumed_this_epoch", 0))
        metric_history = payload.get("metric_history", metric_history)
        a_p_history = payload.get("extra", {}).get("a_p_history", {})
        if gate is not None and payload.get("extra", {}).get("gate_state") is not None:
            gate.load_state_dict(payload["extra"]["gate_state"])
        ctx.logger.info(
            "PRIMARY_GAME_A", "resumed", seed=ctx.seed, coalition=coalition,
            policy=policy, from_step=start_step, checkpoint=resume_from,
        )

    ctx.logger.info(
        "PRIMARY_GAME_A", "coalition_training_start", seed=ctx.seed,
        coalition=coalition, policy=policy, dataset=ctx.cfg.dataset, steps=ctx.recipe.steps,
    )

    iterator = _view_batches(ctx, ctx.seed, epoch)
    interval = ctx.checkpoint_interval or int(ctx.cfg.training["checkpoint_interval"])
    model.train()
    consumed_this_epoch = 0
    # resume fidelity: re-consume the already-trained batches of the current
    # epoch BEFORE the first training step of the resumed run
    while skip_batches > 0:
        try:
            next(iterator)
            consumed_this_epoch += 1
        except StopIteration:
            epoch += 1
            consumed_this_epoch = 0
            iterator = _view_batches(ctx, ctx.seed, epoch)
        skip_batches -= 1
    for step in range(start_step, ctx.recipe.steps):
        try:
            user_ids, seqs = next(iterator)
            consumed_this_epoch += 1
        except StopIteration:
            epoch += 1
            consumed_this_epoch = 0
            iterator = _view_batches(ctx, ctx.seed, epoch)
            user_ids, seqs = next(iterator)
            consumed_this_epoch = 1
        seqs = seqs.to(ctx.device)
        step_record = _train_one_step(
            ctx, model, optimizer, scheduler, step, user_ids, seqs, views,
            weights=weights, gate=gate, gate_optimizer=gate_optimizer,
            entropy_coef=entropy_coef,
        )
        for key in ("loss", "rec_loss", "cl_loss", "m_c", "grad_norm_pre", "grad_norm_post"):
            metric_history[key].append(step_record[key])
        for view, val in zip(views, step_record["a_p"]):
            a_p_history.setdefault(view, []).append(val)

        if (step + 1) % interval == 0 or step + 1 == ctx.recipe.steps:
            save_coalition_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step + 1,
                seed=ctx.seed,
                coalition=coalition,
                policy=policy,
                rng_states={"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
                config_hash=ctx.config_hash,
                data_hash=ctx.data_hash,
                recipe_hash=ctx.recipe_hash,
                metric_history=metric_history,
                extra={"a_p_history": a_p_history, "weights": weights, "entropy_coef": entropy_coef,
                    "gate_state": gate.state_dict() if gate is not None else None,
                    "batches_consumed_this_epoch": consumed_this_epoch},
                dataset=ctx.cfg.dataset,
            )
            # numbered snapshot (kept for recipe step curves / crash recovery)
            save_coalition_checkpoint(
                os.path.join(cdir, f"step_{step + 1}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                step=step + 1,
                seed=ctx.seed,
                coalition=coalition,
                policy=policy,
                rng_states={"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
                config_hash=ctx.config_hash,
                data_hash=ctx.data_hash,
                recipe_hash=ctx.recipe_hash,
                metric_history=metric_history,
                extra={"a_p_history": a_p_history, "weights": weights, "entropy_coef": entropy_coef,
                    "gate_state": gate.state_dict() if gate is not None else None,
                    "batches_consumed_this_epoch": consumed_this_epoch},
                dataset=ctx.cfg.dataset,
            )
        if (step + 1) % ctx.log_interval == 0:
            ctx.logger.info(
                "PRIMARY_GAME_A", "training_step", seed=ctx.seed, coalition=coalition,
                policy=policy, dataset=ctx.cfg.dataset, step=step + 1,
                loss=round(step_record["loss"], 6), m_c=round(step_record["m_c"], 6),
                grad_norm=round(step_record["grad_norm_pre"], 4),
            )

    final_hash = save_coalition_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        step=ctx.recipe.steps,
        seed=ctx.seed,
        coalition=coalition,
        policy=policy,
        rng_states={"torch": torch.get_rng_state(), "numpy": np.random.get_state()},
        config_hash=ctx.config_hash,
        data_hash=ctx.data_hash,
        recipe_hash=ctx.recipe_hash,
        metric_history=metric_history,
        extra={"a_p_history": a_p_history, "weights": weights, "entropy_coef": entropy_coef,
               "gate_state": gate.state_dict() if gate is not None else None,
               "batches_consumed_this_epoch": consumed_this_epoch},
        dataset=ctx.cfg.dataset,
    )
    ctx.manifest.mark_coalition_complete(
        dataset=ctx.cfg.dataset,
        seed=ctx.seed,
        coalition=coalition,
        policy=policy,
        step=ctx.recipe.steps,
        checkpoint_path=final_path,
        checkpoint_hash=final_hash,
        config_hash=ctx.config_hash,
        data_hash=ctx.data_hash,
        recipe_hash=ctx.recipe_hash,
    )
    ctx.logger.info(
        "PRIMARY_GAME_A", "coalition_training_end", seed=ctx.seed, coalition=coalition,
        policy=policy, dataset=ctx.cfg.dataset, wall_seconds=round(time.time() - t0, 2),
    )
    return _result_payload(
        ctx, coalition, policy, model, ctx.recipe.steps, epoch, final_path, final_hash,
        metric_history, a_p_history, cache_hit=False, wall_seconds=time.time() - t0,
    )


def coalition_views(ctx: TrainContext) -> Tuple[str, ...]:
    """Main K=3 player set for this dataset/context."""
    from . import MAIN_PLAYERS

    return MAIN_PLAYERS


def coalition_dir_for(ctx: TrainContext, coalition: Sequence[str], policy: str) -> str:
    """Checkpoint directory for one coalition; ctx.run_dir is already the
    checkpoints root (results/runs/<run>/checkpoints)."""
    cname = "+".join(sorted(coalition)) if coalition else "empty"
    d = os.path.join(ctx.run_dir, f"seed{ctx.seed}", policy, cname)
    os.makedirs(d, exist_ok=True)
    return d


__all__ = [
    "Recipe",
    "TrainContext",
    "CoalitionTrainResult",
    "DatasetLevelGates",
    "make_optimizer",
    "make_scheduler",
    "common_initialization",
    "train_coalition",
    "coalition_dir",
    "coalition_dir_for",
    "coalition_views",
]
