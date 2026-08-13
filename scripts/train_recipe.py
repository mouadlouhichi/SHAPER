"""RECIPE_CALIBRATION stage (spec A.6, "Fixed training budget and
hyperparameter comparability"). Single pass, V_tune only.

Step 1 (seed 901): empty coalition, 5,000 steps, learning rates
{1e-4, 5e-4, 1e-3} -> lowest full-catalog target NLL; tie -> smaller rate.
Step 2 (seeds 901-903, selected LR, provisional lambda=0.1 tau=0.1): empty
and grand to 10,000 steps, checkpoints at {2500, 5000, 10000} -> step count
minimizing mean empty/grand V_tune NLL; tie -> fewer steps.
Step 3 (seed 901 at locked steps): all nine (lambda, tau) pairs from
{0.05,0.1,0.2} x {0.07,0.1,0.2}; best two advance to seeds 902/903; select
lower mean-NLL pair; ties -> smaller lambda, then larger tau.

Freezes learning rate, step count, lambda_cl and tau for every declared
coalition game. Results are written to <run>/recipe/calibration.json and
consumed by ARCHIVE_FREEZE. Also runs the bounded singleton lambda stress
test (V_tune, seed 901, {0.05, 0.1, 0.2}).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from shaper.checkpoints import CheckpointManifest, load_coalition_checkpoint  # noqa: E402
from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.metrics import evaluate_model  # noqa: E402
from shaper.training import Recipe, TrainContext, build_model, train_coalition  # noqa: E402


def _evaluate_from_checkpoint(cfg, data, ctx, path, device):
    payload = load_coalition_checkpoint(path)
    model = build_model(cfg, data.n_items)
    model.load_state_dict(payload["model"])
    res = evaluate_model(model, data.eval_inputs("tune"), cfg.evaluation["k"], device=device)
    return res.nll


def step1(cfg, data, run, device, seeds_cfg):
    """Learning-rate selection on empty coalition, seed 901."""
    lr_grid = cfg.statistics["recipe"]["learning_rates"]
    steps = cfg.statistics["recipe"]["step1_steps"]
    scores = {}
    for lr in lr_grid:
        ctx = _ctx(cfg, data, run, seed=seeds_cfg["recipe_alpha"][0], recipe=Recipe(lr, steps, 0.0, 0.1), device=device)
        result = train_coalition(ctx, [])
        nll = _evaluate_from_checkpoint(cfg, data, ctx, result.checkpoint_path, device)
        scores[lr] = nll
        run.tracker.n_coalition_models += 1
        run.tracker.training_steps += result.steps
    best = min(scores, key=lambda k: (scores[k], k))  # tie -> smaller LR
    return {"scores": {str(k): v for k, v in scores.items()}, "selected_learning_rate": best}


def step2(cfg, data, run, device, seeds_cfg, best_lr):
    """Step-count selection from empty/grand NLL curves on seeds 901-903."""
    step_grid = cfg.statistics["recipe"]["step_grid"]
    lambda_cl = cfg.statistics["recipe"]["provisional_lambda"]
    tau = cfg.statistics["recipe"]["provisional_tau"]
    max_steps = max(step_grid)
    seed_scores = {s: {} for s in seeds_cfg["recipe_alpha"]}
    for seed in seeds_cfg["recipe_alpha"]:
        ctx = _ctx(cfg, data, run, seed=seed, recipe=Recipe(best_lr, max_steps, lambda_cl, tau), device=device)
        for coalition in ([], ["crop", "mask", "reorder"]):
            result = train_coalition(ctx, coalition)
            run.tracker.n_coalition_models += 1
            run.tracker.training_steps += result.steps
            cdir = os.path.join(run.checkpoint_root(), f"seed{seed}", ctx.policy, "+".join(coalition) if coalition else "empty")
            for step in step_grid:
                snap = os.path.join(cdir, f"step_{step}.pt")
                nll = _evaluate_from_checkpoint(cfg, data, ctx, snap, device)
                seed_scores[seed].setdefault(step, []).append(nll)
    means = {step: sum(v for vals in seed_scores.values() for v in vals[step]) / sum(len(vals[step]) for vals in seed_scores.values()) for step in step_grid}
    best = min(means, key=lambda k: (means[k], k))  # tie -> fewer steps
    return {"mean_nll_by_step": means, "selected_steps": best, "per_seed": {str(s): {str(k): v for k, v in d.items()} for s, d in seed_scores.items()}}


def step3(cfg, data, run, device, seeds_cfg, best_lr, best_steps):
    """(lambda, tau) grid: 9 pairs on seed 901; best two advance to 902/903."""
    lambda_grid = cfg.statistics["recipe"]["lambda_grid"]
    tau_grid = cfg.statistics["recipe"]["tau_grid"]
    pairs = list(itertools.product(lambda_grid, tau_grid))
    scores_901 = {}
    for (lam, tau) in pairs:
        ctx = _ctx(cfg, data, run, seed=seeds_cfg["recipe_alpha"][0], recipe=Recipe(best_lr, best_steps, lam, tau), device=device)
        result = train_coalition(ctx, ["crop", "mask", "reorder"])
        run.tracker.n_coalition_models += 1
        run.tracker.training_steps += result.steps
        scores_901[(lam, tau)] = _evaluate_from_checkpoint(cfg, data, ctx, result.checkpoint_path, device)
    ranked = sorted(pairs, key=lambda pt: (scores_901[pt], pt[0], -pt[1]))
    advanced = ranked[: cfg.statistics["recipe"]["step2_advance"]]
    final_scores = {pt: scores_901[pt] for pt in pairs}
    for seed in seeds_cfg["recipe_alpha"][1:]:
        for (lam, tau) in advanced:
            ctx = _ctx(cfg, data, run, seed=seed, recipe=Recipe(best_lr, best_steps, lam, tau), device=device)
            result = train_coalition(ctx, ["crop", "mask", "reorder"])
            run.tracker.n_coalition_models += 1
            run.tracker.training_steps += result.steps
            nll = _evaluate_from_checkpoint(cfg, data, ctx, result.checkpoint_path, device)
            final_scores[(lam, tau)] += nll
    means = {pt: final_scores[pt] / 3 for pt in final_scores}
    best = min(advanced, key=lambda pt: (means[pt], pt[0], -pt[1]))  # smaller lambda, then larger tau
    return {"scores_seed901": {str(k): v for k, v in scores_901.items()},
            "advanced_pairs": [list(p) for p in advanced],
            "mean_nll": {str(k): v for k, v in means.items()},
            "selected": {"lambda_cl": best[0], "tau": best[1]}}


def singleton_lambda_stress(cfg, data, run, device, seeds_cfg, recipe):
    """Bounded singleton lambda stress test (V_tune, seed 901, 3 values)."""
    out = {}
    for lam in cfg.statistics["recipe"]["singleton_lambda_stress"]:
        r = Recipe(recipe.learning_rate, recipe.steps, lam, recipe.tau)
        ctx = _ctx(cfg, data, run, seed=seeds_cfg["recipe_alpha"][0], recipe=r, device=device)
        result = train_coalition(ctx, ["crop"])
        run.tracker.n_coalition_models += 1
        run.tracker.training_steps += result.steps
        out[str(lam)] = _evaluate_from_checkpoint(cfg, data, ctx, result.checkpoint_path, device)
    return out


def _ctx(cfg, data, run, seed, recipe, device):
    manifest = CheckpointManifest(os.path.join(run.checkpoint_root(), "recipe"))
    # Recipe calibration trains empty/grand coalitions under Game-A semantics
    # (grand objective is identical to Game B's for K=3); seeds 901-903 never
    # appear in confirmatory runs, so the policy label "game_a" is safe.
    return TrainContext(
        cfg=cfg,
        data=data,
        recipe=recipe,
        seed=seed,
        policy="game_a",
        run_dir=run.checkpoint_root(),
        logger=run.logger,
        manifest=manifest,
        config_hash=cfg.config_hash(),
        device=device,
        log_interval=max(10, recipe.steps // 10),
        checkpoint_interval=max(1, min(cfg.training["checkpoint_interval"], recipe.steps)),
    )


def main() -> int:
    from _cli import base_parser, frozen_data, resolve_device

    p = base_parser("SHAPER recipe calibration (single pass, V_tune)")
    p.add_argument("--skip-stress", action="store_true")
    p.add_argument("--stress-only", action="store_true")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    # verification-only grid shrink for the synthetic dataset (documented in
    # configs/synthetic.yaml); real datasets use the locked grids unchanged
    if cfg.raw.get("recipe_override"):
        cfg.statistics["recipe"].update(cfg.raw["recipe_override"])
    data = frozen_data(cfg)
    device = resolve_device(args)

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"recipe-{cfg.dataset}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg)
    run.logger.stage_start("RECIPE_CALIBRATION", dataset=cfg.dataset)
    t0 = time.time()

    if not args.stress_only:
        s1 = step1(cfg, data, run, device, cfg.seeds)
        s2 = step2(cfg, data, run, device, cfg.seeds, s1["selected_learning_rate"])
        s3 = step3(cfg, data, run, device, cfg.seeds, s1["selected_learning_rate"], s2["selected_steps"])
        recipe = Recipe(
            learning_rate=s1["selected_learning_rate"],
            steps=s2["selected_steps"],
            lambda_cl=s3["selected"]["lambda_cl"],
            tau=s3["selected"]["tau"],
        )
    else:
        existing = json.load(open(os.path.join(run.root, "recipe", "calibration.json")))
        recipe = Recipe(**existing["recipe"])

    stress = {} if args.skip_stress else singleton_lambda_stress(cfg, data, run, device, cfg.seeds, recipe)

    out = {
        "recipe": recipe.to_dict(),
        "recipe_hash": recipe.hash(),
        "step1": {} if args.stress_only else s1,
        "step2": {} if args.stress_only else s2,
        "step3": {} if args.stress_only else s3,
        "singleton_lambda_stress": stress,
        "wall_seconds": round(time.time() - t0, 2),
        "note": "single-pass calibration on V_tune; frozen for every declared coalition game",
    }
    run.write_json("recipe", "calibration.json", out)
    run.update_manifest(recipe_hash=recipe.hash())
    run.stage_status("RECIPE_CALIBRATION", "completed", recipe=recipe.to_dict())
    run.logger.stage_end("RECIPE_CALIBRATION", dataset=cfg.dataset)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
