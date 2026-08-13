"""Coalition training stages.

    python scripts/train_coalitions.py --dataset synthetic --stage game-a
        [--seeds 2001,2002 --coalition crop+mask]

Stages:
  pilot        Game A, all 8 coalitions, excluded pilot seeds (1001, 1002)
  game-a       PRIMARY_GAME_A: all 8 coalitions x confirmatory seeds 2001-2005
               (conditional 2006-2010 extension only under the locked
               direction-blind rule, requested explicitly with --extend)
  game-b       SECONDARY_GAME_B: ONLY the 6 intermediate coalitions for seeds
               2001-2003 (empty/grand are reused from Game A at evaluation)
  k4-mc        K4_BEAUTY_MC: antithetic permutation paths + grand-LOO
               coalitions for seeds 3001-3005 (<=10 unique models/seed)
  nondet-floor identical-configuration repeat of the canonical Game-A grand
               coalition on seed 2001 (measures the nondeterminism floor)

Every coalition trains from the common seed-specific initialization with a
fixed optimizer-step budget; completed coalitions are cached and reused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from shaper import K4_PLAYERS, MAIN_PLAYERS  # noqa: E402
from shaper.checkpoints import CheckpointManifest  # noqa: E402
from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.game import (  # noqa: E402
    CoalitionValueRecord,
    all_coalitions,
    evaluate_and_record,
    intermediate_coalitions,
    save_coalition_table,
    subtract_baseline,
)
from shaper.monte_carlo import K4SeedResult, required_coalitions, sample_permutation  # noqa: E402
from shaper.training import Recipe, TrainContext, train_coalition  # noqa: E402


def make_ctx(cfg, data, run, seed, recipe, device, policy, base_state_path):
    manifest = CheckpointManifest(os.path.join(run.checkpoint_root(), "manifest-root"))
    return TrainContext(
        cfg=cfg,
        data=data,
        recipe=recipe,
        seed=seed,
        policy=policy,
        run_dir=run.checkpoint_root(),
        logger=run.logger,
        manifest=manifest,
        config_hash=cfg.config_hash(),
        device=device,
        log_interval=max(1, min(50, recipe.steps // 5)),
        checkpoint_interval=max(1, min(cfg.training["checkpoint_interval"], recipe.steps)),
        base_state_path=base_state_path,
    )


def train_one_coalition(cfg, data, run, seed, recipe, device, policy, coalition, base_state_path, stage):
    ctx = make_ctx(cfg, data, run, seed, recipe, device, policy, base_state_path)
    result = train_coalition(ctx, coalition)
    run.tracker.cache_hits += int(result.cache_hit)
    run.tracker.cache_misses += int(not result.cache_hit)
    if not result.cache_hit:
        run.tracker.n_coalition_models += 1
        run.tracker.training_steps += result.steps
    record = evaluate_and_record(
        result.model, data, cfg.dataset, seed, policy, coalition,
        cfg.evaluation["k"], roles=("tune", "game", "select"),
    )
    return result, record


def stage_game_a(cfg, data, run, args, recipe, device, stage_name):
    seeds = args.seeds or cfg.seeds["confirmatory_game_a"]
    policy = "game_a"
    records_by_seed = {}
    for seed in seeds:
        init_path = os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt")
        records = []
        for coalition in all_coalitions(MAIN_PLAYERS):
            _, record = train_one_coalition(cfg, data, run, seed, recipe, device, policy, coalition, init_path, stage_name)
            records.append(record)
        subtract_baseline(records)
        path = os.path.join(run.path("coalition_tables"), f"game_a_seed{seed}.json")
        save_coalition_table(records, path)
        records_by_seed[seed] = path
    return {"policy": "game_a", "seeds": seeds, "tables": records_by_seed}


def stage_game_b(cfg, data, run, args, recipe, device, stage_name):
    seeds = cfg.seeds["game_b"]  # locked: 2001-2003 only
    policy = "game_b"
    records_by_seed = {}
    for seed in seeds:
        init_path = os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt")
        records = []
        for coalition in intermediate_coalitions(MAIN_PLAYERS):  # 6 intermediates ONLY
            _, record = train_one_coalition(cfg, data, run, seed, recipe, device, policy, coalition, init_path, stage_name)
            records.append(record)
        # empty/grand are REUSED from Game A (never retrained for Game B)
        game_a_path = os.path.join(run.root, "coalition_tables", f"game_a_seed{seed}.json")
        if not os.path.exists(game_a_path):
            raise SystemExit(
                f"SECONDARY_GAME_B requires the Game-A empty/grand models for seed {seed} "
                f"(missing {game_a_path}); run PRIMARY_GAME_A for that seed first."
            )
        from shaper.game import load_coalition_table, load_per_user_table

        game_a = load_coalition_table(game_a_path)
        game_a_pu = load_per_user_table(game_a_path) or {}
        for rec in game_a:
            if rec.coalition in ((), tuple(MAIN_PLAYERS)):
                clone = CoalitionValueRecord(
                    dataset=rec.dataset, seed=rec.seed, policy=policy,
                    coalition=rec.coalition,
                    metrics_by_role=rec.metrics_by_role,
                    per_user_ndcg_game=game_a_pu.get(rec.coalition),
                    per_user_nll_game=rec.per_user_nll_game,
                )
                records.append(clone)
        subtract_baseline(records)
        path = os.path.join(run.path("coalition_tables"), f"game_b_seed{seed}.json")
        save_coalition_table(records, path)
        records_by_seed[seed] = path
    return {"policy": "game_b", "seeds": seeds, "tables": records_by_seed,
            "note": "empty/grand reused from Game A; 6 intermediate coalitions only"}


def stage_k4_mc(cfg, data, run, args, recipe, device, stage_name):
    if cfg.dataset != "beauty" and cfg.dataset != "synthetic":
        raise SystemExit("K4_BEAUTY_MC is registered for Beauty only")
    seeds = args.seeds or cfg.seeds["k4_beauty_mc"]
    players = K4_PLAYERS
    policy = "game_a"
    out = {}
    for seed in seeds:
        init_path = os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt")
        pi = sample_permutation(seed, players)  # frozen run state: never resampled
        k4 = K4SeedResult(
            seed=seed,
            permutation=pi,
            reverse_permutation=tuple(reversed(pi)),
            coalition_registry=required_coalitions(pi, players),
        )
        records = []
        for coalition in sorted(k4.coalition_registry.values(), key=lambda c: (len(c), c)):
            _, record = train_one_coalition(cfg, data, run, seed, recipe, device, policy, coalition, init_path, stage_name)
            records.append(record)
            k4.completed.append("+".join(coalition) if coalition else "empty")
        subtract_baseline(records)
        k4.values = {rec.coalition: rec.value_ndcg for rec in records}
        path = run.write_json("coalition_tables", f"k4_seed{seed}.json", k4.to_dict())
        out[seed] = path
    return {"policy": "game_a", "seeds": seeds, "tables": out,
            "label": "Monte-Carlo approximate",
            "max_unique_models_per_seed": 10}


def stage_pilot(cfg, data, run, args, recipe, device, stage_name):
    seeds = cfg.seeds["pilots"]
    policy = "game_a"
    out = {}
    for seed in seeds:
        init_path = os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt")
        records = []
        for coalition in all_coalitions(MAIN_PLAYERS):
            _, record = train_one_coalition(cfg, data, run, seed, recipe, device, policy, coalition, init_path, stage_name)
            records.append(record)
        subtract_baseline(records)
        path = os.path.join(run.path("coalition_tables"), f"pilot_seed{seed}.json")
        save_coalition_table(records, path)
        out[seed] = path
    return {"policy": "game_a", "seeds": seeds, "tables": out,
            "note": "pilot outcomes excluded from confirmatory estimates"}


def stage_nondet_floor(cfg, data, run, args, recipe, device, stage_name):
    """Identical-configuration repeat of the canonical Game-A grand coalition
    on seed 2001 with an additional run suffix (spec A.2).

    The repeat trains with the IDENTICAL configuration (same policy, recipe,
    schedules, init) but into a separate checkpoint directory with its own
    manifest, so it is a genuine second execution rather than a cache hit.
    """
    from shaper.checkpoints import CheckpointManifest
    from shaper.logging_utils import StructuredLogger

    seed = cfg.seeds["nondeterminism_floor_repeat_seed"]
    init_path = os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt")
    coalition = list(MAIN_PLAYERS)
    # main run
    _, rec_main = train_one_coalition(cfg, data, run, seed, recipe, device, "game_a", coalition, init_path, stage_name)
    # repeat: separate directory + manifest => forced fresh training
    repeat_dir = os.path.join(run.checkpoint_root(), "nondet-floor-repeat")
    repeat_manifest = CheckpointManifest(os.path.join(repeat_dir, "manifest"))
    ctx = TrainContext(
        cfg=cfg,
        data=data,
        recipe=recipe,
        seed=seed,
        policy="game_a",
        run_dir=repeat_dir,
        logger=StructuredLogger(os.path.join(repeat_dir, "logs"), run.run_id),
        manifest=repeat_manifest,
        config_hash=cfg.config_hash(),
        device=device,
        log_interval=max(1, min(50, recipe.steps // 5)),
        checkpoint_interval=max(1, min(cfg.training["checkpoint_interval"], recipe.steps)),
        base_state_path=init_path,
    )
    rep = train_coalition(ctx, coalition)
    assert not rep.cache_hit, "nondeterminism-floor repeat must be a fresh execution"
    run.tracker.n_coalition_models += 1
    run.tracker.training_steps += rep.steps
    rec_rep = evaluate_and_record(rep.model, data, cfg.dataset, seed, "game_a", coalition, cfg.evaluation["k"])
    floor = abs(rec_main.metrics_by_role["game"]["ndcg"] - rec_rep.metrics_by_role["game"]["ndcg"])
    out = {
        "seed": seed,
        "coalition": coalition,
        "main_ndcg_game": rec_main.metrics_by_role["game"]["ndcg"],
        "repeat_ndcg_game": rec_rep.metrics_by_role["game"]["ndcg"],
        "nondeterminism_floor": floor,
        "note": "contextual marginals whose magnitude does not exceed this floor are "
                "labeled indistinguishable from execution nondeterminism",
    }
    run.write_json("metrics", "nondeterminism_floor.json", out)
    return out


STAGES = {
    "pilot": stage_pilot,
    "game-a": stage_game_a,
    "game-b": stage_game_b,
    "k4-mc": stage_k4_mc,
    "nondet-floor": stage_nondet_floor,
}


def main() -> int:
    from _cli import base_parser, frozen_data, resolve_device, resolve_recipe

    p = base_parser("train SHAPER coalition models")
    p.add_argument("--stage", required=True, choices=list(STAGES), help="which training stage to run")
    p.add_argument("--seeds", default=None, help="comma-separated seed list (game-a/k4-mc only)")
    p.add_argument("--coalition", default=None, help="single coalition (players joined by +; 'empty' for rec-only)")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    data = frozen_data(cfg)
    device = resolve_device(args)
    # pre-archive stages (pilots) read the recipe from the run's calibration
    # file; post-archive stages read the frozen manifest.
    recipe = None
    if args.run_id:
        calib = os.path.join(cfg.paths["results"], args.run_id, "recipe", "calibration.json")
        if os.path.exists(calib):
            with open(calib) as fh:
                recipe = Recipe(**json.load(fh)["recipe"])
    if recipe is None:
        recipe = resolve_recipe(cfg)

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"coalitions-{cfg.dataset}-{args.stage}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)
    run.update_manifest(data_hash=data.data_hash, recipe_hash=recipe.hash())
    stage_name = {
        "pilot": "PILOT_1001_1002",
        "game-a": "PRIMARY_GAME_A",
        "game-b": "SECONDARY_GAME_B",
        "k4-mc": "K4_BEAUTY_MC",
        "nondet-floor": "SEVERITY_DIAGNOSTICS",
    }[args.stage]
    run.logger.stage_start(stage_name, dataset=cfg.dataset)
    t0 = time.time()
    try:
        out = STAGES[args.stage](cfg, data, run, args, recipe, device, stage_name)
        run.tracker.record_memory()
        run.stage_status(stage_name, "completed", wall_seconds=round(time.time() - t0, 2), **out)
        run.write_json("metrics", f"compute_{args.stage}.json", run.tracker.summary())
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
    except Exception as exc:  # noqa: BLE001 - stage failure is recorded, not swallowed
        run.logger.stage_fail(stage_name, str(exc), dataset=cfg.dataset)
        run.stage_status(stage_name, "failed", error=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
