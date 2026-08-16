"""SURROGATE_VALIDATION stage (spec A.6 appendix + feasibility amendment).

Trains the cached ranking-adapter surrogate for all 8 Game-A coalitions of
one seed (default: the amendment's validation seed 2001) starting from the
rec-only (empty-coalition) checkpoint, evaluates on the frozen roles, and
reports the surrogate attribution error against the full-retraining table:

    results/runs/<run>/surrogate/surrogate_validation_seed<seed>.json

The surrogate is a feasibility validation layer; it never replaces the
primary full-retraining table in confirmatory reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from shaper import MAIN_PLAYERS  # noqa: E402
from shaper.checkpoints import CheckpointManifest, load_coalition_checkpoint  # noqa: E402
from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.game import (  # noqa: E402
    all_coalitions,
    evaluate_and_record,
    load_coalition_table,
    save_coalition_table,
    subtract_baseline,
)
from shaper.surrogate import build_surrogate_factory, surrogate_validation_report  # noqa: E402
from shaper.training import Recipe, TrainContext, train_coalition  # noqa: E402


def main() -> int:
    from _cli import base_parser, frozen_data, resolve_device

    p = base_parser("cached ranking-adapter surrogate validation (appendix)")
    p.add_argument("--seed", type=int, default=None,
                   help="full Game-A seed to validate against (default: amendment validation seed)")
    p.add_argument("--surrogate-steps", type=int, default=None,
                   help="surrogate training steps (default: amendment surrogate.steps)")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    data = frozen_data(cfg)
    device = resolve_device(args)

    surrogate_cfg = cfg.statistics.get("surrogate", {})
    # default: the amendment-pinned validation seed, else the first
    # confirmatory Game-A seed (which has a full-retraining table to
    # validate against); seed 4001 is the registered Beauty appendix seed.
    seed = int(args.seed or surrogate_cfg.get("validation_seed")
               or cfg.seeds["confirmatory_game_a"][0])
    steps = int(args.surrogate_steps or surrogate_cfg.get("steps", 500))

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"surrogate-{cfg.dataset}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)
    run.logger.stage_start("SURROGATE_VALIDATION", dataset=cfg.dataset, seed=seed)
    t0 = time.time()

    # 1. frozen rec-only base checkpoint (empty coalition of the full game)
    base_path = os.path.join(
        run.checkpoint_root(), f"seed{seed}", "game_a", "empty", "final.pt"
    )
    if not os.path.exists(base_path):
        raise SystemExit(
            f"surrogate requires the full Game-A rec-only checkpoint for seed {seed} "
            f"(missing {base_path}); run PRIMARY_GAME_A for that seed first"
        )
    base_payload = load_coalition_checkpoint(base_path)
    base_state = base_payload["model"]

    # 2. full-retraining table for the comparison
    full_table_path = os.path.join(run.root, "coalition_tables", f"game_a_seed{seed}.json")
    if not os.path.exists(full_table_path):
        raise SystemExit(
            f"surrogate validation requires the full Game-A table {full_table_path}"
        )
    full_records = load_coalition_table(full_table_path)
    full_values = {rec.coalition: rec.value_ndcg for rec in full_records}

    # 3. train the surrogate for every coalition (short budget, frozen backbone)
    recipe = Recipe(learning_rate=1.0e-3, steps=steps, lambda_cl=0.1, tau=0.1)
    policy = f"surrogate_s{seed}"
    factory = build_surrogate_factory(base_state, d=int(cfg.model["d"]))
    manifest = CheckpointManifest(os.path.join(run.checkpoint_root(), "surrogate-manifest"))
    records = []
    for coalition in all_coalitions(MAIN_PLAYERS):
        ctx = TrainContext(
            cfg=cfg, data=data, recipe=recipe, seed=seed, policy=policy,
            run_dir=run.checkpoint_root(), logger=run.logger, manifest=manifest,
            config_hash=cfg.config_hash(), device=device,
            log_interval=max(1, min(50, steps // 5)),
            checkpoint_interval=max(1, steps),
            model_factory=factory,
        )
        result = train_coalition(ctx, coalition)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        run.tracker.training_steps += result.steps if not result.cache_hit else 0
        record = evaluate_and_record(
            result.model, data, cfg.dataset, seed, policy, coalition,
            cfg.evaluation["k"], roles=("tune", "game", "select"),
        )
        records.append(record)

    subtract_baseline(records)
    save_coalition_table(records, os.path.join(run.path("coalition_tables"), f"surrogate_seed{seed}.json"))
    surrogate_values = {rec.coalition: rec.value_ndcg for rec in records}

    report = surrogate_validation_report(surrogate_values, full_values, MAIN_PLAYERS)
    report.update({
        "seed": seed,
        "surrogate_steps": steps,
        "base_checkpoint": base_path,
        "full_table": full_table_path,
        "wall_seconds": round(time.time() - t0, 2),
        "amendment_id": cfg.amendment.get("id"),
    })
    run.write_json("surrogate", f"surrogate_validation_seed{seed}.json", report)
    run.stage_status("SURROGATE_VALIDATION", "completed",
                     wall_seconds=round(time.time() - t0, 2),
                     spearman_rho=report["spearman_rho_over_values"],
                     value_mae=report["value_mae"])
    run.logger.stage_end("SURROGATE_VALIDATION", dataset=cfg.dataset, seed=seed)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
