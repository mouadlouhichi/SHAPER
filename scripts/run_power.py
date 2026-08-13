"""POWER stage (Appendix J).

Builds the pilot-informed MDE/CI-width table from finalized V_game sizes,
pilot variance estimates, and the prespecified variance grid, using the
conservative planning formula
    h = t_(.975, S-1) * sqrt(sigma_seed^2 / S + sigma_user^2 / N_game)
plus a hierarchical-bootstrap estimate. If pilots have not run, the
prespecified grid alone is reported and labeled provisional.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.power import pilot_informed_table, variance_grid_table  # noqa: E402
from shaper.stats import hierarchical_bootstrap_ci, planning_half_width  # noqa: E402


def pilot_variances(run, data, dataset):
    """Seed variance of raw Shapley values across the excluded pilot seeds."""
    from shaper.game import load_coalition_table
    from shaper.shapley import exact_shapley
    from shaper import MAIN_PLAYERS

    per_seed = []
    for seed in (1001, 1002):
        path = os.path.join(run.root, "coalition_tables", f"pilot_seed{seed}.json")
        if not os.path.exists(path):
            continue
        records = load_coalition_table(path)
        values = {rec.coalition: rec.value_ndcg for rec in records}
        per_seed.append(exact_shapley(values, MAIN_PLAYERS))
    if not per_seed:
        return None, None
    phi = {p: [s[p] for s in per_seed] for p in MAIN_PLAYERS}
    variances = [float(np.var(v, ddof=1)) for v in phi.values()] if len(per_seed) > 1 else [float("nan")]
    return per_seed, variances


def main() -> int:
    from _cli import base_parser, frozen_data

    p = base_parser("SHAPER Appendix J power tables")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    data = frozen_data(cfg)

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"power-{cfg.dataset}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)
    run.logger.stage_start("POWER", dataset=cfg.dataset)
    t0 = time.time()

    N_game = int(len(data.user_ids_for_role("game")))
    grid = cfg.statistics["planning"]["variance_grid"]
    deltas = cfg.statistics["thresholds"]

    per_seed, variances = pilot_variances(run, data, cfg.dataset)
    if per_seed:
        # pilot-informed: estimate sigma_user from the two pilot seeds
        sigma_user_pilot = float(np.sqrt(np.mean(variances)))
        table = pilot_informed_table(N_game, [float(np.var([s["crop"] for s in per_seed], ddof=1))], sigma_user_pilot, deltas)
        table["hierarchical_bootstrap"] = hierarchical_bootstrap_ci(
            [[s[p] for s in per_seed] for p in ("crop", "mask", "reorder")]
        )
        status = "pilot-informed"
    else:
        table = variance_grid_table(
            N_game, [5, 10], grid["sigma_user"], grid["sigma_seed"]
        )
        table["deltas"] = deltas
        table["note"] = "provisional grid only; regenerate after the two excluded pilot seeds"
        status = "provisional (pilots not yet run)"
    table["status"] = status
    table["N_game"] = N_game
    run.write_json("metrics", "appendix_j_power.json", table)
    run.stage_status("POWER", "completed", wall_seconds=round(time.time() - t0, 2), status=status)
    run.logger.stage_end("POWER", dataset=cfg.dataset)
    print(json.dumps(table, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
