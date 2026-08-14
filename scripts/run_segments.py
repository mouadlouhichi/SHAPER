"""SEGMENTS stage (RQ3).

Confirmatory inference uses ONLY exact K=3 Game-A per-user values on the
fixed V_game users with the frozen Q1-Q4 pre-truncation training-length
segments. Game B and the K=4 MC extension are excluded.

Tests (studentized, 10,000 user-level label permutations; ONE permuted user
map applied across ALL seed records):
    T_mask = sum_m (m - 2.5) * mu[m, mask]     (confirmatory increasing trend)
    T_all  = sum_{m,p} (mu[m,p]-mu[p])^2 / (SE[m,p]^2 + eps)   (omnibus)

Crop/reorder trends are exploratory. Outputs: raw segment tables + Figure 5.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from shaper import MAIN_PLAYERS  # noqa: E402
from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.segments import (  # noqa: E402
    label_permutation_test,
    segment_means,
    segment_ses,
)
from shaper.shapley import per_user_shapley  # noqa: E402


def load_per_user_tables(run, data, policy: str, seeds):
    from shaper.game import load_per_user_table

    tables = []
    found = []
    for seed in seeds:
        path = os.path.join(run.root, "coalition_tables", f"{policy}_seed{seed}.json")
        if not os.path.exists(path):
            continue
        table = load_per_user_table(path)
        if table is None:
            continue
        tables.append(table)
        found.append(seed)
    return tables, found


def main() -> int:
    from _cli import base_parser, frozen_data

    p = base_parser("SHAPER RQ3 segment analysis (exact Game A only)")
    p.add_argument("--n-permutations", type=int, default=None)
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    data = frozen_data(cfg)

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"segments-{cfg.dataset}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)
    run.logger.stage_start("SEGMENTS", dataset=cfg.dataset)
    t0 = time.time()

    seeds = cfg.seeds["confirmatory_game_a"]
    tables, found = load_per_user_tables(run, data, "game_a", seeds)
    if not tables:
        raise SystemExit("SEGMENTS requires per-user Game-A tables (run SHAPLEY first)")

    per_seed_phi = [per_user_shapley([t], MAIN_PLAYERS) for t in tables]
    phi_mean = {p: np.mean([s[p] for s in per_seed_phi], axis=0) for p in MAIN_PLAYERS}
    game_users = data.user_ids_for_role("game")
    quartiles = data.quartiles[game_users]  # V_game users only
    mu = segment_means(phi_mean, quartiles)
    se = segment_ses(phi_mean, quartiles)

    n_perm = args.n_permutations or cfg.statistics["segments"]["permutations"]
    perm = label_permutation_test(per_seed_phi, quartiles, n_permutations=n_perm, seed=0)

    out = {
        "seeds_used": found,
        "n_vgame_users": {str(m): int((quartiles == m).sum()) for m in range(1, 5)},
        "quartile_edges": data.quartile_edges,
        "mu": mu,
        "se": se,
        "mask_trend_T": perm["T_mask_observed"],
        "mask_trend_p": perm["p_mask_one_sided"],
        "omnibus_T": perm["T_all_observed"],
        "omnibus_p": perm["p_all"],
        "n_permutations": n_perm,
        "note": ("confirmatory: mask trend + omnibus profile; crop/reorder trends "
                 "exploratory; Game B and K=4 MC excluded from confirmatory inference"),
    }
    run.write_json("segments", "segment_attribution.json", out)
    from shaper.report import figure_segment_lines, segment_table

    table_md = segment_table(mu, se, MAIN_PLAYERS)
    run.write_markdown("tables", "table6_segments.md", table_md)
    fig = figure_segment_lines(mu, MAIN_PLAYERS, run.path("figures", "fig5_segments.png"))
    run.stage_status("SEGMENTS", "completed", wall_seconds=round(time.time() - t0, 2),
                     figures=[fig], tables=["table6_segments.md"])
    run.logger.stage_end("SEGMENTS", dataset=cfg.dataset)
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
