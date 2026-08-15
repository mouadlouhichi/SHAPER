"""RQ4 stages: WEIGHT_CALIBRATION, SELECT_CALIBRATION, BASELINE_CONTROLS,
FINAL_INTERVENTION_TEST.

SHAPER-Weight: dataset-level mean exact Game-A Shapley -> positive transform
-> w(alpha) = (1-alpha)/K + alpha*q; alpha swept on V_select with the same
three prespecified recipe-calibration initializations; one alpha locked per
dataset; five final evaluation seeds. Locked activation rule decides
deployment; NOT_ACTIVATED -> rec-only fallback.

SHAPER-Select: rank by mean raw Game-A Shapley (tie order
crop<mask<reorder<dropout); no action if the two lowest differ by less than
delta_phi; else remove the lowest-Shapley view REUSING the enumerated
coalition model. Never selected on test metrics.

Controls: uniform (grand, reused), LOO-derived weights, learned dataset-level
gates, drop-lowest-LOO, random removal, 15-point simplex direct search, ten
Dirichlet reference candidates. All budgets are recorded.

Final test: locked decisions on ALL eligible users with the test prefix
(training history + validation item); repeated-target rate reported.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from shaper import MAIN_PLAYERS  # noqa: E402
from shaper.adaptive import (  # noqa: E402
    calibrate_alpha,
    mean_game_a_shapley,
    select_decision,
    shaper_weights,
    weight_activation_rule,
)
from shaper.baselines import (  # noqa: E402
    control_budget_record,
    dirichlet_reference_candidates,
    drop_lowest_loo,
    loo_derived_weights,
    random_removal,
    simplex_design_15,
)
from shaper.checkpoints import CheckpointManifest  # noqa: E402
from shaper.config import load_run_config  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.metrics import evaluate_model  # noqa: E402
from shaper.stats import seed_summary  # noqa: E402
from shaper.training import (  # noqa: E402
    DatasetLevelGates,
    TrainContext,
    train_coalition,
)


def load_game_a(run) -> Dict[str, Any]:
    path = os.path.join(run.root, "shapley", "game_a_shapley.json")
    if not os.path.exists(path):
        raise SystemExit("run SHAPLEY first (game_a_shapley.json missing)")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def grand_uplift_ci(run, cfg) -> Dict[str, Any]:
    from shaper.game import load_coalition_table

    vals = []
    for seed in cfg.seeds["confirmatory_game_a"]:
        path = os.path.join(run.root, "coalition_tables", f"game_a_seed{seed}.json")
        if not os.path.exists(path):
            continue
        records = load_coalition_table(path)
        values = {rec.coalition: rec.value_ndcg for rec in records}
        vals.append(values[tuple(MAIN_PLAYERS)])
    if not vals:
        raise SystemExit("no Game-A grand coalition values found")
    return seed_summary(vals)


def _ctx(cfg, data, run, seed, recipe, device, policy):
    manifest = CheckpointManifest(os.path.join(run.checkpoint_root(), "interventions"))
    return TrainContext(
        cfg=cfg, data=data, recipe=recipe, seed=seed, policy=policy,
        run_dir=run.checkpoint_root(), logger=run.logger, manifest=manifest,
        config_hash=cfg.config_hash(), device=device,
        log_interval=max(1, min(50, recipe.steps // 5)),
        checkpoint_interval=max(1, min(cfg.training["checkpoint_interval"], recipe.steps)),
        base_state_path=os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt"),
    )


def vselect_ndcg(cfg, data, run, result, device) -> float:
    res = evaluate_model(result.model, data.eval_inputs("select"), cfg.evaluation["k"], device=device)
    return res.ndcg


def test_metrics(cfg, data, model, device) -> Dict[str, float]:
    res = evaluate_model(model, data.test_inputs(), cfg.evaluation["k"], device=device)
    return {"ndcg": res.ndcg, "hr": res.hr, "mrr": res.mrr,
            "repeated_target_rate": res.repeated_target_rate}


def load_model_from_checkpoint(cfg, data, path, device, model_factory=None):
    from shaper.backbone import build_model
    from shaper.checkpoints import load_coalition_checkpoint

    payload = load_coalition_checkpoint(path)
    model = (model_factory or build_model)(cfg, data.n_items)
    model.load_state_dict(payload["model"])
    model.to(device)
    return model


def stage_weight(cfg, data, run, args, recipe, device):
    game_a = load_game_a(run)
    phi_mean = {p: game_a["mean_phi"][p] for p in MAIN_PLAYERS}
    uplift = grand_uplift_ci(run, cfg)
    activation = weight_activation_rule(uplift["ci"], game_a["per_seed_phi"], phi_mean)

    alpha_grid = cfg.statistics["interventions"]["alpha_grid"]
    calib_seeds = cfg.seeds["recipe_alpha"]  # same three recipe-calibration inits
    scores: Dict[float, List[float]] = {}
    for alpha in alpha_grid:
        weights = shaper_weights(phi_mean, alpha)
        vals = []
        for seed in calib_seeds:
            ctx = _ctx(cfg, data, run, seed, recipe, device, "weighted")
            result = train_coalition(ctx, MAIN_PLAYERS, weights=weights)
            run.tracker.n_coalition_models += int(not result.cache_hit)
            run.tracker.training_steps += result.steps if not result.cache_hit else 0
            vals.append(vselect_ndcg(cfg, data, run, result, device))
        scores[alpha] = vals
    alpha_scores = {a: float(np.mean(v)) for a, v in scores.items()}
    locked = calibrate_alpha(alpha_grid, alpha_scores)

    out = {
        "phi_mean": phi_mean,
        "target_q": {p: float(q) for p, q in __import__("shaper.adaptive", fromlist=["positive_transform"]).positive_transform(phi_mean).items()},
        "alpha_scores_vselect": {str(a): v for a, v in alpha_scores.items()},
        "per_init_scores": {str(a): v for a, v in scores.items()},
        "locked_alpha": locked,
        "activation": activation,
        "grand_uplift_ci": uplift["ci"],
        "weights": shaper_weights(phi_mean, locked["alpha"]),
    }
    run.write_json("interventions", "weight_calibration.json", out)
    return out


def stage_select(cfg, data, run, args, recipe, device):
    game_a = load_game_a(run)
    phi_mean = {p: game_a["mean_phi"][p] for p in MAIN_PLAYERS}
    uplift = grand_uplift_ci(run, cfg)
    activation = weight_activation_rule(uplift["ci"], game_a["per_seed_phi"], phi_mean)
    decision = select_decision(
        phi_mean, cfg.statistics["thresholds"]["delta_phi"],
        players=MAIN_PLAYERS, activation_active=activation["activated"],
    )
    out = {"phi_mean": phi_mean, "activation": activation,
           "delta_phi": cfg.statistics["thresholds"]["delta_phi"],
           "decision": decision}
    run.write_json("interventions", "select_calibration.json", out)
    return out


def stage_controls(cfg, data, run, args, recipe, device):
    game_a = load_game_a(run)
    loo = {p: float(np.mean([s[p] for s in game_a["loo_per_seed"]])) for p in MAIN_PLAYERS}
    calib_seed = cfg.seeds["recipe_alpha"][0]
    alpha_grid = cfg.statistics["interventions"]["alpha_grid"]
    budget = control_budget_record()
    out: Dict[str, Any] = {}

    # LOO-derived weights: same clipping + coarse alpha grid + final seeds
    loo_scores = {}
    for alpha in alpha_grid:
        weights = loo_derived_weights(loo, alpha)
        ctx = _ctx(cfg, data, run, calib_seed, recipe, device, "weighted")
        result = train_coalition(ctx, MAIN_PLAYERS, weights=weights)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        loo_scores[alpha] = vselect_ndcg(cfg, data, run, result, device)
    loo_locked = calibrate_alpha(alpha_grid, {a: v for a, v in loo_scores.items()})
    out["loo_weights"] = {"loo": loo, "alpha_scores": {str(a): v for a, v in loo_scores.items()},
                          "locked": loo_locked, "weights": loo_derived_weights(loo, loo_locked["alpha"])}

    # Learned dataset-level gates: entropy coefficient selected on V_select
    gate_scores = {}
    for ent in cfg.statistics["interventions"]["gate_entropy_coefficients"]:
        gate = DatasetLevelGates(len(MAIN_PLAYERS))
        ctx = _ctx(cfg, data, run, calib_seed, recipe, device, "gated")
        result = train_coalition(ctx, MAIN_PLAYERS, gate=gate, entropy_coef=ent)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        gate_scores[ent] = vselect_ndcg(cfg, data, run, result, device)
    best_ent = max(gate_scores, key=gate_scores.get)
    out["gates"] = {"entropy_scores": {str(e): v for e, v in gate_scores.items()},
                    "locked_entropy_coefficient": best_ent,
                    "note": "training-only gates; never used at inference"}

    # drop-lowest-LOO and random removal reuse enumerated coalition models
    out["drop_lowest_loo"] = drop_lowest_loo(MAIN_PLAYERS, loo)
    out["random_removal"] = random_removal(MAIN_PLAYERS, key_seed=cfg.seeds["recipe_alpha"][0])

    # 15-point simplex direct search on ONE declared calibration initialization
    design = simplex_design_15()
    search_scores = {}
    for i, weights in enumerate(design):
        ctx = _ctx(cfg, data, run, calib_seed, recipe, device, "weighted")
        result = train_coalition(ctx, MAIN_PLAYERS, weights=weights)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        search_scores[i] = vselect_ndcg(cfg, data, run, result, device)
    best_i = max(search_scores, key=search_scores.get)
    out["direct_search"] = {
        "label": "15-point, one-initialization search plus five-seed confirmation",
        "design": design, "scores": search_scores, "selected_index": best_i,
        "selected_weights": design[best_i], "budget": budget,
    }

    # Ten Dirichlet reference candidates on the same calibration initialization
    dirichlet = dirichlet_reference_candidates(seed=calib_seed)
    dir_scores = []
    for weights in dirichlet:
        ctx = _ctx(cfg, data, run, calib_seed, recipe, device, "weighted")
        result = train_coalition(ctx, MAIN_PLAYERS, weights=weights)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        dir_scores.append(vselect_ndcg(cfg, data, run, result, device))
    out["dirichlet_reference"] = {
        "candidates": dirichlet, "scores": dir_scores,
        "median": float(np.median(dir_scores)), "min": float(np.min(dir_scores)),
        "max": float(np.max(dir_scores)),
        "note": "V_select reference distribution, not ten test-tuned methods",
    }
    out["budget"] = budget
    run.write_json("interventions", "controls.json", out)
    return out


def _load_game_a_model(cfg, data, run, seed, coalition, device):
    path = os.path.join(run.checkpoint_root(), f"seed{seed}", "game_a",
                        "+".join(sorted(coalition)) if coalition else "empty", "final.pt")
    return load_model_from_checkpoint(cfg, data, path, device)


def stage_final_test(cfg, data, run, args, recipe, device):
    """Final test: locked decisions on all eligible users; Tables 7A + 7B."""
    weight_path = os.path.join(run.root, "interventions", "weight_calibration.json")
    select_path = os.path.join(run.root, "interventions", "select_calibration.json")
    controls_path = os.path.join(run.root, "interventions", "controls.json")
    if not (os.path.exists(weight_path) and os.path.exists(select_path)):
        raise SystemExit("calibrate Weight and Select before the final test")
    with open(weight_path) as fh:
        weight_cal = json.load(fh)
    with open(select_path) as fh:
        select_cal = json.load(fh)
    with open(controls_path) as fh:
        controls = json.load(fh)

    final_seeds = cfg.seeds["confirmatory_game_a"]
    rows = []

    def _mean_test(metrics_list):
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in ("ndcg", "hr", "mrr")}

    # rec-only (empty) and uniform (grand) — reused coalition models
    rec_only, uniform = [], []
    for seed in final_seeds:
        rec_only.append(test_metrics(cfg, data, _load_game_a_model(cfg, data, run, seed, [], device), device))
        uniform.append(test_metrics(cfg, data, _load_game_a_model(cfg, data, run, seed, MAIN_PLAYERS, device), device))
    rows.append({"method": "rec-only", **_mean_test(rec_only), "cost": "0 (reused)", "note": "empty coalition"})
    rows.append({"method": "uniform", **_mean_test(uniform), "cost": "0 (reused)", "note": "grand coalition (1/K weights)"})

    # forced Weight (locked alpha, final seeds)
    weights = {p: weight_cal["weights"][p] for p in MAIN_PLAYERS}
    forced_weight = []
    for seed in final_seeds:
        ctx = _ctx(cfg, data, run, seed, recipe, device, "weighted")
        result = train_coalition(ctx, MAIN_PLAYERS, weights=weights)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        forced_weight.append(test_metrics(cfg, data, result.model, device))
    rows.append({"method": "forced SHAPER-Weight", **_mean_test(forced_weight),
                 "cost": f"{len(final_seeds)} final-seed trainings", "note": "reported regardless of activation"})

    # forced lowest-Shapley removal (reused coalition)
    removed = select_cal["decision"].get("removed_view")
    forced_select = []
    for seed in final_seeds:
        coalition = [p for p in MAIN_PLAYERS if p != removed] if removed else list(MAIN_PLAYERS)
        forced_select.append(test_metrics(cfg, data, _load_game_a_model(cfg, data, run, seed, coalition, device), device))
    rows.append({"method": "forced lowest-Shapley removal", **_mean_test(forced_select),
                 "cost": "0 (reused)", "note": f"removed={removed}"})

    # LOO weights (locked alpha, final seeds)
    loo_weights = {p: controls["loo_weights"]["weights"][p] for p in MAIN_PLAYERS}
    loo_rows = []
    for seed in final_seeds:
        ctx = _ctx(cfg, data, run, seed, recipe, device, "weighted")
        result = train_coalition(ctx, MAIN_PLAYERS, weights=loo_weights)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        loo_rows.append(test_metrics(cfg, data, result.model, device))
    rows.append({"method": "LOO-derived weights", **_mean_test(loo_rows),
                 "cost": f"{len(final_seeds)} final-seed trainings", "note": "same clipping/alpha grid"})

    # learned gates (locked entropy coefficient, final seeds)
    ent = controls["gates"]["locked_entropy_coefficient"]
    gate_rows = []
    for seed in final_seeds:
        gate = DatasetLevelGates(len(MAIN_PLAYERS))
        ctx = _ctx(cfg, data, run, seed, recipe, device, "gated")
        result = train_coalition(ctx, MAIN_PLAYERS, gate=gate, entropy_coef=ent)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        gate_rows.append(test_metrics(cfg, data, result.model, device))
    rows.append({"method": "learned gates", **_mean_test(gate_rows),
                 "cost": f"{len(final_seeds)} final-seed trainings", "note": f"entropy coef {ent}"})

    # direct search (selected vector, final seeds)
    ds_weights = {p: controls["direct_search"]["selected_weights"][p] for p in MAIN_PLAYERS}
    ds_rows = []
    for seed in final_seeds:
        ctx = _ctx(cfg, data, run, seed, recipe, device, "weighted")
        result = train_coalition(ctx, MAIN_PLAYERS, weights=ds_weights)
        run.tracker.n_coalition_models += int(not result.cache_hit)
        ds_rows.append(test_metrics(cfg, data, result.model, device))
    rows.append({"method": "direct search (15-point)", **_mean_test(ds_rows),
                 "cost": "15 calibration + 5 confirmation trainings", "note": "one-initialization search"})

    # random removal (reused) and Dirichlet reference
    rand_removed = controls["random_removal"]["removed_view"]
    rand_rows = []
    for seed in final_seeds:
        coalition = [p for p in MAIN_PLAYERS if p != rand_removed]
        rand_rows.append(test_metrics(cfg, data, _load_game_a_model(cfg, data, run, seed, coalition, device), device))
    rows.append({"method": "random removal", **_mean_test(rand_rows), "cost": "0 (reused)",
                 "note": f"removed={rand_removed}"})
    rows.append({"method": "Dirichlet reference (median of 10)", "ndcg": controls["dirichlet_reference"]["median"],
                 "hr": None, "mrr": None, "cost": "10 calibration trainings",
                 "note": "V_select reference distribution"})

    # Table 7B: activation-conditioned deployment
    activated = weight_cal["activation"]["activated"]
    if not activated:
        row7b = {"deployment": "rec-only", **_mean_test(rec_only), "status": "not activated"}
    else:
        row7b = {"deployment": f"SHAPER-Weight (alpha={weight_cal['locked_alpha']['alpha']})",
                 **_mean_test(forced_weight), "status": "activated"}
        if select_cal["decision"].get("action") == "remove_lowest":
            row7b["note"] = f"Select: removed {removed}"

    # recommendation baselines (paper 4.2): GRU4Rec trained by the
    # BASELINE_CONTROLS stage, CL4SRec = uniform grand-coalition row reuse.
    rec_baselines = {}
    gru_rows = []
    from shaper.baseline_models import build_gru4rec

    for seed in final_seeds:
        path = os.path.join(run.checkpoint_root(), f"seed{seed}", "gru4rec_baseline",
                            "empty", "final.pt")
        if os.path.exists(path):
            gru_rows.append(test_metrics(
                cfg, data,
                load_model_from_checkpoint(cfg, data, path, device, model_factory=build_gru4rec),
                device,
            ))
    if gru_rows:
        rec_baselines["gru4rec"] = {
            **_mean_test(gru_rows),
            "cost": f"{len(gru_rows)} final-seed trainings (frozen recipe)",
            "note": "rec-only GRU baseline",
        }
    rec_baselines["cl4srec"] = {
        **_mean_test(uniform),
        "cost": "0 (grand-coalition reuse)",
        "note": "protocol-compatible CL4SRec reference = uniform three-view contrastive SASRec",
    }

    out = {
        "final_seeds": final_seeds,
        "recommendation_baselines": rec_baselines,
        "test_users": "all eligible users",
        "prefix": "training history + validation item",
        "table7a_unconditional": rows,
        "table7b_activation_conditioned": row7b,
        "note": "fallback is never counted as an adaptive success",
    }
    run.write_json("interventions", "final_test.json", out)
    return out


def main() -> int:
    from _cli import base_parser, frozen_data, resolve_device, resolve_recipe

    p = base_parser("SHAPER RQ4 interventions")
    p.add_argument("--stage", required=True,
                   choices=["weight", "select", "controls", "final-test"])
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    if cfg.raw.get("interventions_override"):
        cfg.statistics["interventions"].update(cfg.raw["interventions_override"])
    data = frozen_data(cfg)
    device = resolve_device(args)
    recipe = resolve_recipe(cfg)

    from shaper.artifacts import RunDirectory

    run_id = args.run_id or f"interventions-{cfg.dataset}-{int(time.time())}"
    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)
    stage_name = {"weight": "WEIGHT_CALIBRATION", "select": "SELECT_CALIBRATION",
                  "controls": "BASELINE_CONTROLS", "final-test": "FINAL_INTERVENTION_TEST"}[args.stage]
    run.logger.stage_start(stage_name, dataset=cfg.dataset)
    t0 = time.time()
    try:
        out = {"weight": stage_weight, "select": stage_select, "controls": stage_controls,
               "final-test": stage_final_test}[args.stage](cfg, data, run, args, recipe, device)
        run.stage_status(stage_name, "completed", wall_seconds=round(time.time() - t0, 2))
        print(json.dumps(out, indent=2, sort_keys=True, default=str))
    except Exception as exc:  # noqa: BLE001
        run.logger.stage_fail(stage_name, str(exc), dataset=cfg.dataset)
        run.stage_status(stage_name, "failed", error=str(exc))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
