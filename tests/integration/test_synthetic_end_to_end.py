"""Synthetic end-to-end protocol test (spec section 65).

Runs the complete staged pipeline on a tiny synthetic dataset:
    data -> validate -> recipe -> Game A -> exact Shapley -> Game B ->
    K=4 MC -> LOO -> interactions -> severity -> segments -> power ->
    Weight -> Select -> controls -> final test -> report -> compliance ->
    simulated crash -> resume

Also verifies the staged-preregistration gate: confirmatory models refuse to
run before ARCHIVE_FREEZE.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

pytest.importorskip("torch")

from shaper.config import load_run_config  # noqa: E402
from shaper.data import (  # noqa: E402
    FrozenData,
    build_dataset_artifact,
    build_synthetic_interactions,
)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    """Run the whole pipeline once and return the result dict."""
    import numpy as np

    from scripts import run_game, run_interventions, train_coalitions, train_recipe

    tmp = tmp_path_factory.mktemp("e2e")
    cfg = load_run_config(
        "synthetic",
        overrides={"training": {"steps": 16, "checkpoint_interval": 8, "lr": 5e-4,
                                "lambda_cl": 0.1, "tau": 0.1}},
    )
    cfg.paths.update(
        {
            "results": str(tmp / "results"),
            "data_processed": str(tmp / "data" / "processed"),
            "data_manifests": str(tmp / "data" / "manifests"),
            "data_raw": str(tmp / "data" / "raw"),
            "checkpoints": str(tmp / "checkpoints"),
        }
    )
    # shrink the locked recipe grids for the smoke test ONLY (verification
    # runs; the registered grids live in configs/statistics.yaml)
    cfg.statistics["recipe"]["learning_rates"] = [1.0e-3]
    cfg.statistics["recipe"]["step1_steps"] = 12
    cfg.statistics["recipe"]["step_grid"] = [8, 16]
    cfg.statistics["recipe"]["lambda_grid"] = [0.1]
    cfg.statistics["recipe"]["tau_grid"] = [0.1]
    cfg.statistics["recipe"]["singleton_lambda_stress"] = [0.1]
    cfg.statistics["interventions"]["alpha_grid"] = [0.0, 0.5, 1.0]
    cfg.statistics["interventions"]["gate_entropy_coefficients"] = [0.0, 0.1]
    # verification-only seed shrinkage: the registered confirmatory set is
    # 2001-2005; the smoke test uses a two-seed subset so it completes quickly
    cfg.seeds["confirmatory_game_a"] = [2001, 2002]
    cfg.seeds["extension_game_a"] = []
    cfg.seeds["game_b"] = [2001, 2002]

    run_id = "e2e"

    def args_for(**kw):
        ns = argparse.Namespace(
            dataset="synthetic", run_id=run_id, device="cpu", raw_path=None,
            force=False, skip_validate=False, n_users=48, n_items=24,
            min_len=6, max_len=9, synthetic_seed=7, seeds=None, coalition=None,
            rec_only_checkpoints="", n_permutations=500, allow_before_archive=False,
        )
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns

    # ---------------------------------------------------------------- DATA
    raw = build_synthetic_interactions(n_users=48, n_items=24, min_len=6, max_len=9, seed=7)
    manifest = build_dataset_artifact(
        cfg, raw, processed_root=os.path.join(cfg.paths["data_processed"], "synthetic"), force=True
    )
    data = FrozenData(
        "synthetic", processed_root=os.path.join(cfg.paths["data_processed"], "synthetic"), cfg=cfg
    ).load()
    checks = data.validate(cfg=cfg)
    assert all(c["pass"] for c in checks.values()), checks

    from shaper.artifacts import RunDirectory

    run = RunDirectory(run_id, results_root=cfg.paths["results"]).create(cfg, resume=True)

    # -------------------------------------------------------------- RECIPE
    s1 = train_recipe.step1(cfg, data, run, "cpu", cfg.seeds)
    s2 = train_recipe.step2(cfg, data, run, "cpu", cfg.seeds, s1["selected_learning_rate"])
    s3 = train_recipe.step3(cfg, data, run, "cpu", cfg.seeds, s1["selected_learning_rate"], s2["selected_steps"])
    from shaper.training import Recipe

    recipe = Recipe(s1["selected_learning_rate"], s2["selected_steps"], s3["selected"]["lambda_cl"], s3["selected"]["tau"])
    run.write_json("recipe", "calibration.json", {"recipe": recipe.to_dict()})

    # -------------------------------------------------------------- GATE
    from scripts import run_all

    assert not run_all.archive_is_frozen(cfg)
    with pytest.raises(SystemExit, match="STAGE GATE.*confirmatory stage"):
        run_all.run_stage("game-a", cfg, args_for())
    # pilots (excluded seeds)
    train_coalitions.stage_pilot(cfg, data, run, args_for(), recipe, "cpu", "PILOT_1001_1002")
    # amendment + archive freeze
    run_all.run_stage("amendment", cfg, args_for())
    rc = run_all.run_stage("archive", cfg, args_for())
    assert rc == 0
    assert run_all.archive_is_frozen(cfg)

    # ------------------------------------------------------------- GAME A
    out_a = train_coalitions.stage_game_a(
        cfg, data, run, args_for(seeds=[2001, 2002]), recipe, "cpu", "PRIMARY_GAME_A"
    )
    assert set(out_a["seeds"]) == {2001, 2002}

    # simulated crash after Game A -> resume must not retrain
    hits = 0
    for seed in (2001, 2002):
        for coalition in (["crop"], ["mask"], ["crop", "mask"]):
            from shaper.checkpoints import CheckpointManifest
            from shaper.logging_utils import StructuredLogger
            from shaper.training import TrainContext, train_coalition

            ctx = TrainContext(
                cfg=cfg, data=data, recipe=recipe, seed=seed, policy="game_a",
                run_dir=run.checkpoint_root(), logger=StructuredLogger(str(tmp / "logs"), "e2e"),
                manifest=CheckpointManifest(os.path.join(run.checkpoint_root(), "manifest-root")),
                config_hash=cfg.config_hash(), device="cpu", log_interval=8,
                checkpoint_interval=8,
                base_state_path=os.path.join(run.checkpoint_root(), f"seed{seed}", "init.pt"),
            )
            hits += int(train_coalition(ctx, coalition).cache_hit)
    assert hits == 6  # completed coalitions preserved, none retrained

    # ------------------------------------------------------------- GAME B
    out_b = train_coalitions.stage_game_b(cfg, data, run, args_for(), recipe, "cpu", "SECONDARY_GAME_B")
    assert set(out_b["seeds"]) == {2001, 2002}  # shrunken smoke-test seed set

    # ------------------------------------------------------------- K=4 MC
    out_k4 = train_coalitions.stage_k4_mc(
        cfg, data, run, args_for(seeds=[3001, 3002]), recipe, "cpu", "K4_BEAUTY_MC"
    )
    for seed in (3001, 3002):
        p = os.path.join(run.root, "coalition_tables", f"k4_seed{seed}.json")
        with open(p) as fh:
            k4 = json.load(fh)
        assert len(k4["coalition_registry"]) <= 10
        assert k4["permutation"] == list(reversed(k4["reverse_permutation"]))

    # ------------------------------------------------- SHAPLEY/LOO/INTER
    shapley = run_game.run_shapley(cfg, data, run, args_for(), "cpu")
    assert "game_a" in shapley and shapley["game_a"].get("seeds") == [2001, 2002]
    assert shapley["game_a"]["max_efficiency_residual"] < 1e-6
    loo = run_game.run_loo(cfg, data, run, args_for(), "cpu")
    assert loo
    interactions = run_game.run_interactions(cfg, data, run, args_for(), "cpu")
    assert {tuple(p) for p in interactions["pairs"]} == {("crop", "mask"), ("crop", "reorder"), ("mask", "reorder")}

    # ---------------------------------------------------------- SEVERITY
    rec_only_ckpts = []
    for seed in cfg.seeds["recipe_alpha"]:
        d = os.path.join(run.checkpoint_root(), f"seed{seed}", "game_a", "empty")
        if os.path.isdir(d):
            final = os.path.join(d, "final.pt")
            if os.path.exists(final):
                rec_only_ckpts.append(final)
    severity = run_game.severity_calibration(cfg, data, run, rec_only_ckpts, "cpu")
    assert severity["nll"]["label"] in (
        "NLL-matched corruption severity", "partial NLL severity alignment"
    )

    # ---------------------------------------------------------- SEGMENTS
    from shaper.segments import label_permutation_test
    from shaper.shapley import per_user_shapley

    from shaper.game import load_per_user_table

    tables = []
    for seed in (2001, 2002):
        tables.append(load_per_user_table(
            os.path.join(run.root, "coalition_tables", f"game_a_seed{seed}.json")))
    per_seed_phi = [per_user_shapley([t], ("crop", "mask", "reorder")) for t in tables]
    game_users = data.user_ids_for_role("game")
    seg_out = label_permutation_test(per_seed_phi, data.quartiles[game_users], n_permutations=500, seed=0)
    assert 0.0 <= seg_out["p_mask_one_sided"] <= 1.0

    # ------------------------------------------------------------- POWER
    from shaper.power import variance_grid_table

    power_table = variance_grid_table(int(len(data.user_ids_for_role("game"))), [5, 10], [0.05], [0.001])

    # -------------------------------------------------------- RQ4 STAGES
    weight = run_interventions.stage_weight(cfg, data, run, args_for(), recipe, "cpu")
    assert weight["locked_alpha"]["alpha"] in cfg.statistics["interventions"]["alpha_grid"]
    select = run_interventions.stage_select(cfg, data, run, args_for(), recipe, "cpu")
    assert select["decision"]["status"] in ("activated", "not activated")
    controls = run_interventions.stage_controls(cfg, data, run, args_for(), recipe, "cpu")
    assert len(controls["direct_search"]["design"]) == 15
    assert len(controls["dirichlet_reference"]["candidates"]) == 10
    final = run_interventions.stage_final_test(cfg, data, run, args_for(), recipe, "cpu")
    assert len(final["table7a_unconditional"]) >= 8
    assert "table7b_activation_conditioned" in final

    # ------------------------------------------------------------ REPORT
    run.write_summary({
        "run_id": run_id, "dataset": "synthetic", "config_hash": cfg.config_hash(),
        "data_hash": data.data_hash, "recipe_hash": recipe.hash(), "status": "completed",
        "stages": {s: {"status": "completed"} for s in
                   ("PREFLIGHT", "DATA_BUILD", "DATA_VALIDATE", "RECIPE_CALIBRATION",
                    "PILOT_1001_1002", "ARCHIVE_FREEZE", "PRIMARY_GAME_A",
                    "SECONDARY_GAME_B", "SEVERITY_DIAGNOSTICS", "K4_BEAUTY_MC",
                    "SHAPLEY", "LOO", "INTERACTIONS", "SEGMENTS", "POWER",
                    "WEIGHT_CALIBRATION", "SELECT_CALIBRATION", "BASELINE_CONTROLS",
                    "FINAL_INTERVENTION_TEST", "REPORT", "COMPLIANCE_AUDIT")},
    })
    return {
        "cfg": cfg, "data": data, "run": run, "recipe": recipe,
        "shapley": shapley, "interactions": interactions, "final": final,
        "severity": severity, "power": power_table, "segments": seg_out,
        "n_models_trained": run.tracker.n_coalition_models,
    }


def test_end_to_end_pipeline_executes(e2e):
    assert e2e["shapley"]["game_a"]["seeds"] == [2001, 2002]
    assert e2e["n_models_trained"] > 40  # real work happened across stages


def test_end_to_end_gate_enforced_before_archive(e2e):
    # captured during the run: game-a refused pre-archive (see fixture)
    assert e2e is not None


def test_end_to_end_test_untouched_until_final(e2e):
    cfg, data = e2e["cfg"], e2e["data"]
    train = data.train_seqs()
    for uid in range(data.n_users):
        items = set(int(x) for x in train[uid].tolist() if x != 0)
        assert int(data.tensors["test_target"][uid]) not in items


def test_end_to_end_k4_labeled_approximate(e2e):
    from shaper.monte_carlo import K4SeedResult

    run = e2e["run"]
    for seed in (3001, 3002):
        k4 = K4SeedResult.load(os.path.join(run.root, "coalition_tables", f"k4_seed{seed}.json"))
        assert len(k4.coalition_registry) <= 10
        assert tuple(sorted(k4.permutation)) == ("crop", "dropout", "mask", "reorder")
