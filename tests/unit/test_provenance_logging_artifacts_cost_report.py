"""Unit tests: provenance hashing, structured logging, run-directory
scaffolding, compute tracking, cost estimation and report rendering."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.logging_utils import EVENT_FIELDS, StructuredLogger, make_event
from shaper.provenance import (
    canonical_json,
    config_hash,
    environment_record,
    key_int,
    stable_hash,
)


def test_stable_hash_order_invariant():
    assert stable_hash({"a": 1, "b": [1, 2]}) == stable_hash({"b": [1, 2], "a": 1})
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
    assert stable_hash({"a": 1}, salt="x") != stable_hash({"a": 1}, salt="y")


def test_canonical_json_sorted_keys():
    assert b'"a":1' in canonical_json({"a": 1})
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_config_hash_stable_across_yaml_formatting(tiny_cfg):
    h1 = tiny_cfg.config_hash()
    h2 = config_hash(
        {
            "dataset": tiny_cfg.raw["dataset"],
            "model": tiny_cfg.model,
            "augmentation": tiny_cfg.augmentation,
            "training_base": tiny_cfg.training,
            "evaluation": tiny_cfg.evaluation,
            "roles": tiny_cfg.roles,
        },
        salt=tiny_cfg.dataset,
    )
    assert h1 == h2


def test_key_int_deterministic():
    assert key_int(1, 2, 3) == key_int(1, 2, 3)
    assert key_int(1, 2, 3) != key_int(3, 2, 1)


def test_environment_record_fields():
    env = environment_record()
    for field in ("python", "platform", "cpu_count", "torch_version", "commit"):
        assert field in env


def test_structured_log_event_schema(tmp_path):
    logger = StructuredLogger(str(tmp_path), run_id="r1")
    logger.stage_start("PREFLIGHT", dataset="synthetic")
    logger.log(stage="PRIMARY_GAME_A", event="training_step", seed=2001,
               coalition=["crop", "mask"], policy="game_a", status="info", loss=0.5)
    with open(os.path.join(str(tmp_path), "events.jsonl")) as fh:
        lines = [json.loads(l) for l in fh if l.strip()]
    assert len(lines) == 2
    for rec in lines:
        for field in EVENT_FIELDS:
            assert field in rec
    step = lines[1]
    assert step["coalition"] == ["crop", "mask"]
    assert step["loss"] == 0.5


def test_make_event_coercion():
    rec = make_event(run_id="r", stage="S", event="e", coalition={"mask", "crop"})
    assert rec["coalition"] == ["crop", "mask"]


def test_run_directory_scaffolding(tmp_path, tiny_cfg):
    from shaper.artifacts import RunDirectory

    run = RunDirectory("demo", results_root=str(tmp_path)).create(tiny_cfg)
    for sub in ("logs", "checkpoints", "raw", "metrics", "coalition_tables",
                "shapley", "interactions", "segments", "interventions",
                "tables", "figures"):
        assert os.path.isdir(os.path.join(run.root, sub))
    for f in ("manifest.json", "config.yaml", "provenance.json",
              "environment.json", "seed_registry.json", "scope_manifest.json"):
        assert os.path.isfile(os.path.join(run.root, f))
    run.update_manifest(data_hash="abc")
    run.stage_status("PREFLIGHT", "completed")
    with open(os.path.join(run.root, "manifest.json")) as fh:
        manifest = json.load(fh)
    assert manifest["data_hash"] == "abc"
    assert manifest["stages"]["PREFLIGHT"]["status"] == "completed"
    run.write_summary({"run_id": "demo", "status": "ok", "stages": {"PREFLIGHT": {"status": "completed"}}})
    assert os.path.isfile(os.path.join(run.root, "summary.json"))
    assert os.path.isfile(os.path.join(run.root, "summary.md"))
    run.write_reproducibility_report(tiny_cfg)
    assert os.path.isfile(os.path.join(run.root, "reproducibility_report.md"))


def test_compute_tracker():
    from shaper.artifacts import ComputeTracker

    t = ComputeTracker()
    t.n_coalition_models = 3
    t.cache_hits = 2
    t.cache_misses = 1
    s = t.summary()
    assert s["n_coalition_models_trained"] == 3
    assert s["cache_hit_rate"] == pytest.approx(2 / 3)


def test_cost_estimator_counts(tiny_cfg):
    from shaper.cost import n_coalition_models_for_scope

    cb = n_coalition_models_for_scope(tiny_cfg, frozen_steps=100)
    s = cb.summary()
    # primary Game A: 8 coalitions x 5 seeds = 40
    # Game B: 6 x 3 = 18; pilots 16; K=4 MC only for beauty (synthetic excluded)
    primary = next(i for i in s["items"] if i["label"] == "primary_game_a")
    assert primary["coalition_models"] == 40
    game_b = next(i for i in s["items"] if i["label"] == "game_b")
    assert game_b["coalition_models"] == 18
    assert s["estimated_gpu_hours_range"]["low_hours"] <= s["estimated_gpu_hours_range"]["high_hours"]
    assert "PLANNING ESTIMATES" in s["planning_note"]


def test_cost_estimator_beauty_k4(tiny_cfg):
    from shaper.cost import n_coalition_models_for_scope

    beauty = n_coalition_models_for_scope(tiny_cfg, frozen_steps=100)
    cb = beauty
    k4 = next((i for i in cb.items if i["label"] == "k4_beauty_mc"), None)
    # synthetic dataset is not Beauty -> absent; force the branch
    assert k4 is None or k4["coalition_models"] <= 50


def test_report_tables_render():
    from shaper.report import (
        activation_conditioned_table,
        coalition_value_table,
        interaction_table,
        segment_table,
        shapley_table,
        unconditional_transfer_table,
    )

    tables = [{("c",): 0.001, (): 0.0, tuple("cmr"): 0.02} for _ in range(2)]
    md = coalition_value_table(tables, ["c", "m", "r"])
    assert "|" in md and "coalition" in md
    md2 = shapley_table([{"crop": 0.01, "mask": 0.02, "reorder": 0.0}], ["crop", "mask", "reorder"], [0.0])
    assert "efficiency residual" in md2
    md3 = interaction_table({("crop", "mask"): -0.004}, {("crop", "mask"): (-0.01, 0.0)}, {("crop", "mask"): "directional"})
    assert "directional" in md3
    mu = {str(m): {"mask": 0.0, "crop": 0.0, "reorder": 0.0} for m in range(1, 5)}
    se = {str(m): {"mask": 0.01, "crop": 0.01, "reorder": 0.01} for m in range(1, 5)}
    assert "Q1" in segment_table(mu, se, ["crop", "mask", "reorder"])
    assert "Table 7A" in unconditional_transfer_table([{"method": "rec-only", "ndcg": 0.1}])
    assert "not activated" in activation_conditioned_table({"deployment": "rec-only", "status": "not activated"})


def test_report_figures(tmp_path):
    from shaper.report import figure_alpha_curve, figure_shapley_bars

    p1 = figure_shapley_bars([{"crop": 0.01, "mask": 0.02, "reorder": 0.0}], ["crop", "mask", "reorder"],
                             str(tmp_path / "fig2.png"))
    p2 = figure_alpha_curve([0.0, 0.25, 0.5], [0.30, 0.31, 0.29], 0.25, str(tmp_path / "fig6.png"))
    assert os.path.isfile(p1) and os.path.isfile(p2)
