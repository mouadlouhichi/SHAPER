"""Gate tests for the staged preregistration sequence (spec B.7).

ARCHIVE_FREEZE:
  - refuses when the run's recipe calibration is missing (the error the user
    hit in the notebook: "requires a recipe calibration for run ...")
  - refuses when the two excluded pilot seeds have not completed (the archive
    is PILOT-INFORMED)
  - freezes when both exist, and is IDEMPOTENT afterwards (re-running the
    notebook cell must not demand a fresh calibration)

Confirmatory stages refuse before the archive exists.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from scripts import run_all
from shaper.config import load_run_config


@pytest.fixture()
def gate_cfg(tmp_path):
    cfg = load_run_config("synthetic")
    cfg.paths.update(
        {
            "results": str(tmp_path / "results"),
            "data_processed": str(tmp_path / "data"),
        }
    )
    return cfg


def _args(run_id, **kw):
    ns = run_all.argparse.Namespace(
        dataset="synthetic", run_id=run_id, raw_path=None, force=False,
        skip_validate=False, n_users=48, n_items=24, min_len=6, max_len=9,
        synthetic_seed=7, seeds=None, coalition=None, rec_only_checkpoints="",
        n_permutations=10000, allow_before_archive=False,
    )
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def _write_recipe(cfg, run_id, recipe=None):
    recipe = recipe or {"learning_rate": 1e-3, "steps": 16, "lambda_cl": 0.1, "tau": 0.1}
    path = os.path.join(cfg.paths["results"], run_id, "recipe", "calibration.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"recipe": recipe}, fh)


def _write_pilots(cfg, run_id):
    for seed in cfg.seeds["pilots"]:
        path = os.path.join(cfg.paths["results"], run_id, "coalition_tables", f"pilot_seed{seed}.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump([], fh)


def test_archive_refuses_without_recipe(gate_cfg):
    """The exact failure mode reported from the notebook: archive without a
    recipe calibration must refuse with an actionable message."""
    with pytest.raises(SystemExit) as exc:
        run_all.run_stage("archive", gate_cfg, _args("run-1"))
    msg = str(exc.value)
    assert "STAGE GATE" in msg
    assert "recipe" in msg.lower()
    assert "recipe" in msg.lower() and "calibration.json" in msg
    assert "Section 7" in msg


def test_archive_refuses_without_pilots(gate_cfg):
    _write_recipe(gate_cfg, "run-1")
    with pytest.raises(SystemExit) as exc:
        run_all.run_stage("archive", gate_cfg, _args("run-1"))
    msg = str(exc.value)
    assert "pilot" in msg.lower()
    assert "Section 8" in msg


def test_archive_freezes_then_is_idempotent(gate_cfg):
    _write_recipe(gate_cfg, "run-1")
    _write_pilots(gate_cfg, "run-1")
    assert run_all.run_stage("archive", gate_cfg, _args("run-1")) == 0
    assert run_all.archive_is_frozen(gate_cfg)
    freeze = run_all.load_yaml(run_all.freeze_path(gate_cfg))
    assert freeze["selected_recipe"]["synthetic"]["learning_rate"] == 1e-3
    # idempotent: re-running with a DIFFERENT run id (no artifacts at all)
    # is a no-op success — the archive is already frozen for the dataset
    assert run_all.run_stage("archive", gate_cfg, _args("run-2")) == 0


def test_confirmatory_stage_refuses_before_archive(gate_cfg):
    with pytest.raises(SystemExit) as exc:
        run_all.run_stage("game-a", gate_cfg, _args("run-1"))
    assert "STAGE GATE" in str(exc.value)
    assert "archive" in str(exc.value).lower()
    # --allow-before-archive downgrades to a warning (labeled non-confirmatory)
    rc = run_all.run_stage("game-a", gate_cfg, _args("run-1", allow_before_archive=True))
    assert rc != 0  # the underlying training run still has nothing to do here


def test_notebook_gate_helper_catches_systemexit():
    """The notebook's stage() helper turns gate refusals into a readable
    'GATE:' line and returns the exit code instead of crashing the cell."""
    import nbformat as nbf

    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    nb = nbf.read(os.path.join(repo, "notebooks", "run_all.ipynb"), as_version=4)
    cfg_cell = next(
        c for c in nb.cells if c.cell_type == "code" and "def stage(name" in c.source
    )
    assert "except SystemExit" in cfg_cell.source
    assert "GATE:" in cfg_cell.source
