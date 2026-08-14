"""Protocol tests: determinism preflight, test-lock enforcement, MC budget,
and the notebook."""

from __future__ import annotations

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from shaper.schedules import configure_determinism  # noqa: E402


def test_preflight_determinism_configuration():
    out = configure_determinism()
    assert isinstance(out["deterministic_mode"], bool)
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.deterministic is True
    if out["deterministic_mode"]:
        assert out["exception"] is None
    else:
        # recorded, never silently suppressed
        assert out["exception"] is not None
        assert "recorded" in out["detail"] or "nondeterministic" in out["detail"]


def test_preflight_exception_not_suppressed(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("deterministic kernel unsupported (injected)")

    monkeypatch.setattr(torch, "use_deterministic_algorithms", boom)
    out = configure_determinism()
    assert out["deterministic_mode"] is False
    assert "deterministic kernel unsupported" in out["exception"]


def test_test_lock_never_selected_on():
    """SHAPER-Select ordering is a pure function of V_game Shapley values;
    test metrics are structurally absent from the decision path."""
    import inspect

    from shaper.adaptive import select_decision

    src = inspect.getsource(select_decision)
    assert "test" not in src.lower().replace("note", "") or "never selected" in src
    # activation + selection consume only (phi, delta_phi, players)
    args = inspect.signature(select_decision).parameters
    assert set(args) == {"phi_mean", "delta_phi", "players", "activation_active"}


def test_mc_budget_and_no_exact_interactions():
    from shaper.monte_carlo import MAX_UNIQUE_MODELS_PER_SEED, required_coalitions, sample_permutation
    from shaper.interactions import grabisch_roubens_interaction

    assert MAX_UNIQUE_MODELS_PER_SEED == 10
    players = ("crop", "mask", "reorder", "dropout")
    for seed in (3001, 3002, 3003, 3004, 3005):
        pi = sample_permutation(seed, players)
        registry = required_coalitions(pi, players)
        assert len(registry) <= 10
    # exact interactions forbidden on an incomplete K=4 table
    partial = {c: 0.0 for c in registry.values()}
    with pytest.raises(ValueError, match="incomplete table"):
        grabisch_roubens_interaction(partial, players, "crop", "mask")


def test_notebook_exists_and_is_valid():
    import nbformat

    path = os.path.join(REPO_ROOT, "notebooks", "run_all.ipynb")
    assert os.path.isfile(path)
    nb = nbformat.read(path, as_version=4)
    assert nb.nbformat == 4
    joined = "\n".join(
        "".join(cell.get("source", "")) for cell in nb.cells if cell.get("cell_type") == "markdown"
    ).lower()
    for section in (
        "environment", "specification", "configuration", "compute estimate",
        "data", "preflight", "recipe calibration", "pilots", "archive",
        "game a", "game b", "nll", "k=4", "shapley", "loo", "rq3",
        "weight", "select", "controls", "final intervention test", "statistics",
        "tables", "figures", "compliance", "reproducibility",
    ):
        assert section in joined, f"notebook section missing: {section}"
    # the notebook must call package functions, not duplicate logic
    code = "\n".join("".join(cell.get("source", "")) for cell in nb.cells if cell.get("cell_type") == "code")
    for marker in ("run_all", "status()", "estimate_cost()", "resume()", "def stage("):
        assert marker in code, f"notebook missing helper: {marker}"
    # the FIRST code cell must bootstrap sys.path (regression: Jupyter does not
    # put the repo root on sys.path; `from scripts import run_all` failed with
    # ModuleNotFoundError before this bootstrap existed)
    first_code = next(c for c in nb.cells if c.cell_type == "code")
    for marker in ("REPO_ROOT", "sys.path", "_is_repo_root"):
        assert marker in first_code.source, f"notebook bootstrap missing: {marker}"


def test_notebook_first_cell_imports_from_repo_cwds(tmp_path):
    """Execute the notebook's first cell in a subprocess launched from the
    notebooks/ directory and from a sibling directory: `from scripts import
    run_all` and the shaper imports must succeed (the user-reported failure
    mode)."""
    import nbformat
    import subprocess

    path = os.path.join(REPO_ROOT, "notebooks", "run_all.ipynb")
    nb = nbformat.read(path, as_version=4)
    first_code = next(c for c in nb.cells if c.cell_type == "code").source
    script = (
        first_code
        + "\nimport shaper\nfrom scripts import run_all\n"
        + "from shaper.config import load_run_config\n"
        + "from shaper.provenance import environment_record\n"
        + "from shaper.schedules import configure_determinism\n"
        + "from shaper.provenance import file_hash\n"
        + "file_hash(shaper.PROTOCOL_SPEC)\nfile_hash(shaper.PAPER_SPEC)\n"
        + "print('IMPORTS OK')\n"
    )
    for cwd in (
        os.path.join(REPO_ROOT, "notebooks"),
        os.path.join(REPO_ROOT, "docs"),
        os.path.dirname(REPO_ROOT),  # repo's parent directory
    ):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300,
        )
        assert result.returncode == 0, f"cwd={cwd}: {result.stderr}"
        assert "IMPORTS OK" in result.stdout


def test_structured_logging_canonical_schema(tmp_path):
    from shaper.logging_utils import StructuredLogger

    logger = StructuredLogger(str(tmp_path), run_id="r")
    logger.log(stage="PREFLIGHT", event="cache_miss", dataset="synthetic",
               seed=2001, coalition=["mask"], policy="game_a", status="info")
    with open(os.path.join(str(tmp_path), "events.jsonl")) as fh:
        rec = json.loads(fh.readline())
    for field in ("timestamp", "run_id", "stage", "dataset", "seed",
                  "coalition", "policy", "event", "status"):
        assert field in rec


def test_stage_registry_and_safe_cli():
    from shaper import STAGES

    expected = [
        "PREFLIGHT", "DATA_BUILD", "DATA_VALIDATE", "RECIPE_CALIBRATION",
        "PILOT_1001_1002", "PILOT_VALIDATION", "PILOT_AMENDMENT", "ARCHIVE_FREEZE",
        "PRIMARY_GAME_A", "SECONDARY_GAME_B", "SEVERITY_DIAGNOSTICS", "K4_BEAUTY_MC",
        "SHAPLEY", "LOO", "INTERACTIONS", "SEGMENTS", "POWER",
        "WEIGHT_CALIBRATION", "SELECT_CALIBRATION", "BASELINE_CONTROLS",
        "FINAL_INTERVENTION_TEST", "REPORT", "COMPLIANCE_AUDIT",
    ]
    assert STAGES == expected
