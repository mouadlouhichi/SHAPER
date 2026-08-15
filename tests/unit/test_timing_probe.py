"""Unit test: the timing probe measures and extrapolates on a tiny budget."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shaper.config import load_run_config
from shaper.data import build_dataset_artifact, build_synthetic_interactions


def _load_probe():
    spec = importlib.util.spec_from_file_location(
        "timing_probe",
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "scripts",
            "timing_probe.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_probe_runs_and_extrapolates(tmp_path, tiny_cfg):
    """Run the probe for 2 steps on the synthetic dataset with an isolated
    artifact root and assert the measured/extrapolated outputs."""
    # real frozen artifact in an isolated root (the probe verifies it exists)
    artifact_root = tmp_path / "data"
    raw = build_synthetic_interactions(n_users=48, n_items=24, min_len=6, max_len=9, seed=7)
    build_dataset_artifact(tiny_cfg, raw, processed_root=str(artifact_root / "synthetic"), force=True)

    probe = _load_probe()
    args = argparse.Namespace(
        dataset="synthetic", steps=2, device="cpu",
        out_dir=str(tmp_path / "timing"),
        processed_root=str(artifact_root),
        combine=False,
    )
    assert probe.run_probe(args) == 0

    record = json.load(open(str(tmp_path / "timing" / "timing-synthetic.json")))
    assert record["steps_probed"] == 2
    assert record["measured"]["grand (all views)"]["seconds_per_step"] > 0
    assert record["measured"]["empty (rec-only)"]["seconds_per_step"] > 0
    assert record["extrapolated_per_coalition"]["10000 steps"]["hours"] > 0
    assert record["projected_study_hours"]["coalition_models_in_scope"] > 0
    assert record["device"] == "cpu"
    det = record["determinism"]
    assert det["deterministic_mode"] is True or det["exception"] is not None
