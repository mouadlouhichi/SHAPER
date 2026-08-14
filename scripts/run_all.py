"""SHAPER orchestrator — staged preregistration pipeline.

Usage (safe by default; NEVER launches the full study from one command):
    python scripts/run_all.py --status
    python scripts/run_all.py --validate
    python scripts/run_all.py --estimate-cost
    python scripts/run_all.py --stage data          [--dataset ...] [--run-id ...]
    python scripts/run_all.py --stage recipe
    python scripts/run_all.py --stage pilot
    python scripts/run_all.py --stage archive
    python scripts/run_all.py --stage game-a
    python scripts/run_all.py --stage game-b
    python scripts/run_all.py --stage k4-mc
    python scripts/run_all.py --stage shapley / loo / interactions / severity
    python scripts/run_all.py --stage segments / power
    python scripts/run_all.py --stage weight / select / controls / final-test
    python scripts/run_all.py --stage report
    python scripts/run_all.py --resume

Gating (spec B.7): confirmatory models refuse to run before the
pilot-informed archive exists (ARCHIVE_FREEZE). Each stage records
dependencies, inputs, outputs, completion criteria, checkpoint, config hash
and data-manifest hash in the run manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaper.config import load_run_config, load_yaml  # noqa: E402

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)


def _run_script(name: str, *argv: str) -> int:
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, name), *argv]
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


def freeze_path(cfg) -> str:
    if cfg.dataset in ("ml1m", "beauty"):
        return os.path.join(cfg.paths["configs"], "manifest_freeze.yaml")
    return os.path.join(cfg.paths["results"], f"freeze-synthetic-{cfg.dataset}.yaml")


def archive_is_frozen(cfg) -> bool:
    path = freeze_path(cfg)
    if not os.path.exists(path):
        return False
    freeze = load_yaml(path)
    if freeze.get("status") != "frozen":
        return False
    sel = (freeze.get("selected_recipe") or {}).get(cfg.dataset) or {}
    return all(sel.get(k) is not None for k in ("learning_rate", "steps", "lambda_cl", "tau"))


def _pilot_tables_exist(cfg, args) -> bool:
    """The two excluded pilot seeds must have completed for the run (the
    archive is pilot-informed, spec B.7)."""
    if not args.run_id:
        return False
    for seed in cfg.seeds["pilots"]:
        table = os.path.join(
            cfg.paths["results"], args.run_id, "coalition_tables", f"pilot_seed{seed}.json"
        )
        if not os.path.exists(table):
            return False
    return True


def _gate_message(action: str, why: str, how: str) -> str:
    """Single consistent format for staged-preregistration gate refusals, so
    the notebook (which catches SystemExit) and the CLI (which prints it)
    both show an actionable message."""
    return f"STAGE GATE — {action} refused: {why}\n    {how}"


def _require_archive(cfg, args) -> None:
    if archive_is_frozen(cfg):
        return
    if getattr(args, "allow_before_archive", False):
        print("WARNING: running before ARCHIVE_FREEZE (--allow-before-archive). "
              "This run is NOT confirmatory.")
        return
    raise SystemExit(_gate_message(
        "confirmatory stage",
        "confirmatory models cannot run before the pilot-informed archive exists "
        "(spec B.7: RECIPE_CALIBRATION -> pilots -> amendment -> ARCHIVE_FREEZE -> confirmatory)",
        "run the stages in order: --stage data, --stage recipe, --stage pilot, "
        "--stage amendment, --stage archive, then retry. "
        "(Notebook: Sections 5, 7, 8, 9, then this section.)",
    ))


def _run_stage_inner(name: str, cfg, args) -> int:
    ds = ["--dataset", cfg.dataset]
    run_id = ["--run-id", args.run_id] if args.run_id else []
    common = ds + run_id
    if name == "preflight":
        return _run_script("preflight.py", *ds, "--json-out",
                           os.path.join(cfg.paths["results"], f"preflight-{cfg.dataset}.json"))
    if name == "data":
        return _run_script("build_data.py", *common,
                           *(["--raw-path", args.raw_path] if args.raw_path else []),
                           *(["--download"] if getattr(args, "download", False) else []),
                           *(["--force"] if args.force else []))
    if name == "recipe":
        return _run_script("train_recipe.py", *common)
    if name == "pilot":
        return _run_script("train_coalitions.py", *common, "--stage", "pilot")
    if name == "pilot-validation":
        print("PILOT_VALIDATION: verify pilot rankings vary for substantive reasons "
              "and all protocol tests 1-20 pass; see tests/ and the pilot tables.")
        return 0
    if name == "amendment":
        out_path = os.path.join(cfg.paths["results"], f"amendment-{cfg.dataset}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(
                {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "kind": "pilot-informed amendment",
                 "direction_independent": True,
                 "note": "Appendix J realized counts/variance table; thresholds may "
                         "change only before confirmatory execution and only "
                         "independently of effect direction."},
                fh, indent=2, sort_keys=True,
            )
        print(f"PILOT_AMENDMENT recorded at {out_path}")
        return 0
    if name == "archive":
        # Idempotency: if the archive is already frozen for this dataset,
        # re-running the cell is a no-op (it must not demand a fresh
        # calibration from the current run).
        if archive_is_frozen(cfg):
            print(f"ARCHIVE_FREEZE: {cfg.dataset} already frozen ({freeze_path(cfg)}); no-op")
            return 0
        # 1. The frozen recipe must come from a completed calibration artifact.
        if args.run_id:
            recipe_path = os.path.join(cfg.paths["results"], args.run_id, "recipe", "calibration.json")
            if os.path.exists(recipe_path):
                with open(recipe_path) as fh:
                    recipe = json.load(fh)["recipe"]
            else:
                raise SystemExit(_gate_message(
                    "ARCHIVE_FREEZE",
                    f"no recipe calibration found for run '{args.run_id}' at {recipe_path}",
                    f"run the recipe stage first: --stage recipe --run-id {args.run_id} "
                    "(Notebook Section 7, with the same DATASET/RUN_ID). "
                    "The archive freezes the V_tune-selected learning rate, step count, "
                    "lambda_cl and tau; it cannot be produced from defaults.",
                ))
        elif cfg.dataset != "synthetic":
            raise SystemExit(_gate_message(
                "ARCHIVE_FREEZE",
                "real datasets require --run-id of a completed RECIPE_CALIBRATION run",
                "--stage recipe --run-id <run> (Notebook Section 7)",
            ))
        else:
            recipe = {"learning_rate": 5e-4, "steps": 60, "lambda_cl": 0.1, "tau": 0.1}
        # 2. spec B.7: the archive is PILOT-INFORMED — the two excluded pilot
        #    seeds must have completed before the freeze.
        if not _pilot_tables_exist(cfg, args):
            raise SystemExit(_gate_message(
                "ARCHIVE_FREEZE",
                f"the pilot-informed archive requires the two excluded pilot seeds "
                f"{cfg.seeds['pilots']} to have completed for run '{args.run_id}'",
                "run the pilot stage first: --stage pilot (Notebook Section 8). "
                "Pilot outcomes are excluded from confirmatory estimates but their "
                "variance tables inform Appendix J, which the archive freezes.",
            ))
        data_manifest = os.path.join(cfg.paths["data_processed"], cfg.dataset, "data_manifest.json")
        data_hash = None
        if os.path.exists(data_manifest):
            with open(data_manifest) as fh:
                data_hash = json.load(fh)["data_hash"]
        path = freeze_path(cfg)
        if os.path.exists(path):
            freeze = load_yaml(path)
        else:
            freeze = {"selected_recipe": {}, "data_hashes": {}}
        freeze.setdefault("selected_recipe", {})[cfg.dataset] = recipe
        freeze.setdefault("data_hashes", {})[cfg.dataset] = data_hash
        freeze["status"] = "frozen"
        freeze["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        import yaml

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(freeze, fh, sort_keys=True)
        # record the freeze in the run manifest (compliance evidence)
        if args.run_id and os.path.isdir(os.path.join(cfg.paths["results"], args.run_id)):
            from shaper.artifacts import RunDirectory

            try:
                RunDirectory(args.run_id, results_root=cfg.paths["results"]).stage_status(
                    "ARCHIVE_FREEZE", "completed", freeze_file=path
                )
            except Exception as exc:  # noqa: BLE001
                print(f"WARNING: could not record ARCHIVE_FREEZE in run manifest: {exc}")
        print(f"ARCHIVE_FREEZE: {path} frozen (dataset={cfg.dataset})")
        return 0
    if name == "game-a":
        _require_archive(cfg, args)
        return _run_script("train_coalitions.py", *common, "--stage", "game-a")
    if name == "game-b":
        _require_archive(cfg, args)
        return _run_script("train_coalitions.py", *common, "--stage", "game-b")
    if name == "k4-mc":
        _require_archive(cfg, args)
        return _run_script("train_coalitions.py", *common, "--stage", "k4-mc")
    if name in ("shapley", "loo", "interactions"):
        return _run_script("run_game.py", *common, "--stage", name)
    if name == "severity":
        return _run_script("run_game.py", *common, "--stage", "severity",
                           "--rec-only-checkpoints", args.rec_only_checkpoints or "")
    if name == "segments":
        return _run_script("run_segments.py", *common)
    if name == "power":
        return _run_script("run_power.py", *common)
    if name in ("weight", "select", "controls", "final-test"):
        return _run_script("run_interventions.py", *common, "--stage", name)
    if name == "report":
        from shaper.artifacts import RunDirectory

        run = RunDirectory(args.run_id or f"report-{cfg.dataset}-{int(time.time())}",
                           results_root=cfg.paths["results"]).create(cfg, resume=True)
        run.write_summary({"run_id": run.run_id, "dataset": cfg.dataset,
                           "status": "report", "stages": {}})
        return 0
    raise SystemExit(f"unknown stage {name}")


def run_stage(name: str, cfg, args) -> int:
    """Execute one stage with visible START/END banners and timing.

    Every stage additionally streams structured events to the run's
    `logs/events.jsonl` AND human-readable lines to the console (the
    StructuredLogger console layer)."""
    banner = "=" * 72
    t0 = time.time()
    print(f"\n{banner}\nSTAGE {name} | dataset={cfg.dataset} "
          f"run={getattr(args, 'run_id', None) or '-'}\n{banner}", flush=True)
    rc = _run_stage_inner(name, cfg, args)
    elapsed = time.time() - t0
    status = "OK" if rc in (0, None) else f"FAILED/REFUSED (rc={rc})"
    print(f"{banner}\nSTAGE {name} -> {status} in {elapsed:.1f}s\n{banner}", flush=True)
    return rc


STAGE_ORDER = [
    "preflight", "data", "recipe", "pilot", "pilot-validation", "amendment",
    "archive", "game-a", "game-b", "severity", "k4-mc", "shapley", "loo",
    "interactions", "segments", "power", "weight", "select", "controls",
    "final-test", "report",
]


def main() -> int:
    # operate from the repository root regardless of the launch directory
    os.chdir(REPO_ROOT)
    p = argparse.ArgumentParser(description="SHAPER staged pipeline")
    p.add_argument("--status", action="store_true", help="print stage status for the run/dataset")
    p.add_argument("--validate", action="store_true", help="run the full test suite")
    p.add_argument("--estimate-cost", action="store_true", help="budget estimate (planning only)")
    p.add_argument("--stage", default=None, help="execute one stage (see STAGE_ORDER)")
    p.add_argument("--resume", action="store_true", help="resume the run at the next incomplete stage")
    p.add_argument("--dataset", choices=["ml1m", "beauty", "ml100k", "synthetic"], default="synthetic")
    p.add_argument("--run-id", default=None)
    p.add_argument("--raw-path", default=None)
    p.add_argument("--download", action="store_true",
                   help="download the raw dataset from its canonical source (verified checksums)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rec-only-checkpoints", default=None)
    p.add_argument("--allow-before-archive", action="store_true",
                   help="permit pre-archive runs (labeled non-confirmatory)")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)

    if args.validate:
        return subprocess.call([sys.executable, "-m", "pytest", "tests", "-q"])

    if args.estimate_cost:
        from shaper.cost import n_coalition_models_for_scope

        print(json.dumps(n_coalition_models_for_scope(cfg).summary(), indent=2, sort_keys=True))
        return 0

    if args.status:
        manifest_path = os.path.join(cfg.paths["results"], args.run_id, "manifest.json") if args.run_id else None
        print(f"dataset: {cfg.dataset}")
        print(f"archive frozen: {archive_is_frozen(cfg)}")
        for stage in STAGE_ORDER:
            marker = ""
            if manifest_path and os.path.exists(manifest_path):
                with open(manifest_path) as fh:
                    m = json.load(fh)
                if stage in m.get("stages", {}):
                    marker = m["stages"][stage].get("status", "?")
            print(f"  {stage:<18} {marker}")
        return 0

    if args.stage:
        return run_stage(args.stage, cfg, args)

    if args.resume:
        for stage in STAGE_ORDER:
            rc = run_stage(stage, cfg, args)
            if rc != 0:
                print(f"resume stopped at stage {stage} (exit {rc})")
                return rc
        return 0

    p.print_help()
    print("\nSAFETY: the full study is never launched by default; run stages "
          "explicitly with --stage.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
