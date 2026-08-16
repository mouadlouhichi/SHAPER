"""TIMING PROBE — measured per-step training cost on THIS machine.

    python scripts/timing_probe.py --dataset ml1m --steps 200

Trains one rec-only (empty) and one grand (all-views) coalition for
`--steps` optimizer steps with a provisional recipe, then reports:

  - seconds per optimizer step (measured, empty and grand),
  - extrapolated wall time per 10,000-step / 5,000-step coalition,
  - projected total for the full declared study scope (cost-estimator
    coalition-model counts x the measured per-coalition rate),
  - device info (cuda / mps / cpu) and the determinism-mode record.

Results are saved under `results/timing/timing-<dataset>.json` so the paper
can replace planning estimates with measured values. This is an engineering
measurement tool (like shaper.cost); it never feeds a confirmatory table.

Typical first use on a new machine (before the full study):

    python scripts/run_all.py --stage data --dataset ml1m   # frozen artifact
    python scripts/timing_probe.py --dataset ml1m --steps 200
    python scripts/run_all.py --stage data --dataset beauty
    python scripts/timing_probe.py --dataset beauty --steps 200
    python scripts/timing_probe.py --dataset ml1m --combine  # both datasets
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psutil  # noqa: E402
import torch  # noqa: E402

from shaper.checkpoints import CheckpointManifest  # noqa: E402
from shaper.config import REPO_ROOT, load_run_config  # noqa: E402
from shaper.cost import n_coalition_models_for_scope  # noqa: E402
from shaper.data import FrozenData  # noqa: E402
from shaper.logging_utils import StructuredLogger  # noqa: E402
from shaper.schedules import configure_determinism  # noqa: E402
from shaper.training import Recipe, TrainContext, train_coalition  # noqa: E402

DEFAULT_STEPS = 200
EXTRAPOLATION_STEPS = (5_000, 10_000)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_probe(args) -> int:
    """Probe entry point taking a parsed argparse.Namespace (also used by
    tests)."""
    out_dir = args.out_dir or os.path.join(REPO_ROOT, "results", "timing")
    os.makedirs(out_dir, exist_ok=True)

    if args.combine:
        rows = []
        for ds in ("ml1m", "beauty"):
            path = os.path.join(out_dir, f"timing-{ds}.json")
            if os.path.exists(path):
                with open(path, encoding="utf-8") as fh:
                    rows.append(json.load(fh))
        if not rows:
            raise SystemExit("no timing records yet; run the probe per dataset first")
        total_hours_low = sum(r["projected_study_hours"]["low"] for r in rows)
        total_hours_high = sum(r["projected_study_hours"]["high"] for r in rows)
        print(json.dumps({
            "combined_declared_scope": {
                "low_hours": round(total_hours_low, 1),
                "high_hours": round(total_hours_high, 1),
                "datasets": [r["dataset"] for r in rows],
                "note": "sequential single-process extrapolation; paper reports measured values",
            }
        }, indent=2))
        return 0

    cfg = load_run_config(args.dataset)
    if args.processed_root:
        cfg.paths["data_processed"] = args.processed_root
    processed = os.path.join(cfg.paths["data_processed"], cfg.dataset, "data_manifest.json")
    if not os.path.exists(processed):
        raise SystemExit(
            f"no frozen artifact for {cfg.dataset}; run the data stage first:\n"
            f"  python scripts/run_all.py --stage data --dataset {cfg.dataset}"
        )
    data = FrozenData(cfg.dataset, processed_root=os.path.join(cfg.paths["data_processed"], cfg.dataset)).load()

    device = args.device or pick_device()
    det = configure_determinism()
    recipe = Recipe(learning_rate=1.0e-3, steps=args.steps, lambda_cl=0.1, tau=0.1)
    probe_dir = os.path.join(out_dir, f"{cfg.dataset}-probe-{device}")
    shutil.rmtree(probe_dir, ignore_errors=True)  # always a fresh measurement

    manifest = CheckpointManifest(os.path.join(probe_dir, "manifest"))
    logger = StructuredLogger(os.path.join(probe_dir, "logs"), run_id="timing-probe")
    results = {}
    for coalition in ([], ["crop", "mask", "reorder"]):
        ctx = TrainContext(
            cfg=cfg, data=data, recipe=recipe, seed=2001, policy="game_a",
            run_dir=os.path.join(probe_dir, "ckpt"), logger=logger, manifest=manifest,
            config_hash=cfg.config_hash(), device=device,
            log_interval=max(1, args.steps // 5),
            checkpoint_interval=max(1, args.steps),
            base_state_path=os.path.join(probe_dir, "ckpt", "seed2001", "init.pt"),
        )
        t0 = time.time()
        result = train_coalition(ctx, coalition)
        wall = time.time() - t0
        label = "empty (rec-only)" if not coalition else "grand (all views)"
        results[label] = {
            "wall_seconds": round(wall, 2),
            "seconds_per_step": round(wall / args.steps, 4),
            "steps": result.steps,
            "final_grad_norm": round(result.final_grad_norm, 4),
        }

    grand_rate = results["grand (all views)"]["seconds_per_step"]
    per_coalition = {
        f"{steps} steps": {
            "hours": round(grand_rate * steps / 3600, 3),
            "minutes": round(grand_rate * steps / 60, 1),
        }
        for steps in EXTRAPOLATION_STEPS
    }

    # declared-scope projection: cost-estimator model counts x measured rate
    cb = n_coalition_models_for_scope(cfg, frozen_steps=max(EXTRAPOLATION_STEPS))
    n_models = cb.total_coalition_models
    per_model_hours = grand_rate * max(EXTRAPOLATION_STEPS) / 3600
    projected = {
        "low": round(0.6 * n_models * per_model_hours, 1),   # Beauty-like share discount
        "high": round(1.0 * n_models * per_model_hours, 1),
        "coalition_models_in_scope": n_models,
        "per_model_hours_at_10k_steps": round(per_model_hours, 3),
    }

    record = {
        "dataset": cfg.dataset,
        "device": device,
        "steps_probed": args.steps,
        "measured": results,
        "extrapolated_per_coalition": per_coalition,
        "projected_study_hours": projected,
        "determinism": det,
        "peak_rss_mb": round(psutil.Process().memory_info().rss / (1 << 20), 1),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": (
            "sequential single-process extrapolation from a SHORT probe; the full "
            "study logs measured per-coalition wall time and the paper reports those"
        ),
    }
    out_path = os.path.join(out_dir, f"timing-{cfg.dataset}.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    print(json.dumps(record, indent=2))
    print(f"\nsaved: {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="measure per-step training cost on this machine")
    p.add_argument("--dataset", choices=["ml1m", "beauty", "ml100k", "synthetic"], default="ml1m")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="probe optimizer steps per coalition")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    p.add_argument("--out-dir", default=None, help="override results/timing output dir")
    p.add_argument("--processed-root", default=None, help="override the frozen-artifact root")
    p.add_argument("--combine", action="store_true",
                   help="print the combined estimate across already-probed datasets")
    args = p.parse_args()
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
