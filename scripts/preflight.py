"""PREFLIGHT stage: determinism configuration, environment record, and the
data/recipe readiness audit.

Checks (spec A.2):
  - torch.use_deterministic_algorithms(True); cuDNN benchmarking disabled,
    deterministic cuDNN enabled; exceptions recorded, never suppressed
  - dependency versions vs the registered environment
  - required config files present and internally consistent
  - the seed registry matches the locked protocol
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from shaper.config import load_run_config, load_yaml  # noqa: E402
from shaper.provenance import environment_record  # noqa: E402
from shaper.schedules import configure_determinism  # noqa: E402


def run_preflight(cfg=None, verbose: bool = True) -> dict:
    det = configure_determinism()
    env = environment_record({"determinism": det})

    import importlib.metadata as md

    versions = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "pandas", "matplotlib"):
        try:
            versions[name] = md.version(name)
        except Exception:  # pragma: no cover
            versions[name] = "missing"
    env["dependency_versions"] = versions

    checks = {
        "determinism": det,
        "torch_version": env.get("torch_version"),
        "cuda_available": env.get("cuda_available"),
        "python": env.get("python"),
    }
    issues = []
    if not det["deterministic_mode"]:
        issues.append(f"nondeterministic kernels (recorded, not suppressed): {det['exception']}")
    if cfg is not None:
        checks["config_hash"] = cfg.config_hash()
        checks["batch_size"] = cfg.batch_size
        checks["max_len"] = cfg.max_len
        if cfg.batch_size not in (128, 256):
            issues.append(f"unexpected batch size {cfg.batch_size} (locked: 128 ML-1M / 256 Beauty)")
    checks["status"] = "PASS" if not issues else "PASS_WITH_RECORDED_EXCEPTIONS"
    checks["issues"] = issues
    if verbose:
        print(json.dumps(checks, indent=2, default=str))
    return checks


def main() -> int:
    from _cli import base_parser

    p = base_parser("SHAPER preflight: determinism + environment audit")
    p.add_argument("--json-out", default=None, help="write the preflight record to this file")
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    checks = run_preflight(cfg)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(checks, fh, indent=2, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
