"""DATA_BUILD + DATA_VALIDATE stages.

Builds the frozen data artifact for one dataset:
    python scripts/build_data.py --dataset ml1m    [--raw-path ...]
    python scripts/build_data.py --dataset beauty  [--raw-path ...]
    python scripts/build_data.py --dataset synthetic [--n-users 96 ...]

Synthetic mode is the test/verification dataset (tiny; completes in seconds)
and is also what the end-to-end suite uses. Real datasets require the raw
files under data/raw (auto-downloaded for ml-1m when possible; Beauty must be
provided per its license terms).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaper.config import load_run_config  # noqa: E402
from shaper.data import (  # noqa: E402
    FrozenData,
    build_dataset_artifact,
    build_synthetic_interactions,
    load_ml1m_raw,
)
from shaper.logging_utils import StructuredLogger  # noqa: E402


def build(cfg, args) -> dict:
    if args.dataset == "synthetic":
        raw = build_synthetic_interactions(
            n_users=args.n_users,
            n_items=args.n_items,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.synthetic_seed,
        )
    elif args.dataset == "ml1m":
        raw_path = args.raw_path or os.path.join(cfg.paths["data_raw"], "ml-1m", "ratings.dat")
        if not os.path.exists(raw_path):
            zip_path = os.path.join(cfg.paths["data_raw"], "ml-1m.zip")
            if os.path.exists(zip_path):
                import zipfile

                with zipfile.ZipFile(zip_path) as zf:
                    zf.extract("ml-1m/ratings.dat", cfg.paths["data_raw"])
                raw_path = os.path.join(cfg.paths["data_raw"], "ml-1m", "ratings.dat")
            else:
                raise SystemExit(
                    "ml-1m.zip not found under data/raw. Download it from GroupLens "
                    "(https://files.grouplens.org/datasets/movielens/ml-1m.zip) and retry."
                )
        raw = load_ml1m_raw(raw_path)
    elif args.dataset == "beauty":
        raw_path = args.raw_path or os.path.join(cfg.paths["data_raw"], "Beauty_5.json.gz")
        if not os.path.exists(raw_path):
            raise SystemExit(
                "Beauty_5.json.gz not found under data/raw. Obtain the Amazon Reviews 2018 "
                "Beauty 5-core file (http://deepyeti.ucsd.edu/jianmo/amazon) and retry."
            )
        from shaper.data import load_beauty_raw

        raw = load_beauty_raw(raw_path)
    else:  # pragma: no cover
        raise SystemExit(f"unknown dataset {args.dataset}")

    logger = StructuredLogger(cfg.paths["data_manifests"], run_id="data-build", name=f"build-{cfg.dataset}")
    manifest = build_dataset_artifact(
        cfg,
        raw,
        processed_root=os.path.join(cfg.paths["data_processed"], cfg.dataset),
        force=args.force,
        logger=logger,
    )
    return manifest


def validate(cfg) -> dict:
    data = FrozenData(cfg.dataset, processed_root=os.path.join(cfg.paths["data_processed"], cfg.dataset)).load()
    checks = data.validate(cfg=cfg)
    all_pass = all(c["pass"] for c in checks.values())
    if not all_pass:
        for name, c in checks.items():
            if not c["pass"]:
                print(f"FAIL {name}: {c['detail']}")
        raise SystemExit("DATA_VALIDATE failed")
    print(json.dumps({"data_validate": "PASS", "checks": checks,
                      "data_hash": data.data_hash}, indent=2, sort_keys=True))
    return checks


def main() -> int:
    from _cli import base_parser

    p = base_parser("build + validate the frozen data artifact")
    p.add_argument("--raw-path", default=None)
    p.add_argument("--force", action="store_true", help="overwrite an existing frozen artifact")
    p.add_argument("--skip-validate", action="store_true")
    p.add_argument("--n-users", type=int, default=96)
    p.add_argument("--n-items", type=int, default=40)
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--max-len", type=int, default=12)
    p.add_argument("--synthetic-seed", type=int, default=7)
    args = p.parse_args()

    cfg = load_run_config(args.dataset)
    manifest = build(cfg, args)
    print(json.dumps(
        {"data_hash": manifest["data_hash"], "config_hash": manifest["config_hash"],
         "stats": manifest["stats"]},
        indent=2, sort_keys=True,
    ))
    if not args.skip_validate:
        validate(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
