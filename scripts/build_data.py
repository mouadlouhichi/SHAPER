"""DATA_BUILD + DATA_VALIDATE stages.

Builds the frozen data artifact for one dataset:
    python scripts/build_data.py --dataset ml1m    [--download] [--raw-path ...]
    python scripts/build_data.py --dataset beauty  [--download] [--raw-path ...]
    python scripts/build_data.py --dataset synthetic [--n-users 96 ...]

Real datasets are downloaded from their canonical sources on request
(--download; shaper/data_download.py streams, verifies checksums, and writes
a sidecar manifest). Synthetic mode is the test/verification dataset (tiny;
completes in seconds) and is also what the end-to-end suite uses.
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
from shaper.data_download import ensure_raw_dataset  # noqa: E402
from shaper.logging_utils import StructuredLogger  # noqa: E402


def _build_logger(cfg) -> StructuredLogger:
    return StructuredLogger(cfg.paths["data_manifests"], run_id="data-build", name=f"build-{cfg.dataset}")


def build(cfg, args) -> dict:
    logger = _build_logger(cfg)
    raw_download_info = None
    if args.dataset == "synthetic":
        raw = build_synthetic_interactions(
            n_users=args.n_users,
            n_items=args.n_items,
            min_len=args.min_len,
            max_len=args.max_len,
            seed=args.synthetic_seed,
        )
    elif args.dataset == "ml1m":
        if args.raw_path:
            raw_path = args.raw_path
            if raw_path.endswith(".zip"):
                raw_path = _extract_ml1m_zip(cfg, raw_path)
        else:
            # canonical flow: ensure the verified archive, then parse ratings.dat
            dl = ensure_raw_dataset("ml1m", cfg.paths["data_raw"], download=args.download,
                                    force=args.force, logger=logger)
            zip_path = os.path.join(cfg.paths["data_raw"], "ml-1m.zip")
            raw_path = _extract_ml1m_zip(cfg, zip_path)
            raw_download_info = dl
        raw = load_ml1m_raw(raw_path)
    elif args.dataset == "beauty":
        if args.raw_path:
            raw_path = args.raw_path
        else:
            info = ensure_raw_dataset("beauty", cfg.paths["data_raw"], download=args.download,
                                      force=args.force, logger=logger)
            raw_path = info["path"]
            raw_download_info = info
        from shaper.data import load_beauty_raw

        raw = load_beauty_raw(raw_path)
    else:  # pragma: no cover
        raise SystemExit(f"unknown dataset {args.dataset}")

    manifest = build_dataset_artifact(
        cfg,
        raw,
        processed_root=os.path.join(cfg.paths["data_processed"], cfg.dataset),
        force=args.force,
        logger=logger,
        raw_download_info=raw_download_info,
    )
    return manifest


def _extract_ml1m_zip(cfg, zip_path: str) -> str:
    """Extract ratings.dat from the verified archive (idempotent)."""
    out_dir = os.path.join(cfg.paths["data_raw"], "ml-1m")
    ratings_path = os.path.join(out_dir, "ratings.dat")
    if os.path.exists(ratings_path):
        return ratings_path
    os.makedirs(out_dir, exist_ok=True)
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("ml-1m/ratings.dat", cfg.paths["data_raw"])
    return ratings_path


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
    p.add_argument("--force", action="store_true", help="overwrite an existing frozen artifact / re-download")
    p.add_argument("--download", action="store_true",
                   help="download the raw dataset from its canonical source (verified checksums)")
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
