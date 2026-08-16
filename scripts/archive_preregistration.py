"""Timestamped preregistration archive (spec B.7, "Archive gate").

Snapshots the protocol documents, frozen configuration files, dependency
lock, repository commit, and generation timestamp into a JSON + Markdown
archive under results/preregistration/. The preregistration happens AFTER
the code/tests freeze and BEFORE the two excluded pilot seeds; the
pilot-informed amendment and ARCHIVE_FREEZE follow it.

This is an evidence artifact (hash-pinned spec + configs), not a substitute
for an external registry (DOI/timestamp), which remains a declared
publication-time action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shaper.config import REPO_ROOT, load_run_config  # noqa: E402
from shaper.provenance import file_hash, repository_info, stable_hash  # noqa: E402


def snapshot_artifacts() -> dict:
    files = [
        "specs/SHAPER_Implementation_Spec.md",
        "specs/SHAPER_Paper_Structure.md",
        "configs/seeds.yaml",
        "configs/scope.yaml",
        "configs/statistics.yaml",
        "configs/ml1m.yaml",
        "configs/beauty.yaml",
        "configs/manifest_freeze.yaml",
        "configs/amendment.yaml",
        "requirements.txt",
        "requirements.lock",
        "pyproject.toml",
    ]
    hashes = {}
    for rel in files:
        path = os.path.join(REPO_ROOT, rel)
        if os.path.exists(path):
            hashes[rel] = {"sha256": file_hash(path)}
        else:
            hashes[rel] = {"sha256": None, "note": "missing"}
    return hashes


def generate_preregistration(dataset: str) -> dict:
    cfg = load_run_config(dataset)
    record = {
        "kind": "staged preregistration archive",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": cfg.dataset,
        "config_hash": cfg.config_hash(),
        "artifacts": snapshot_artifacts(),
        "repository": repository_info(cwd=REPO_ROOT),
        "note": (
            "Preregistered BEFORE the two excluded pilot seeds (spec B.7). "
            "The pilot-informed amendment and ARCHIVE_FREEZE follow; an external "
            "registry DOI/timestamp is added at publication time."
        ),
    }
    record["archive_hash"] = stable_hash(
        {k: v["sha256"] for k, v in record["artifacts"].items()}, salt="preregistration"
    )
    return record


def main() -> int:
    p = argparse.ArgumentParser(description="generate the preregistration archive")
    p.add_argument("--dataset", choices=["ml1m", "beauty", "ml100k", "synthetic"], default="ml1m")
    args = p.parse_args()

    record = generate_preregistration(args.dataset)
    out_dir = os.path.join(REPO_ROOT, "results", "preregistration")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f"preregistration-{args.dataset}.json")
    md_path = os.path.join(out_dir, f"preregistration-{args.dataset}.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=True)
    lines = [
        "# SHAPER staged preregistration archive",
        "",
        f"- timestamp: `{record['timestamp']}`",
        f"- dataset: `{record['dataset']}`",
        f"- config hash: `{record['config_hash']}`",
        f"- archive hash (spec + configs + lock): `{record['archive_hash']}`",
        f"- repository commit: `{record['repository'].get('commit')}`; dirty: `{record['repository'].get('dirty')}`",
        "",
        "## Artifact hashes",
        "",
        "| artifact | sha256 |",
        "|---|---|",
    ]
    for rel, info in record["artifacts"].items():
        lines.append(f"| {rel} | `{info['sha256']}` |")
    lines += ["", record["note"]]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"PREREGISTRATION archived: {json_path}")
    print(json.dumps({k: record[k] for k in ("timestamp", "dataset", "config_hash", "archive_hash")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
