"""Run artifacts: results directory scaffolding, compute tracking, manifests,
spec-compliance and reproducibility reports (spec sections 57-59, 67).

results/runs/<run_id>/
├── manifest.json            # run identity + hashes
├── config.yaml              # frozen configuration snapshot
├── provenance.json          # repo commit/dirty state, spec versions
├── environment.json         # OS/CPU/GPU/CUDA/torch/versions/determinism
├── seed_registry.json       # the locked seed registry
├── scope_manifest.json      # the locked experiment scope
├── logs/                    # structured JSONL event logs
├── checkpoints/             # atomic coalition checkpoints (+ manifest)
├── raw/                     # raw metric outputs
├── metrics/                 # aggregated metrics
├── coalition_tables/        # realized value tables
├── shapley/                 # exact allocations, efficiency residuals
├── interactions/            # Grabisch-Roubens indices, epsilon_pq
├── segments/                # RQ3 outputs
├── interventions/           # RQ4 outputs
├── tables/                  # rendered tables
├── figures/                 # rendered figures
├── summary.json / summary.md
├── spec_compliance.json / spec_compliance.md
└── reproducibility_report.md
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch

from .config import RunConfig, dataset_config_summary
from .logging_utils import StructuredLogger
from .provenance import canonical_json, environment_record, file_hash, stable_hash

SUBDIRS = [
    "logs", "checkpoints", "raw", "metrics", "coalition_tables", "shapley",
    "interactions", "segments", "interventions", "tables", "figures",
]


class ComputeTracker:
    """Measured compute accounting (spec section 59).

    The paper reports MEASURED values only; planning estimates are never
    presented as results.
    """

    def __init__(self):
        self.wall_seconds: Dict[str, float] = {}
        self.gpu_time_seconds: float = 0.0
        self.peak_memory_allocated_bytes: Optional[int] = None
        self.peak_memory_reserved_bytes: Optional[int] = None
        self.n_coalition_models: int = 0
        self.training_steps: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.shapley_seconds: float = 0.0
        self.evaluation_seconds: float = 0.0
        self.stage_runtime: Dict[str, float] = {}

    def record_memory(self) -> None:
        if torch.cuda.is_available():  # pragma: no cover - GPU environments
            self.peak_memory_allocated_bytes = torch.cuda.max_memory_allocated()
            self.peak_memory_reserved_bytes = torch.cuda.max_memory_reserved()

    def summary(self) -> Dict[str, Any]:
        return {
            "wall_seconds_by_stage": self.wall_seconds,
            "gpu_time_seconds": self.gpu_time_seconds,
            "peak_memory_allocated_bytes": self.peak_memory_allocated_bytes,
            "peak_memory_reserved_bytes": self.peak_memory_reserved_bytes,
            "n_coalition_models_trained": self.n_coalition_models,
            "total_training_steps": self.training_steps,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": (
                self.cache_hits / (self.cache_hits + self.cache_misses)
                if (self.cache_hits + self.cache_misses) > 0
                else None
            ),
            "shapley_seconds": self.shapley_seconds,
            "evaluation_seconds": self.evaluation_seconds,
            "measured_note": "measured values; planning estimates are never reported as results",
        }


class RunDirectory:
    """One results/runs/<run_id> workspace."""

    def __init__(self, run_id: str, results_root: Optional[str] = None,
                 repo_root: Optional[str] = None):
        from .config import REPO_ROOT

        self.repo_root = repo_root or REPO_ROOT
        self.results_root = results_root or os.path.join(self.repo_root, "results", "runs")
        self.run_id = run_id
        self.root = os.path.join(self.results_root, run_id)
        self.tracker = ComputeTracker()
        self.logger: Optional[StructuredLogger] = None

    # -- scaffolding ----------------------------------------------------------
    def create(self, cfg: RunConfig, resume: bool = False) -> "RunDirectory":
        os.makedirs(self.root, exist_ok=True)
        for sub in SUBDIRS:
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)
        if not os.path.exists(os.path.join(self.root, "manifest.json")) or not resume:
            self._write_identity(cfg)
        self.logger = StructuredLogger(os.path.join(self.root, "logs"), self.run_id)
        return self

    def _write_identity(self, cfg: RunConfig) -> None:
        import yaml as _yaml

        env = environment_record(
            {
                "determinism": _determinism_flags(),
                "python_target": "3.12 (spec); verified interpreter recorded in environment.json",
            }
        )
        with open(os.path.join(self.root, "environment.json"), "w", encoding="utf-8") as fh:
            json.dump(env, fh, indent=2, sort_keys=True)
        with open(os.path.join(self.root, "config.yaml"), "w", encoding="utf-8") as fh:
            _yaml.safe_dump(dataset_config_summary(cfg), fh, sort_keys=True)
        with open(os.path.join(self.root, "seed_registry.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg.seeds, fh, indent=2, sort_keys=True)
        with open(os.path.join(self.root, "scope_manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg.scope, fh, indent=2, sort_keys=True)
        with open(os.path.join(self.root, "provenance.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "repository": env,
                    "config_hash": cfg.config_hash(),
                    "dataset": cfg.dataset,
                    "specs": {
                        "implementation_spec": file_hash(os.path.join(self.repo_root, "specs", "SHAPER_Implementation_Spec.md")),
                        "paper_structure": file_hash(os.path.join(self.repo_root, "specs", "SHAPER_Paper_Structure.md")),
                    },
                    "overrides": cfg.overrides,
                },
                fh,
                indent=2,
                sort_keys=True,
            )
        manifest = {
            "run_id": self.run_id,
            "dataset": cfg.dataset,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_hash": cfg.config_hash(),
            "data_hash": None,
            "recipe_hash": None,
            "status": "created",
            "stages": {},
        }
        with open(os.path.join(self.root, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    # -- manifest bookkeeping ---------------------------------------------------
    def update_manifest(self, **fields: Any) -> None:
        path = os.path.join(self.root, "manifest.json")
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        manifest.update(fields)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    def stage_status(self, stage: str, status: str, **fields: Any) -> None:
        path = os.path.join(self.root, "manifest.json")
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        entry = {"status": status, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        entry.update(fields)
        manifest.setdefault("stages", {})[stage] = entry
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

    def path(self, sub: str, name: Optional[str] = None) -> str:
        p = os.path.join(self.root, sub)
        if name is not None:
            p = os.path.join(p, name)
        os.makedirs(os.path.dirname(p) if name is not None else p, exist_ok=True)
        return p

    def write_json(self, sub: str, name: str, obj: Any) -> str:
        path = self.path(sub, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        return path

    def write_markdown(self, sub: str, name: str, text: str) -> str:
        path = self.path(sub, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def checkpoint_root(self) -> str:
        return self.path("checkpoints")

    # -- final reports -----------------------------------------------------------
    def write_summary(self, summary: Dict[str, Any]) -> str:
        self.write_json("", "summary.json", summary)
        return self.write_markdown("", "summary.md", render_summary_markdown(summary))

    def write_reproducibility_report(self, cfg: RunConfig, extra: Optional[Dict[str, Any]] = None) -> str:
        env = environment_record(_determinism_flags())
        lines = [
            "# Reproducibility report",
            "",
            f"- run_id: `{self.run_id}`",
            f"- dataset: `{cfg.dataset}`",
            f"- config hash: `{cfg.config_hash()}`",
            f"- python (verified interpreter): `{env['python']}`",
            f"- torch: `{env.get('torch_version', 'n/a')}`; CUDA available: `{env.get('cuda_available', 'n/a')}`",
            f"- OS: `{env['os']} {env.get('os_release', '')}`; CPU count: `{env.get('cpu_count', 'n/a')}`",
            f"- repository commit: `{env.get('commit')}`; dirty: `{env.get('dirty')}`",
            f"- deterministic flags: `{json.dumps(_determinism_flags())}`",
            f"- peak allocated GPU memory: `{self.tracker.peak_memory_allocated_bytes}` bytes",
            f"- peak reserved GPU memory: `{self.tracker.peak_memory_reserved_bytes}` bytes",
            "",
            "## Lock file",
            "",
            "Exact resolved dependency versions are recorded in `requirements.lock`",
            "(authoring constraint: Python 3.12, torch >= 2.4, numpy >= 2.4,<2.5,",
            "scipy >= 1.18; the CI sandbox verified the code on Python 3.11 with the",
            "newest compatible SciPy, recorded in environment.json).",
            "",
            "## Determinism",
            "",
            "`torch.use_deterministic_algorithms(True)` with cuDNN benchmarking disabled",
            "and deterministic cuDNN enabled (see preflight). The canonical grand",
            "coalition is repeated on seed 2001 and the absolute utility difference",
            "is reported as the execution-nondeterminism floor; contextual marginals",
            "whose magnitude does not exceed the floor are labeled indistinguishable",
            "from execution nondeterminism.",
            "",
            "## Random schedules",
            "",
            "Augmentation draws are keyed by (seed, optimizer_step, global_example_id,",
            "occurrence, view); recommendation negatives by (seed, optimizer_step,",
            "global_user_id, target_position); epoch permutations by (seed, epoch,",
            "dataset_hash); dropout forwards by (seed, optimizer_step, purpose, view,",
            "pass_index). Worker count and coalition enumeration order cannot change",
            "any schedule.",
            "",
        ]
        if extra:
            lines += ["## Run-specific notes", ""]
            for key, value in extra.items():
                lines.append(f"- {key}: {value}")
        text = "\n".join(lines) + "\n"
        return self.write_markdown("", "reproducibility_report.md", text)


def _determinism_flags() -> Dict[str, Any]:
    flags = {"cudnn_benchmark": torch.backends.cudnn.benchmark,
             "cudnn_deterministic": torch.backends.cudnn.deterministic}
    try:
        flags["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    except Exception:  # pragma: no cover
        flags["torch_deterministic_algorithms"] = "unknown"
    return flags


def render_summary_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# SHAPER run summary", ""]
    for key in ("run_id", "dataset", "config_hash", "data_hash", "recipe_hash", "status"):
        if key in summary:
            lines.append(f"- **{key}**: `{summary[key]}`")
    lines.append("")
    stages = summary.get("stages", {})
    if stages:
        lines.append("## Stages")
        lines.append("")
        lines.append("| stage | status | notes |")
        lines.append("|---|---|---|")
        for stage, info in stages.items():
            status = info.get("status", "?") if isinstance(info, dict) else str(info)
            notes = info.get("note", "") if isinstance(info, dict) else ""
            lines.append(f"| {stage} | {status} | {notes} |")
    compute = summary.get("compute")
    if compute:
        lines.append("")
        lines.append("## Compute (measured)")
        lines.append("")
        for key, value in compute.items():
            lines.append(f"- **{key}**: {value}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ComputeTracker",
    "RunDirectory",
    "render_summary_markdown",
]
