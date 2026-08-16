"""Atomic checkpointing with a checkpoint manifest and corruption rejection.

Rules (spec: "ATOMIC CHECKPOINTS", "RESUME", "CACHE VALIDATION"):
  - active checkpoints are never written directly: temporary -> flush ->
    fsync -> atomic rename
  - `checkpoint_manifest.json` records the latest valid checkpoint, completed
    steps and completed coalitions with artifact hashes
  - a corrupted checkpoint is REJECTED and the previous valid checkpoint is
    restored (never silently loaded)
  - cached coalition results are reused only when dataset/seed/policy/
    coalition/config-hash/data-hash/recipe-hash all match AND the checkpoint
    validates.

Coalition checkpoints contain (spec): model state, optimizer state, scheduler
state, epoch, optimizer step, seed, coalition, policy, RNG states, config
hash, data hash, recipe hash, metric history.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from .provenance import canonical_json, file_hash, stable_hash

MANIFEST_NAME = "checkpoint_manifest.json"


def _atomic_write_bytes(path: str, data: bytes) -> None:
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".ckpt-tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_save_torch(obj: Any, path: str) -> str:
    """torch.save to a temp file, fsync, atomic rename. Returns file hash."""
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".pt-tmp-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        with open(tmp, "rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return file_hash(path)


def atomic_write_json(obj: Any, path: str) -> None:
    _atomic_write_bytes(path, canonical_json(obj) + b"\n")


def save_coalition_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    seed: int,
    coalition: List[str],
    policy: str,
    rng_states: Dict[str, Any],
    config_hash: str,
    data_hash: str,
    recipe_hash: str,
    metric_history: Dict[str, List[float]],
    extra: Optional[Dict[str, Any]] = None,
    dataset: Optional[str] = None,
) -> str:
    """Atomically persist a coalition checkpoint and return its file hash.

    `dataset` is part of the cache-validity key (spec CACHE VALIDATION).
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "seed": seed,
        "coalition": sorted(coalition),
        "policy": policy,
        "rng_states": rng_states,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "recipe_hash": recipe_hash,
        "metric_history": metric_history,
        "extra": extra or {},
        "dataset": dataset,
    }
    h = atomic_save_torch(payload, path)
    return h


def validate_checkpoint_file(path: str) -> Tuple[bool, str]:
    """Check that a checkpoint file exists and loads."""
    if not os.path.exists(path):
        return False, "missing file"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        required = {
            "model",
            "optimizer",
            "scheduler",
            "epoch",
            "step",
            "seed",
            "coalition",
            "policy",
            "rng_states",
            "config_hash",
            "data_hash",
            "recipe_hash",
        }
        missing = required - set(payload.keys())
        if missing:
            return False, f"missing keys: {sorted(missing)}"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - corruption must be reported
        return False, f"corrupted: {type(exc).__name__}: {exc}"


def validate_cache_keys(payload: Dict[str, Any], *, dataset: str, seed: int,
                        policy: str, coalition: List[str], config_hash: str,
                        data_hash: str, recipe_hash: str) -> Tuple[bool, str]:
    """All eight cache-validity fields must match (spec CACHE VALIDATION)."""
    checks = [
        ("dataset", payload.get("dataset"), dataset),
        ("seed", payload.get("seed"), seed),
        ("policy", payload.get("policy"), policy),
        ("coalition", sorted(payload.get("coalition", [])), sorted(coalition)),
        ("config_hash", payload.get("config_hash"), config_hash),
        ("data_hash", payload.get("data_hash"), data_hash),
        ("recipe_hash", payload.get("recipe_hash"), recipe_hash),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            return False, f"{name} mismatch: {actual!r} != {expected!r}"
    return True, "ok"


class CheckpointManifest:
    """Persistent registry of completed work in one run/scope directory."""

    def __init__(self, root: str):
        self.root = root
        self.path = os.path.join(root, MANIFEST_NAME)
        os.makedirs(root, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed_coalitions": [],
            "completed_steps": {},
            "artifacts": {},
        }

    def _flush(self) -> None:
        atomic_write_json(self.data, self.path)

    def mark_coalition_complete(
        self,
        *,
        dataset: str,
        seed: int,
        coalition: List[str],
        policy: str,
        step: int,
        checkpoint_path: str,
        checkpoint_hash: str,
        config_hash: str,
        data_hash: str,
        recipe_hash: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        key = f"{dataset}:{policy}:{seed}:{'+'.join(sorted(coalition))}"
        entry = {
            "dataset": dataset,
            "seed": seed,
            "coalition": sorted(coalition),
            "policy": policy,
            "step": step,
            "checkpoint_path": checkpoint_path,
            "checkpoint_hash": checkpoint_hash,
            "config_hash": config_hash,
            "data_hash": data_hash,
            "recipe_hash": recipe_hash,
            "metrics": metrics or {},
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if key not in self.data["completed_coalitions"]:
            self.data["completed_coalitions"].append(key)
        self.data["artifacts"][key] = entry
        self._flush()

    def get(self, *, dataset: str, seed: int, coalition: List[str], policy: str) -> Optional[Dict[str, Any]]:
        key = f"{dataset}:{policy}:{seed}:{'+'.join(sorted(coalition))}"
        return self.data["artifacts"].get(key)

    def is_complete(self, *, dataset: str, seed: int, coalition: List[str], policy: str) -> bool:
        key = f"{dataset}:{policy}:{seed}:{'+'.join(sorted(coalition))}"
        return key in self.data["completed_coalitions"]

    def completed_coalitions(self, dataset: str, policy: str) -> List[str]:
        prefix = f"{dataset}:{policy}:"
        return [k for k in self.data["completed_coalitions"] if k.startswith(prefix)]

    def find_previous_valid_checkpoint(
        self, *, dataset: str, seed: int, coalition: List[str], policy: str
    ) -> Optional[str]:
        """Latest recorded checkpoint for this coalition that still validates.

        If the registered final checkpoint is corrupted it is REJECTED and the
        search falls back to `latest.pt` and numbered step snapshots in the
        same directory (restore-the-previous-valid-checkpoint semantics).
        """
        entry = self.get(dataset=dataset, seed=seed, coalition=coalition, policy=policy)
        if entry is None:
            return None
        d = os.path.dirname(entry["checkpoint_path"])
        candidates: List[str] = [entry["checkpoint_path"]]
        latest = os.path.join(d, "latest.pt")
        if latest not in candidates:
            candidates.append(latest)

        def _step_num(name: str) -> int:
            return int(name[len("step_"): -len(".pt")])

        snapshots = sorted(
            (os.path.join(d, n) for n in os.listdir(d)
             if n.startswith("step_") and n.endswith(".pt")),
            key=lambda p: _step_num(os.path.basename(p)),
            reverse=True,
        )
        candidates.extend(p for p in snapshots if p not in candidates)
        for path in candidates:
            ok, _ = validate_checkpoint_file(path)
            if ok:
                return path
        return None  # corrupted: rejected; no valid fallback remains


def load_coalition_checkpoint(path: str) -> Dict[str, Any]:
    ok, reason = validate_checkpoint_file(path)
    if not ok:
        raise RuntimeError(f"checkpoint rejected: {reason}")
    return torch.load(path, map_location="cpu", weights_only=False)


__all__ = [
    "MANIFEST_NAME",
    "atomic_save_torch",
    "atomic_write_json",
    "save_coalition_checkpoint",
    "validate_checkpoint_file",
    "validate_cache_keys",
    "load_coalition_checkpoint",
    "CheckpointManifest",
]
