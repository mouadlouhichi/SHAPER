"""Provenance primitives: canonical stable hashing, salts, run manifests.

All hashes used for cache validation / artifact freezing use one canonical
serialization (sorted keys, no whitespace) so that dict ordering, YAML
formatting, and worker count never affect them.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

HASH_ALGORITHM = "blake2b"  # frozen hash algorithm (recorded in data artifacts)


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON bytes: sorted keys, compact separators, UTF-8."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    ).encode("utf-8")


def stable_hash(obj: Any, salt: str = "") -> str:
    """Digest of the canonical serialization of `obj` under `HASH_ALGORITHM`."""
    h = hashlib.blake2b(digest_size=16)
    h.update(salt.encode("utf-8"))
    h.update(canonical_json(obj))
    return h.hexdigest()


def file_hash(path: str) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key_int(*parts: Any, salt: str = "shaper-key") -> int:
    """Deterministic nonnegative integer key for counter-based RNG seeding.

    Worker-invariant and coalition-enumeration-order-invariant because it is
    a pure function of its arguments.
    """
    return int(stable_hash(parts, salt=salt), 16)


def make_salt(seed: int, purpose: str) -> str:
    """Persisted per-artifact salt. Salts live in the frozen artifact so
    downstream stages can never silently change the hash keying."""
    return f"shaper:{purpose}:{seed}:{HASH_ALGORITHM}"


def config_hash(config: Dict[str, Any], salt: str = "") -> str:
    """Hash of the scientific configuration (dataset/model/aug/training/eval)."""
    return stable_hash(config, salt="config" + salt)


def data_manifest_hash(manifest: Dict[str, Any], salt: str = "") -> str:
    return stable_hash(manifest, salt="data-manifest" + salt)


def recipe_hash(recipe: Dict[str, Any], salt: str = "") -> str:
    return stable_hash(recipe, salt="recipe" + salt)


def repository_info() -> Dict[str, Any]:
    """Baseline repository commit and dirty state (empty if not a git repo)."""
    info: Dict[str, Any] = {"commit": None, "dirty": None, "branch": None}
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        info["commit_error"] = str(exc)
    try:
        info["branch"] = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as exc:  # pragma: no cover
        info["branch_error"] = str(exc)
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
        info["dirty"] = bool(dirty.strip())
    except Exception as exc:  # pragma: no cover
        info["dirty_error"] = str(exc)
    return info


def environment_record(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Machine/hardware/version record required in every run manifest."""
    import torch  # imported lazily so provenance can be used pre-install

    env: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "os": platform.system(),
        "os_release": platform.release(),
    }
    try:
        env["cuda_available"] = torch.cuda.is_available()
        env["cuda_version"] = torch.version.cuda
        env["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["torch_version"] = torch.__version__
        env["torch_threads"] = torch.get_num_threads()
    except Exception as exc:  # pragma: no cover
        env["torch_error"] = str(exc)
    for name in ("numpy", "scipy", "sklearn", "pandas", "matplotlib", "yaml"):
        try:
            mod = __import__("sklearn" if name == "sklearn" else name)
            env[f"{name}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover
            env[f"{name}_version"] = "missing"
    env.update(repository_info())
    if extra:
        env.update(extra)
    return env


__all__ = [
    "HASH_ALGORITHM",
    "canonical_json",
    "stable_hash",
    "file_hash",
    "key_int",
    "make_salt",
    "config_hash",
    "data_manifest_hash",
    "recipe_hash",
    "repository_info",
    "environment_record",
]
