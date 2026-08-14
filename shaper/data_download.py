"""Raw-dataset downloader with checksum verification (engineering-detail
module, like shaper.config / shaper.cost — holds no scientific logic).

Spec A.1: ``data/raw/`` holds "downloaded archives; checksums, never
committed if licensing forbids" — this module implements exactly that:

  - streams the archive from the canonical source URL(s) to a `.part` file
    and atomically renames it on success;
  - verifies the published MD5 for MovieLens-1M (c4d9eecf... locked in
    configs/ml1m.yaml) and the gzip/zip magic bytes + a minimum size for
    Beauty (no published checksum exists for the Amazon file; the SHA-256
    computed during download is recorded in a sidecar manifest instead);
  - writes a `*.download.json` sidecar manifest next to the archive with the
    URL used, retrieval timestamp, size and hashes, which the data-artifact
    manifest records as `raw_download`;
  - refuses to overwrite an existing archive unless `force=True`, and
    re-verifies existing files on every call;
  - retries each candidate URL once and removes the partial file on failure.

Licensing: the archives are downloaded for local research use only and are
never committed (data/raw/* is gitignored). Redistribution terms belong to
GroupLens and the Amazon Reviews authors respectively.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import zipfile
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------
# Source registry (canonical URLs; operational metadata, NOT part of the
# frozen scientific configuration, so config hashes stay unchanged)
# --------------------------------------------------------------------------

SOURCES: Dict[str, Dict[str, Any]] = {
    "ml1m": {
        "filename": "ml-1m.zip",
        "urls": [
            "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        ],
        # published GroupLens checksum (also recorded in configs/ml1m.yaml)
        "md5": "c4d9eecfca2ab87c1945afe126590906",
        "min_size_bytes": 5_000_000,
        "kind": "zip",
        "license_note": (
            "GroupLens MovieLens-1M; research use. Archive downloaded locally; "
            "never committed (data/raw/* is gitignored)."
        ),
    },
    "ml100k": {
        "filename": "ml-100k.zip",
        "urls": [
            "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        ],
        # published GroupLens checksum (d2l DATA_HUB registry of the official
        # file; SHA-1, 40 hex chars)
        "sha1": "cd4dcac4241c8a4ad7badc7ca635da8a69dddb83",
        # fallback mirror: a GitHub repo with the complete extracted ml-100k
        # files (u.data et al.). Mirror archives carry NO published checksum,
        # so they are verified by size + magic bytes, and the extracted
        # u.data is content-verified by the canonical fingerprint in
        # shaper.data.load_ml100k_raw (100,000 rows; users 1..943; items
        # 1..1682; ratings 1..5; canonical first rows).
        "mirror_urls": [
            "https://codeload.github.com/SudeshGowda/ml-100k-dataset/tar.gz/main",
        ],
        "mirror_filename": "ml-100k-mirror.tar.gz",
        "min_size_bytes": 1_000_000,
        "kind": "zip",
        "license_note": (
            "GroupLens MovieLens-100K (VERIFICATION/TEST dataset — not part of "
            "the registered study). Archive downloaded locally; never committed."
        ),
    },
    "beauty": {
        "filename": "Beauty_5.json.gz",
        "urls": [
            "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Beauty_5.json.gz",
            "http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Beauty_5.json.gz",
        ],
        "md5": None,  # no published checksum; SHA-256 recorded in the sidecar
        "min_size_bytes": 1_000_000,
        "kind": "gzip",
        "license_note": (
            "Amazon Reviews 2018 Beauty 5-core (J. McAuley et al.). Archive "
            "downloaded locally; never committed; redistribution per the "
            "dataset's own terms."
        ),
    },
}

USER_AGENT = "SHAPER-data-download/1.0 (research pipeline; one-off retrieval)"
CHUNK = 1 << 20  # 1 MiB


def source_for(dataset: str) -> Dict[str, Any]:
    if dataset not in SOURCES:
        raise ValueError(
            f"no download source registered for dataset {dataset!r} "
            f"(registered: {sorted(SOURCES)})"
        )
    return SOURCES[dataset]


def _hash_file(path: str) -> Dict[str, str]:
    h256 = hashlib.sha256()
    hmd5 = hashlib.md5()
    hsha1 = hashlib.sha1()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h256.update(chunk)
            hmd5.update(chunk)
            hsha1.update(chunk)
    return {"sha256": h256.hexdigest(), "md5": hmd5.hexdigest(), "sha1": hsha1.hexdigest()}


def _magic_ok(path: str, kind: str) -> bool:
    with open(path, "rb") as fh:
        head = fh.read(4)
    if kind == "zip":
        return head[:4] == b"PK\x03\x04"
    if kind == "gzip":
        return head[:2] == b"\x1f\x8b"
    raise ValueError(f"unknown archive kind {kind!r}")


def _verify(path: str, source: Dict[str, Any], require_hash: bool = True) -> Dict[str, Any]:
    """Verify a downloaded archive; returns {ok, reason, hashes}.

    `require_hash=False` (mirror downloads) checks size + magic bytes only
    and records the computed SHA-256; content-level verification is then the
    loader's responsibility (canonical fingerprint at parse time).
    """
    size = os.path.getsize(path)
    if size < int(source.get("min_size_bytes", 0)):
        return {
            "ok": False,
            "reason": f"size {size} below minimum {source['min_size_bytes']}",
            "hashes": {},
        }
    # mirror archives use a different container (e.g. tar.gz); derive the
    # expected magic from the filename (the temp file keeps its ".part"
    # suffix while being verified)
    base = path[: -len(".part")] if path.endswith(".part") else path
    if base.endswith(".tar.gz") or base.endswith(".tgz"):
        kind = "gzip"
    else:
        kind = str(source["kind"])
    if not _magic_ok(path, kind):
        return {"ok": False, "reason": f"magic bytes mismatch (expected {kind})", "hashes": {}}
    hashes = _hash_file(path)
    if require_hash and source.get("md5") and hashes["md5"] != source["md5"]:
        return {
            "ok": False,
            "reason": f"md5 mismatch: got {hashes['md5']}, expected {source['md5']}",
            "hashes": hashes,
        }
    if require_hash and source.get("sha1") and hashes["sha1"] != source["sha1"]:
        return {
            "ok": False,
            "reason": f"sha1 mismatch: got {hashes['sha1']}, expected {source['sha1']}",
            "hashes": hashes,
        }
    return {"ok": True, "reason": "verified", "hashes": hashes}


def _fetch_once(url: str, dest: str, timeout: int = 120) -> int:
    """Stream one URL to `dest`; returns bytes written. Raises on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "wb") as fh:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                written += len(chunk)
    return written


def _download_with_progress(
    url: str, dest: str, total_hint: int = 0, timeout: int = 120
) -> int:
    """Stream with a tqdm progress bar (total unknown -> bytes counter)."""
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - tqdm is a declared dependency
        return _fetch_once(url, dest, timeout=timeout)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    written = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0) or total_hint
        with tqdm(
            total=total or None, unit="B", unit_scale=True, unit_divisor=1024,
            desc=os.path.basename(dest).replace(".part", ""), miniters=1,
        ) as bar:
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    written += len(chunk)
                    bar.update(len(chunk))
    return written


def download_dataset(
    dataset: str,
    raw_dir: str,
    force: bool = False,
    logger: Optional[Any] = None,
    progress: bool = True,
) -> Dict[str, Any]:
    """Ensure the raw archive for `dataset` exists under `raw_dir` and is
    verified. Returns a record {path, url, retrieved_at, size, hashes,
    verified, already_present}."""
    source = source_for(dataset)
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, source["filename"])
    mirror_dest = os.path.join(raw_dir, source.get("mirror_filename", source["filename"]))
    existing = dest if os.path.exists(dest) else (mirror_dest if os.path.exists(mirror_dest) else None)
    sidecar = dest + ".download.json"

    if existing is not None and not force:
        # canonical-named archives are held to the published checksum;
        # mirror-named archives to size + magic bytes (content fingerprint
        # verification happens at parse time)
        check = _verify(
            existing, source,
            require_hash=os.path.basename(existing) == source["filename"],
        )
        if check["ok"]:
            info: Dict[str, Any] = {
                "verified": True,
                "already_present": True,
                "url": None,
                "retrieved_at": None,
                "size": os.path.getsize(existing),
                "hashes": check["hashes"],
                "reason": "existing archive verified",
            }
            sc = existing + ".download.json"
            if os.path.exists(sc):
                with open(sc, encoding="utf-8") as fh:
                    stored = json.load(fh)
                info.update({k: v for k, v in stored.items() if k != "already_present"})
            info["path"] = existing
            return info
        raise RuntimeError(
            f"existing {existing} failed verification ({check['reason']}); "
            "re-run with force=True to re-download"
        )

    last_error: Optional[Exception] = None
    # candidates: (url, dest, require_published_hash)
    candidates: List[tuple] = [(u, dest, True) for u in source["urls"]]
    for u in source.get("mirror_urls", []):
        candidates.append(
            (u, os.path.join(raw_dir, source.get("mirror_filename", source["filename"])), False)
        )
    for attempt in range(2):
        for url, target, require_hash in candidates:
            part = target + ".part"
            try:
                written = (
                    _download_with_progress(url, part)
                    if progress
                    else _fetch_once(url, part)
                )
                if written < int(source.get("min_size_bytes", 0)):
                    raise RuntimeError(
                        f"downloaded only {written} bytes from {url} "
                        f"(minimum {source['min_size_bytes']})"
                    )
                # verify the PARTIAL file BEFORE the atomic rename, so a bad
                # download can never replace a good archive
                check = _verify(part, source, require_hash=require_hash)
                if not check["ok"]:
                    raise RuntimeError(f"verification failed: {check['reason']}")
                os.replace(part, target)
                info = {
                    "dataset": dataset,
                    "path": target,
                    "url": url,
                    "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "size": os.path.getsize(target),
                    "hashes": check["hashes"],
                    "verified": True,
                    "already_present": False,
                    "verification": (
                        "published checksum" if require_hash
                        else "archive integrity only (mirror); content fingerprint verified at parse time"
                    ),
                    "license_note": source["license_note"],
                }
                sidecar = target + ".download.json"
                with open(sidecar, "w", encoding="utf-8") as fh:
                    json.dump(info, fh, indent=2, sort_keys=True)
                if logger is not None:
                    logger.info(
                        "DATA_BUILD",
                        "raw_download",
                        dataset=dataset,
                        url=url,
                        size=info["size"],
                        sha256=check["hashes"]["sha256"][:16],
                        verification=info["verification"],
                    )
                return info
            except Exception as exc:  # noqa: BLE001 - try next URL / attempt
                last_error = exc
                if os.path.exists(part):
                    os.unlink(part)
                if logger is not None:
                    logger.warning(
                        "DATA_BUILD", "raw_download_retry", dataset=dataset, url=url, error=str(exc)
                    )
        time.sleep(2)
    raise RuntimeError(
        f"failed to download {dataset} from {source['urls'] + source.get('mirror_urls', [])}: {last_error}"
    )


def ensure_raw_dataset(
    dataset: str,
    raw_dir: str,
    download: bool = False,
    force: bool = False,
    logger: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Return the verified archive record if present, download if requested,
    or raise an actionable SystemExit telling the user how to obtain it."""
    source = source_for(dataset)
    dest = os.path.join(raw_dir, source["filename"])
    mirror_dest = os.path.join(raw_dir, source.get("mirror_filename", source["filename"]))
    present = dest if os.path.exists(dest) else (mirror_dest if os.path.exists(mirror_dest) else None)
    if present is not None:
        return download_dataset(dataset, raw_dir, force=False, logger=logger)
    if not download:
        raise SystemExit(
            f"{source['filename']} not found under {raw_dir}. Either download it "
            f"from {source['urls'][0]} and place it there, or re-run with "
            f"--download (notebook: stage('data', download=True)). "
            f"License note: {source['license_note']}"
        )
    return download_dataset(dataset, raw_dir, force=force, logger=logger)


def archive_integrity_check(path: str, dataset: str) -> Dict[str, Any]:
    """Integrity probe used by the tests and the build step: verify magic
    bytes, (optionally) the published MD5, and that the archive opens."""
    source = source_for(dataset)
    check = _verify(path, source)
    if check["ok"] and source["kind"] == "zip":
        try:
            with zipfile.ZipFile(path) as zf:
                if zf.testzip() is not None:
                    check = {"ok": False, "reason": "zip member CRC failure", "hashes": check["hashes"]}
        except Exception as exc:  # noqa: BLE001
            check = {"ok": False, "reason": f"zip open failed: {exc}", "hashes": check["hashes"]}
    if check["ok"] and source["kind"] == "gzip":
        try:
            with gzip.open(path, "rb") as fh:
                fh.read(1 << 16)
        except Exception as exc:  # noqa: BLE001
            check = {"ok": False, "reason": f"gzip open failed: {exc}", "hashes": check["hashes"]}
    return check


__all__ = [
    "SOURCES",
    "source_for",
    "download_dataset",
    "ensure_raw_dataset",
    "archive_integrity_check",
]
