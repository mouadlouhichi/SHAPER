"""Integration: the REAL dataset download path end-to-end, exercised against
a local HTTP server (this sandbox cannot reach files.grouplens.org; the
canonical URL stays registered, the URL is redirected to a local server for
the test only).

Covers: --download flag -> streaming download -> published-MD5 verification
-> sidecar provenance manifest -> zip extraction -> ratings.dat parsing ->
rating>=4 conversion -> iterative 5-core -> artifact freeze -> validation.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import io
import os
import sys
import threading
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper import data_download as dd
from shaper.config import load_run_config
from shaper.data import FrozenData


def _ml1m_ratings_bytes(n_users=12, n_items=10, per_user=6, seed=7):
    import random

    rng = random.Random(seed)
    lines = []
    ts = 1_600_000_000
    for u in range(1, n_users + 1):
        for k in range(per_user):
            item = 1 + (u + k) % n_items
            rating = rng.choice([4, 5])
            lines.append(f"{u}::{item}::{rating}::{ts}")
            ts += 60
    return ("\n".join(lines) + "\n").encode()


class _ZipHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.end_headers()
        self.wfile.write(type(self).payload)

    def log_message(self, *args):
        pass


def test_real_ml1m_download_path_end_to_end(tmp_path, monkeypatch):
    # --- serve a genuine ml-1m.zip layout from a local server --------------
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ml-1m/ratings.dat", _ml1m_ratings_bytes())
    payload = buf.getvalue()
    md5 = hashlib.md5(payload).hexdigest()
    _ZipHandler.payload = payload
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ZipHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/ml-1m.zip"
    try:
        # redirect ONLY this test's download to the local server; the
        # published MD5 (locked in configs/ml1m.yaml) must match the served
        # archive, so the source registry's md5 is overridden with the served
        # file's md5 and the downloader verifies it exactly as in production
        monkeypatch.setitem(
            dd.SOURCES["ml1m"], "urls", [url]
        )
        monkeypatch.setitem(dd.SOURCES["ml1m"], "md5", md5)
        monkeypatch.setitem(dd.SOURCES["ml1m"], "min_size_bytes", 1)

        cfg = load_run_config("ml1m")
        cfg.paths.update(
            {
                "results": str(tmp_path / "results"),
                "data_raw": str(tmp_path / "raw"),
                "data_processed": str(tmp_path / "processed"),
                "data_manifests": str(tmp_path / "manifests"),
            }
        )
        args = argparse.Namespace(
            dataset="ml1m", run_id="dl-test", raw_path=None, force=False,
            skip_validate=False, n_users=96, n_items=40, min_len=6, max_len=12,
            synthetic_seed=7, download=True,
        )

        from scripts import build_data

        manifest = build_data.build(cfg, args)

        # 1. the archive was downloaded and verified
        raw_zip = tmp_path / "raw" / "ml-1m.zip"
        assert raw_zip.exists()
        sidecar = json_sidecar(raw_zip)
        assert sidecar["verified"] and sidecar["hashes"]["md5"] == md5
        # 2. the download provenance is recorded in the frozen artifact
        assert manifest["raw_download"]["hashes"]["md5"] == md5
        assert manifest["raw_download"]["url"] == url
        # 3. the artifact is valid and its stats reflect the rating>=4
        #    conversion (never the raw count)
        data = FrozenData("ml1m", processed_root=str(tmp_path / "processed" / "ml1m"), cfg=cfg).load()
        checks = data.validate(cfg=cfg)
        assert all(c["pass"] for c in checks.values()), checks
        assert manifest["stats"]["interactions"] == 12 * 6
        assert manifest["positive_before_core"] == 12 * 6
        # 4. data validation gate passes
        assert checks["temporal_order"]["pass"]

        # 5. re-run: the frozen artifact is reused byte-identically (never
        #    rebuilt), and the downloader re-verifies without re-downloading
        manifest2 = build_data.build(cfg, args)
        assert manifest2["data_hash"] == manifest["data_hash"]
        from shaper.data_download import ensure_raw_dataset

        dl = ensure_raw_dataset("ml1m", cfg.paths["data_raw"], download=True)
        assert dl["already_present"] is True
        assert dl["verified"] is True
    finally:
        server.shutdown()


def json_sidecar(zip_path):
    import json

    with open(str(zip_path) + ".download.json", encoding="utf-8") as fh:
        return json.load(fh)
