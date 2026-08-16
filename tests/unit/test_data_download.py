"""Downloader tests: streaming, checksum verification, sidecar manifest,
idempotency, force re-download, and failure cleanup — exercised against a
local HTTP server (no external network required)."""

from __future__ import annotations

import gzip
import hashlib
import http.server
import io
import json
import os
import sys
import threading
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper import data_download as dd
from shaper.data_download import (
    archive_integrity_check,
    download_dataset,
    ensure_raw_dataset,
    source_for,
)


class _Handler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""
    hits: list = []
    md5_override: str = ""

    def do_GET(self):  # noqa: N802
        type(self).hits.append(self.path)
        if type(self).md5_override:
            # serve corrupted bytes when an override md5 is configured
            body = type(self).payload + b"corruption"
        else:
            body = type(self).payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture()
def http_server(tmp_path):
    # a tiny gzip archive with a known md5
    raw = b'{"reviewerID":"u1","asin":"i1","unixReviewTime":1}\n' * 50
    payload = gzip.compress(raw)
    _Handler.payload = payload
    _Handler.hits = []
    _Handler.md5_override = ""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/data.gz"
    yield tmp_path, url, hashlib.md5(payload).hexdigest(), payload
    server.shutdown()


@pytest.fixture()
def gz_source(monkeypatch, http_server):
    tmp_path, url, md5, _ = http_server
    monkeypatch.setitem(dd.SOURCES, "testds", {
        "filename": "test.gz",
        "urls": [url],
        "md5": md5,
        "min_size_bytes": 1,
        "kind": "gzip",
        "license_note": "test",
    })
    return tmp_path, url, md5


def test_download_writes_file_and_sidecar(gz_source):
    raw_dir, url, md5 = gz_source
    info = download_dataset("testds", str(raw_dir), progress=False)
    dest = raw_dir / "test.gz"
    assert info["path"] == str(dest)
    assert info["verified"] and not info["already_present"]
    assert info["url"] == url
    assert info["hashes"]["md5"] == md5
    assert info["retrieved_at"] is not None
    sidecar = json.loads((raw_dir / "test.gz.download.json").read_text())
    assert sidecar["hashes"]["sha256"]
    assert sidecar["license_note"] == "test"
    # gzip magic verified
    with open(dest, "rb") as fh:
        assert fh.read(2) == b"\x1f\x8b"


def test_existing_valid_file_no_network(gz_source):
    raw_dir, _, md5 = gz_source
    download_dataset("testds", str(raw_dir), progress=False)
    _Handler.hits.clear()
    info = download_dataset("testds", str(raw_dir), progress=False)
    assert info["already_present"] is True
    assert info["verified"] is True
    assert _Handler.hits == []  # no re-download


def test_corrupt_existing_file_refused_without_force(gz_source):
    raw_dir, _, _ = gz_source
    download_dataset("testds", str(raw_dir), progress=False)
    with open(raw_dir / "test.gz", "ab") as fh:
        fh.write(b"junk")
    with pytest.raises(RuntimeError, match="failed verification"):
        download_dataset("testds", str(raw_dir), progress=False)


def test_force_redownloads(gz_source):
    raw_dir, _, _ = gz_source
    download_dataset("testds", str(raw_dir), progress=False)
    with open(raw_dir / "test.gz", "ab") as fh:
        fh.write(b"junk")
    info = download_dataset("testds", str(raw_dir), force=True, progress=False)
    assert not info["already_present"]
    assert info["verified"]


def test_bad_checksum_from_server_raises_and_cleans_part(gz_source):
    raw_dir, _, _ = gz_source
    _Handler.md5_override = "corrupt"
    with pytest.raises(RuntimeError, match="failed to download"):
        download_dataset("testds", str(raw_dir), progress=False)
    assert not os.path.exists(raw_dir / "test.gz")
    assert not os.path.exists(raw_dir / "test.gz.part")


def test_ensure_without_download_raises_actionable(gz_source, tmp_path):
    raw_dir, _, _ = gz_source
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        ensure_raw_dataset("testds", str(empty), download=False)
    msg = str(exc.value)
    assert "not found" in msg
    assert "--download" in msg
    assert "stage('data', download=True)" in msg


def test_ensure_with_download(gz_source):
    raw_dir, _, md5 = gz_source
    info = ensure_raw_dataset("testds", str(raw_dir), download=True)
    assert info["verified"]
    assert info["hashes"]["md5"] == md5


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="no download source"):
        source_for("nope")


def test_archive_integrity_check_zip(tmp_path, monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("ratings.dat", "1::2::5::1\n")
    data = payload.getvalue()
    md5 = hashlib.md5(data).hexdigest()
    monkeypatch.setitem(dd.SOURCES, "ziptest", {
        "filename": "x.zip", "urls": ["http://127.0.0.1:1/x"], "md5": md5,
        "min_size_bytes": 1, "kind": "zip", "license_note": "test",
    })
    path = tmp_path / "x.zip"
    path.write_bytes(data)
    check = archive_integrity_check(str(path), "ziptest")
    assert check["ok"], check
    # corrupted zip member
    bad = tmp_path / "bad.zip"
    bad.write_bytes(data[:-8] + b"\x00" * 8)
    check = archive_integrity_check(str(bad), "ziptest")
    assert not check["ok"]


def test_sources_registry_real_datasets():
    for dataset in ("ml1m", "beauty"):
        src = source_for(dataset)
        assert src["urls"]
        assert src["min_size_bytes"] > 0
        assert src["kind"] in ("zip", "gzip")
        assert src["license_note"]
    # the registered GroupLens md5 is locked in the config as well
    from shaper.config import load_run_config

    cfg = load_run_config("ml1m")
    assert cfg.raw["dataset"]["expected_raw_zip_md5"] == source_for("ml1m")["md5"]
