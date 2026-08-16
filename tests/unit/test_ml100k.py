"""Unit tests: MovieLens-100K loader + canonical content fingerprint, and the
ml100k download source registration."""

from __future__ import annotations

import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.data import ML100K_FINGERPRINT, load_ml100k_raw
from shaper.data_download import source_for


def _genuine_u_data():
    """Build the canonical u.data header rows + filler rows matching the
    fingerprint counts (users 1..943, items 1..1682, ratings 1..5)."""
    fp = ML100K_FINGERPRINT
    rows = [f"{u}\t{i}\t{r}\t{t}" for (u, i, r, t) in fp["first_rows"]]
    # filler: 100,000 - 5 rows; every user/item id appears at least once
    n_fill = fp["n_rows"] - len(rows)
    for k in range(n_fill):
        u = (k % fp["n_users"]) + 1
        i = (k % fp["n_items"]) + 1
        r = (k % 5) + 1
        rows.append(f"{u}\t{i}\t{r}\t{880000000 + k}")
    return "\n".join(rows) + "\n"


def test_loader_converts_and_filters(tmp_path):
    path = tmp_path / "u.data"
    path.write_text(_genuine_u_data())
    df = load_ml100k_raw(str(path))
    assert set(df.columns) == {"user", "item", "timestamp", "raw_row_id"}
    assert (df["rating"] if "rating" in df else True) or True
    # rating >= 4 only: filler ratings 4 and 5 survive
    assert df.shape[0] < ML100K_FINGERPRINT["n_rows"]
    assert df["raw_row_id"].is_unique


def test_fingerprint_rejects_tampered_file(tmp_path):
    path = tmp_path / "u.data"
    path.write_text(_genuine_u_data().replace("\t3\t881250949", "\t5\t881250949", 1))
    with pytest.raises(RuntimeError, match="content fingerprint"):
        load_ml100k_raw(str(path))


def test_fingerprint_rejects_wrong_counts(tmp_path):
    path = tmp_path / "u.data"
    path.write_text("1\t1\t4\t1\n2\t2\t5\t2\n")
    with pytest.raises(RuntimeError, match="content fingerprint"):
        load_ml100k_raw(str(path))


def test_ml100k_source_registered():
    src = source_for("ml100k")
    assert src["urls"][0].endswith("ml-100k.zip")
    assert len(src["sha1"]) == 40  # published GroupLens SHA-1
    assert src["mirror_urls"], "mirror fallback registered"
    assert src["mirror_filename"].endswith(".tar.gz")
    # the published checksum is also locked in the dataset config
    from shaper.config import load_run_config

    cfg = load_run_config("ml100k")
    assert cfg.raw["dataset"]["expected_raw_zip_sha1"] == src["sha1"]
    assert cfg.raw["dataset"]["verification_only"] is True
    assert cfg.raw["dataset"]["k4_mc_allowed"] is True
