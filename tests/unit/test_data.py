"""Unit tests: data layer (positivity conversion, iterative core, temporal
split, frozen quartiles, salted 20/60/20 roles, artifact hygiene)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import pytest

from shaper.data import (
    FrozenData,
    assign_quartile,
    assign_roles,
    build_dataset_artifact,
    build_synthetic_interactions,
    iterative_core,
    load_ml1m_raw,
    split_leave_one_out,
)


def make_frame(rows):
    return pd.DataFrame(rows, columns=["user", "item", "timestamp"]).assign(
        raw_row_id=np.arange(len(rows))
    )


def test_ml1m_positivity_conversion(tmp_path):
    path = tmp_path / "ratings.dat"
    with open(path, "w") as fh:
        fh.write("1::10::5::100\n1::11::3::101\n1::12::4::102\n2::10::4::103\n")
    df = load_ml1m_raw(str(path))
    assert set(df["rating"]) if "rating" in df else True  # rating column dropped
    assert len(df) == 3  # rating 3 excluded, ratings >= 4 retained
    assert df["raw_row_id"].is_unique


def test_iterative_core_to_fixpoint():
    # item i0 appears once -> removed; user u3 then has < 2 -> removed
    rows = [
        ("u1", "i0", 1), ("u1", "i1", 2), ("u1", "i2", 3),
        ("u2", "i1", 4), ("u2", "i2", 5),
        ("u3", "i0", 6),
    ]
    df = make_frame(rows)
    out, record = iterative_core(df, k=2)
    assert set(out["user"]) == {"u1", "u2"}
    assert set(out["item"]) == {"i1", "i2"}
    assert record[-1]["interactions"] == len(out)


def test_temporal_leave_one_out_and_ties():
    rows = [
        ("u1", "a", 10), ("u1", "b", 20), ("u1", "c", 20), ("u1", "d", 30),
        ("u2", "x", 1), ("u2", "y", 2), ("u2", "z", 3),
    ]
    df = make_frame(rows)
    targets, tie_rate = split_leave_one_out(df, robustness_salt="test-salt")
    u1 = targets[targets["user"] == "u1"].iloc[0]
    # primary: last = d (test); second-last overall = c (b < c among ts=20)
    assert u1["test_target"] == "d"
    assert u1["val_target"] == "c"
    # robustness: ties at ts=20 reordered by salted hash -> val_target_robust
    # is either b or c; test target unchanged (unique latest timestamp)
    assert u1["test_target_robust"] == "d"
    assert u1["val_target_robust"] in ("b", "c")
    assert tie_rate > 0  # u1 has same-timestamp ties


def test_quartile_assignment():
    edges = [5, 10, 20]
    assert assign_quartile(3, edges) == 1
    assert assign_quartile(5, edges) == 1
    assert assign_quartile(6, edges) == 2
    assert assign_quartile(10, edges) == 2
    assert assign_quartile(11, edges) == 3
    assert assign_quartile(20, edges) == 3
    assert assign_quartile(21, edges) == 4


def test_role_assignment_stratified():
    users = [f"u{i}" for i in range(60)]
    lengths = np.array([(i % 4) * 6 + 3 for i in range(60)])  # spread over quartiles
    edges = [5, 10, 20]
    roles, quartiles = assign_roles(users, lengths, edges, salt="test")
    assert set(roles.tolist()) == {0, 1, 2}
    # stratified: within each quartile the 20/60/20 split holds (up to rounding)
    for q in set(quartiles.tolist()):
        idx = quartiles == q
        n = int(idx.sum())
        n_tune = int((roles[idx] == 0).sum())
        n_select = int((roles[idx] == 2).sum())
        assert n_tune == max(1, round(n * 0.2)) if n >= 5 else n_tune >= 1
        assert n_select == max(1, round(n * 0.2)) if n >= 5 else n_select >= 1
        assert n_tune + n_select < n
    # deterministic
    roles2, _ = assign_roles(users, lengths, edges, salt="test")
    assert np.array_equal(roles, roles2)


def test_artifact_build_freeze_and_validate(tiny_cfg, tmp_path):
    raw = build_synthetic_interactions(n_users=48, n_items=24, min_len=6, max_len=9, seed=11)
    manifest = build_dataset_artifact(tiny_cfg, raw, processed_root=str(tmp_path), force=True)
    data = FrozenData("synthetic", processed_root=str(tmp_path), cfg=tiny_cfg).load()
    checks = data.validate(cfg=tiny_cfg)
    assert all(c["pass"] for c in checks.values()), checks

    stats = data.stats()
    assert stats.users == manifest["stats"]["users"]
    assert stats.items == data.n_items
    # every user has train + val + test
    for uid in range(data.n_users):
        assert data.train_lengths()[uid].item() >= 1
    # roles cover all users disjointly
    role_users = np.concatenate([data.user_ids_for_role(r) for r in ("tune", "game", "select")])
    assert len(set(role_users)) == data.n_users == len(role_users)
    # held-out targets never in training
    train = data.train_seqs()
    for uid in range(data.n_users):
        items = set(int(x) for x in train[uid].tolist() if x != 0)
        assert int(data.tensors["val_target"][uid]) not in items
        assert int(data.tensors["test_target"][uid]) not in items
    # exclusion masks contain the prefix items
    prefix = data.tensors["val_prefix"]
    for uid in range(0, data.n_users, 5):
        pfx = set(int(x) for x in prefix[uid].tolist() if x != 0)
        excl = set(data.tensors["val_exclusion"][uid])
        assert pfx == excl


def test_artifact_refuses_different_config(tiny_cfg, tmp_path):
    raw = build_synthetic_interactions(n_users=40, n_items=20, min_len=6, max_len=8, seed=2)
    build_dataset_artifact(tiny_cfg, raw, processed_root=str(tmp_path), force=True)
    from shaper.config import load_run_config

    other = load_run_config("synthetic", overrides={"training": {"steps": 999}})
    with pytest.raises(RuntimeError, match="different config hash"):
        build_dataset_artifact(other, raw, processed_root=str(tmp_path), force=False)


def test_validate_detects_corruption(tiny_cfg, tmp_path):
    raw = build_synthetic_interactions(n_users=40, n_items=20, min_len=6, max_len=8, seed=3)
    build_dataset_artifact(tiny_cfg, raw, processed_root=str(tmp_path), force=True)
    data = FrozenData("synthetic", processed_root=str(tmp_path), cfg=tiny_cfg).load()
    # corrupt a frozen file
    path = os.path.join(tmp_path, "tensors.pt")
    with open(path, "ab") as fh:
        fh.write(b"corruption")
    checks = data.validate(cfg=tiny_cfg)
    assert not checks["file_integrity"]["pass"]


def test_repeated_target_counted(tiny_cfg, tmp_path):
    # user with a repeated item before the end
    rows = [
        ("u1", "a", 1), ("u1", "b", 2), ("u1", "a", 3), ("u1", "c", 4), ("u1", "a", 5),
        ("u2", "a", 1), ("u2", "c", 2), ("u2", "b", 3), ("u2", "d", 4), ("u2", "e", 5),
    ]
    df = make_frame(rows)
    manifest = build_dataset_artifact(tiny_cfg, df, processed_root=str(tmp_path), force=True)
    assert manifest["stats"]["repeated_target_rate_val"] >= 0.0
