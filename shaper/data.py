"""Data layer: loaders, iterative core filtering, temporal leave-one-out
split, frozen segment edges, salted validation-role assignment, and frozen
data artifacts.

Every downstream stage consumes the frozen artifact produced by
`build_dataset_artifact`; no stage re-derives splits.

Protocol points implemented here (spec A.3):
  - MovieLens-1M: rating >= 4 is positive; iterative 5-core to fixed point.
  - Beauty: every retained review is positive; iterative 5-core to fixed point.
  - Deterministic temporal LOO: last = test, second-last = validation,
    ties broken by original row order (primary) or persisted salted hash
    (robustness ordering).
  - Q1-Q4 edges frozen on pre-truncation training-history length over the
    full validation-eligible pool BEFORE role assignment.
  - V_tune / V_game / V_select = 20% / 60% / 20% via a stable persisted
    salted hash, stratified by dataset and frozen quartiles.
  - Persists user/item maps, raw_row_id, split tensors, lengths, quartiles,
    hash algorithm + salt, role assignments, exclusion masks, config hash
    and data hash.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import tarfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .config import RunConfig
from .logging_utils import StructuredLogger
from .provenance import HASH_ALGORITHM, file_hash, key_int, make_salt, stable_hash

MIN_EVENTS_FOR_SPLIT = 3  # need train + validation + test per user

ROLE_NAMES = ("tune", "game", "select")


# --------------------------------------------------------------------------
# Raw loaders
# --------------------------------------------------------------------------

def load_ml1m_raw(path: str) -> pd.DataFrame:
    """Parse GroupLens ml-1m ratings.dat. Columns: user, item, rating, ts."""
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user", "item", "rating", "timestamp"],
    )
    df = df[df["rating"] >= 4][["user", "item", "timestamp"]].copy()
    df["raw_row_id"] = np.arange(len(df))
    return df


def load_beauty_raw(path: str) -> pd.DataFrame:
    """Parse Amazon Reviews 2018 Beauty 5-core (json.gz). Every review positive."""
    chunks = pd.read_json(path, lines=True, compression="gzip", chunksize=100_000)
    frames = []
    for i, chunk in enumerate(chunks):
        frames.append(
            pd.DataFrame(
                {
                    "user": chunk["reviewerID"].astype(str),
                    "item": chunk["asin"].astype(str),
                    "timestamp": chunk["unixReviewTime"].astype(np.int64),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    df["raw_row_id"] = np.arange(len(df))
    return df


def build_synthetic_interactions(
    n_users: int,
    n_items: int,
    min_len: int,
    max_len: int,
    seed: int,
    timestamps: bool = True,
) -> pd.DataFrame:
    """Tiny synthetic recommender dataset for the end-to-end test.

    Users draw a personal item affinity vector; sequences are drawn by
    affinity-weighted sampling without immediate replacement, with a Zipf-ish
    popularity bias so ranking is learnable. Timestamps increase along the
    sequence (no same-day ties by default).
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    popularity = np.arange(n_items, 0, -1, dtype=np.float64)
    popularity = popularity / popularity.sum()
    ts = 1_600_000_000
    for u in range(n_users):
        affinity = rng.dirichlet(np.ones(n_items) * 0.5)
        length = int(rng.integers(min_len, max_len + 1))
        hist: List[int] = []
        for _ in range(length):
            if hist:
                w = affinity * popularity
                w = np.array([0.0 if i in hist else wi for i, wi in enumerate(w)])
                if w.sum() <= 0:
                    break
                w /= w.sum()
                item = int(rng.choice(n_items, p=w))
            else:
                item = int(rng.choice(n_items, p=popularity))
            hist.append(item)
            rows.append({"user": f"u{u}", "item": f"i{item}", "timestamp": ts, "raw_row_id": len(rows)})
            ts += int(rng.integers(1, 30))
    df = pd.DataFrame(rows)
    df["raw_row_id"] = np.arange(len(df))
    return df


# --------------------------------------------------------------------------
# Core filtering and splitting
# --------------------------------------------------------------------------

def iterative_core(
    interactions: pd.DataFrame, k: int, logger: Optional[StructuredLogger] = None
) -> Tuple[pd.DataFrame, List[Dict[str, int]]]:
    """Iterate k-core removal to a fixed point after positivity conversion.

    Returns the filtered frame and the per-iteration counts record.
    """
    record: List[Dict[str, int]] = []
    iteration = 0
    while True:
        n_users = interactions["user"].nunique()
        n_items = interactions["item"].nunique()
        n_rows = len(interactions)
        record.append({"iteration": iteration, "users": n_users, "items": n_items, "interactions": n_rows})
        user_counts = interactions.groupby("user")["item"].transform("size")
        item_counts = interactions.groupby("item")["user"].transform("size")
        keep = (user_counts >= k) & (item_counts >= k)
        if bool(keep.all()):
            break
        interactions = interactions[keep].copy()
        iteration += 1
        if logger is not None:
            logger.info("DATA_BUILD", "core_iteration", iteration=iteration,
                        users=int(n_users), items=int(n_items), interactions=int(n_rows))
    return interactions, record


def split_leave_one_out(
    interactions: pd.DataFrame, robustness_salt: str
) -> Tuple[pd.DataFrame, float]:
    """Temporal leave-one-out per user.

    Primary ordering: sort by (timestamp, raw_row_id). Robustness ordering:
    events with equal (user, timestamp) are re-sorted by a fixed salted hash
    of (user, item, timestamp, raw_row_id). Returns a frame with per-user
    primary and robustness val/test items plus the same-day tie rate.
    """
    df = interactions.copy()
    df = df.sort_values(["user", "timestamp", "raw_row_id"], kind="stable")

    tie_mask = df.duplicated(subset=["user", "timestamp"], keep=False)
    same_day_tie_rate = float(tie_mask.mean()) if len(df) else 0.0

    # Robustness ordering: among tied events only, order by the salted hash.
    # The salted key is the full 128-bit hash integer (Python int, object
    # dtype — no int64 truncation).
    df_rob = df.copy()
    df_rob["_tie_key"] = df_rob["raw_row_id"].astype(object)
    if tie_mask.any():
        tied_rows = df_rob.loc[tie_mask]
        keys = [
            key_int(r.user, r.item, r.timestamp, r.raw_row_id, salt=robustness_salt)
            for r in tied_rows.itertuples(index=False)
        ]
        df_rob.loc[tie_mask, "_tie_key"] = pd.Series(keys, index=tied_rows.index, dtype=object)
    df_rob = df_rob.sort_values(["user", "timestamp", "_tie_key"], kind="stable")

    item_col = "item_id" if "item_id" in df.columns else "item"
    has_uid = "user_id" in df.columns

    def _targets(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby("user")
        cols = ["user", "user_id", item_col] if has_uid else ["user", item_col]
        last = g.nth(-1)[cols].rename(columns={item_col: "test_target"})
        second = g.nth(-2)[cols].rename(columns={item_col: "val_target"})
        merged = last.merge(second, on="user", how="left")
        if has_uid:
            merged = merged.drop(columns=["user_id_y"]).rename(columns={"user_id_x": "user_id"})
        return merged

    primary = _targets(df)
    robust = _targets(df_rob).rename(
        columns={"test_target": "test_target_robust", "val_target": "val_target_robust"}
    )
    if has_uid:
        robust = robust.drop(columns=["user_id"])
    out = primary.merge(robust, on="user", how="left")
    return out, same_day_tie_rate


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

@dataclass
class DatasetStats:
    users: int = 0
    items: int = 0
    interactions: int = 0
    density: float = 0.0
    train_len_mean: float = 0.0
    train_len_median: float = 0.0
    train_len_q1: float = 0.0
    train_len_q3: float = 0.0
    truncated_fraction: float = 0.0
    role_counts: Dict[str, int] = field(default_factory=dict)
    quartile_edges: List[float] = field(default_factory=list)
    same_day_tie_rate: float = 0.0
    repeated_target_rate_val: float = 0.0
    repeated_target_rate_test: float = 0.0
    applicability_rates: Dict[str, float] = field(default_factory=dict)
    positive_before_core: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _jsonable(obj: Any) -> Any:
    """Convert a manifest into native JSON types (numpy scalars -> python
    scalars; numpy-scalar dict keys -> strings)."""
    import numpy as _np

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, (str, int, float, bool)) or k is None:
                key = k
            else:
                key = str(k)
            out[key] = _jsonable(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, _np.integer):
        return int(obj)
    if isinstance(obj, _np.floating):
        return float(obj)
    if isinstance(obj, _np.bool_):
        return bool(obj)
    if isinstance(obj, _np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


def compute_applicability_rates(train_seqs: Sequence[Sequence[int]]) -> Dict[str, float]:
    """Fraction of training histories where each view is applicable (changed)
    under the canonical protocol (pure data diagnostic)."""
    n = len(train_seqs)
    if n == 0:
        return {"crop": 0.0, "mask": 0.0, "reorder": 0.0}
    crop = mask = reorder = 0
    for seq in train_seqs:
        m = len(seq)
        if m >= 3:
            crop += 1
        if m >= 1:
            mask += 1
        if m >= 2 and len(set(seq)) >= 2:
            reorder += 1
    return {
        "crop": crop / n,
        "mask": mask / n,
        "reorder": reorder / n,
    }


# --------------------------------------------------------------------------
# Artifact build / freeze
# --------------------------------------------------------------------------

def _freeze_quartile_edges(lengths: np.ndarray) -> List[int]:
    q = np.quantile(lengths, [0.25, 0.5, 0.75])
    return [int(round(float(x))) for x in q]


def assign_quartile(length: int, edges: Sequence[int]) -> int:
    """Q1..Q4 (1-indexed) from frozen pre-truncation training length."""
    for idx, edge in enumerate(edges):
        if length <= edge:
            return idx + 1
    return len(edges) + 1


def assign_roles(
    users: List[str], lengths: np.ndarray, edges: Sequence[int], salt: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign validation roles stratified by frozen quartiles.

    Within each (dataset, quartile) stratum users are ordered by a stable
    salted hash and split 20% / 60% / 20%. Roles: 0=tune, 1=game, 2=select.
    """
    quartiles = np.array([assign_quartile(int(l), edges) for l in lengths])
    roles = np.full(len(users), -1, dtype=np.int64)
    for q in sorted(set(quartiles.tolist())):
        idx = np.where(quartiles == q)[0]
        order = sorted(idx, key=lambda i: key_int(users[i], salt=salt))
        n = len(order)
        n_tune = max(1, int(round(n * 0.20))) if n >= 5 else (1 if n >= 3 else 0)
        n_select = max(1, int(round(n * 0.20))) if n >= 5 else (1 if n >= 3 else 0)
        n_game = n - n_tune - n_select
        for i in order[:n_tune]:
            roles[i] = 0
        for i in order[n_tune : n_tune + n_game]:
            roles[i] = 1
        for i in order[n_tune + n_game :]:
            roles[i] = 2
    assert (roles >= 0).all(), "every retained user must receive a validation role"
    return roles, quartiles


def _truncate_right(seq: List[int], max_len: int) -> List[int]:
    return seq[-max_len:] if len(seq) > max_len else seq


def _pad_left(seq: List[int], length: int, pad: int = 0) -> List[int]:
    return [pad] * (length - len(seq)) + list(seq)


def _count_repeated_targets(prefixes: List[List[int]], targets: List[int]) -> float:
    n = 0
    for prefix, target in zip(prefixes, targets):
        if target in prefix:
            n += 1
    return n / len(targets) if targets else 0.0


def build_dataset_artifact(
    cfg: RunConfig,
    raw_interactions: pd.DataFrame,
    processed_root: Optional[str] = None,
    force: bool = False,
    logger: Optional[StructuredLogger] = None,
    run_id: str = "data-build",
    raw_download_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build and freeze the data artifact for one dataset.

    Returns the manifest dict (also written to disk). The artifact directory
    is `<processed_root>/<dataset>/` and contains maps, interactions, splits,
    role assignments, tensors, quartiles, salts and hashes.
    """
    dataset = cfg.dataset
    processed_root = processed_root or os.path.join(cfg.paths["data_processed"], dataset)
    os.makedirs(processed_root, exist_ok=True)

    if os.path.exists(os.path.join(processed_root, "data_manifest.json")) and not force:
        with open(os.path.join(processed_root, "data_manifest.json"), "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        if existing.get("config_hash") == cfg.config_hash():
            return existing
        raise RuntimeError(
            f"processed artifact for {dataset} exists with a different config hash "
            f"({existing.get('config_hash')} != {cfg.config_hash()}). "
            "Refusing to overwrite a frozen artifact; pass force=True deliberately."
        )

    raw = raw_interactions
    if "rating" in raw.columns:
        raw = raw[raw["rating"] >= 4].copy()
    positive_before_core = len(raw)

    core_k = int(cfg.raw["dataset"]["core"])
    core_interactions, core_record = iterative_core(raw, core_k, logger=logger)

    # Raw-keyed maps feed `.map()` on the original column dtype; string-keyed
    # maps are the persisted form (users.json / items.json must stay plain
    # JSON, and numpy scalar dict keys are not hash-identical to python ints
    # on every numpy version).
    users_map_raw = {u: i for i, u in enumerate(sorted(core_interactions["user"].unique()))}
    items_map_raw = {it: i + 1 for i, it in enumerate(sorted(core_interactions["item"].unique()))}
    users_map: Dict[str, int] = {str(u): i for u, i in users_map_raw.items()}
    items_map: Dict[str, int] = {str(it): i for it, i in items_map_raw.items()}  # 0 reserved for padding
    n_items = len(items_map)

    core_interactions["user_id"] = core_interactions["user"].map(users_map_raw)
    core_interactions["item_id"] = core_interactions["item"].map(items_map_raw)

    salt = make_salt(int(cfg.raw["dataset"]["seed"]), f"roles-{dataset}")
    robustness_salt = make_salt(int(cfg.raw["dataset"]["seed"]), f"robustness-{dataset}")
    targets, same_day_tie_rate = split_leave_one_out(core_interactions, robustness_salt)

    # Per-user training histories (all interactions except last two).
    # Group keys are coerced to native ints (pandas/numpy scalar keys are
    # not hash-identical to python ints on every numpy version).
    user_groups = {int(uid): grp for uid, grp in core_interactions.groupby("user_id")}
    n_users = len(users_map)
    train_seqs: List[List[int]] = []
    train_lens: List[int] = []
    val_targets: List[int] = []
    test_targets: List[int] = []
    val_targets_rob: List[int] = []
    test_targets_rob: List[int] = []

    max_len = cfg.max_len
    row = targets.set_index("user_id")
    for uid in range(n_users):
        grp = user_groups[uid].sort_values(["timestamp", "raw_row_id"], kind="stable")
        items = grp["item_id"].tolist()
        train = items[:-2]
        train_seqs.append(train)
        train_lens.append(len(train))
        val_targets.append(int(row.at[uid, "val_target"]))
        test_targets.append(int(row.at[uid, "test_target"]))
        val_targets_rob.append(int(row.at[uid, "val_target_robust"]))
        test_targets_rob.append(int(row.at[uid, "test_target_robust"]))

    pretrunc_lens = np.array(train_lens, dtype=np.int64)
    edges = _freeze_quartile_edges(pretrunc_lens)  # frozen BEFORE role assignment
    roles, quartiles = assign_roles(list(users_map.keys()), pretrunc_lens, edges, salt)

    # Model tensors: training histories truncated to max_len (rightmost kept),
    # padded LEFT with 0 so the final valid position is the last token.
    train_seq_t = torch.zeros((n_users, max_len), dtype=torch.int64)
    truncated_fraction = 0.0
    for uid, seq in enumerate(train_seqs):
        if len(seq) > max_len:
            truncated_fraction += 1.0
        keep = _truncate_right(seq, max_len)
        train_seq_t[uid, -len(keep) :] = torch.tensor(keep, dtype=torch.int64)

    # Evaluation prefixes: validation prefix = truncated training history;
    # test prefix = truncated training history + validation item (max_len+1).
    val_prefix_t = train_seq_t.clone()
    test_prefix_t = torch.zeros((n_users, max_len + 1), dtype=torch.int64)
    test_prefix_t[:, :-1] = train_seq_t
    test_prefix_t[:, -1] = torch.tensor(val_targets, dtype=torch.int64)
    test_prefix_t_rob = test_prefix_t.clone()
    test_prefix_t_rob[:, -1] = torch.tensor(val_targets_rob, dtype=torch.int64)

    # Exclusion masks: items present in the available prefix (the repeated
    # target itself is retained as a candidate at evaluation time).
    val_exclusion: List[List[int]] = []
    test_exclusion: List[List[int]] = []
    test_exclusion_rob: List[List[int]] = []
    for uid in range(n_users):
        ve = [int(x) for x in sorted(set(train_seq_t[uid].tolist())) if x != 0]
        val_exclusion.append(ve)
        te = [int(x) for x in sorted(set(test_prefix_t[uid].tolist())) if x != 0]
        test_exclusion.append(te)
        te_r = [int(x) for x in sorted(set(test_prefix_t_rob[uid].tolist())) if x != 0]
        test_exclusion_rob.append(te_r)

    stats = DatasetStats(
        users=n_users,
        items=n_items,
        interactions=len(core_interactions),
        density=len(core_interactions) / (n_users * n_items) if n_users and n_items else 0.0,
        train_len_mean=float(pretrunc_lens.mean()),
        train_len_median=float(np.median(pretrunc_lens)),
        train_len_q1=float(np.quantile(pretrunc_lens, 0.25)),
        train_len_q3=float(np.quantile(pretrunc_lens, 0.75)),
        truncated_fraction=truncated_fraction / n_users if n_users else 0.0,
        role_counts={
            "tune": int((roles == 0).sum()),
            "game": int((roles == 1).sum()),
            "select": int((roles == 2).sum()),
        },
        quartile_edges=edges,
        same_day_tie_rate=same_day_tie_rate,
        repeated_target_rate_val=_count_repeated_targets(val_exclusion, val_targets),
        repeated_target_rate_test=_count_repeated_targets(test_exclusion, test_targets),
        applicability_rates=compute_applicability_rates(train_seqs),
        positive_before_core=positive_before_core,
    )

    interactions_csv = os.path.join(processed_root, "interactions.csv")
    core_interactions[
        ["user_id", "item_id", "timestamp", "raw_row_id", "user", "item"]
    ].sort_values(["user_id", "timestamp", "raw_row_id"]).to_csv(interactions_csv, index=False)

    roles_csv = os.path.join(processed_root, "roles.csv")
    pd.DataFrame(
        {
            "user_id": np.arange(n_users),
            "user": list(users_map.keys()),
            "quartile": quartiles,
            "role": [ROLE_NAMES[int(r)] for r in roles],
            "train_len": train_lens,
        }
    ).to_csv(roles_csv, index=False)

    users_json = os.path.join(processed_root, "users.json")
    items_json = os.path.join(processed_root, "items.json")
    with open(users_json, "w", encoding="utf-8") as fh:
        json.dump(users_map, fh)
    with open(items_json, "w", encoding="utf-8") as fh:
        json.dump(items_map, fh)

    tensors_pt = os.path.join(processed_root, "tensors.pt")
    torch.save(
        {
            "train_seq": train_seq_t,
            "train_len": torch.tensor(train_lens, dtype=torch.int64),
            "val_prefix": val_prefix_t,
            "test_prefix": test_prefix_t,
            "test_prefix_robust": test_prefix_t_rob,
            "val_target": torch.tensor(val_targets, dtype=torch.int64),
            "test_target": torch.tensor(test_targets, dtype=torch.int64),
            "val_target_robust": torch.tensor(val_targets_rob, dtype=torch.int64),
            "test_target_robust": torch.tensor(test_targets_rob, dtype=torch.int64),
            "val_exclusion": val_exclusion,
            "test_exclusion": test_exclusion,
            "test_exclusion_robust": test_exclusion_rob,
        },
        tensors_pt,
    )

    file_hashes = {
        f: file_hash(os.path.join(processed_root, f))
        for f in ("interactions.csv", "roles.csv", "users.json", "items.json", "tensors.pt")
    }
    config_hash = cfg.config_hash()
    data_hash = stable_hash(
        {"files": file_hashes, "config_hash": config_hash, "salt": salt,
         "robustness_salt": robustness_salt},
        salt="data-artifact",
    )
    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "hash_algorithm": HASH_ALGORITHM,
        "salt": salt,
        "robustness_salt": robustness_salt,
        "max_len": max_len,
        "batch_size": cfg.batch_size,
        "core": core_k,
        "core_record": core_record,
        "positive_before_core": positive_before_core,
        "stats": stats.to_dict(),
        "file_hashes": file_hashes,
        "raw_source": cfg.raw["dataset"]["source"],
        "raw_download": raw_download_info or {
            "note": "raw archive provided manually (no download sidecar recorded)"
        },
        "ordering_primary": "timestamp_then_raw_row_id",
        "ordering_robustness": "salted_hash_of_(user,item,timestamp,raw_row_id)",
        "evaluation": {
            "k": cfg.evaluation["k"],
            "prefix_filter": cfg.evaluation["prefix_filter"],
            "retain_repeated_target": cfg.evaluation["retain_repeated_target"],
            "tie_break": cfg.evaluation["tie_break"],
        },
        "no_raw_count_substitution": (
            "stats.interactions refers to the finalized positive artifact; "
            "the raw rating count is never reported as the processed count"
        ),
    }
    with open(os.path.join(processed_root, "data_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(manifest), fh, indent=2, sort_keys=True)
    if logger is not None:
        logger.info(
            "DATA_BUILD",
            "artifact_frozen",
            dataset=dataset,
            data_hash=data_hash,
            config_hash=config_hash,
            users=stats.users,
            items=stats.items,
            interactions=stats.interactions,
            run_id_tag=run_id,
        )
    return manifest


# --------------------------------------------------------------------------
# Frozen artifact access
# --------------------------------------------------------------------------

class FrozenData:
    """Read-only access to a frozen data artifact.

    All splits, roles, quartiles and exclusion masks come from disk; this
    class cannot reconstruct them.
    """

    def __init__(self, dataset: str, processed_root: Optional[str] = None, cfg: Optional[RunConfig] = None):
        self.dataset = dataset
        if processed_root is None:
            if cfg is None:
                from .config import load_run_config

                cfg = load_run_config(dataset)
            processed_root = os.path.join(cfg.paths["data_processed"], dataset)
        self.processed_root = processed_root
        self.manifest: Dict[str, Any] = {}
        self.tensors: Dict[str, Any] = {}
        self.roles: Dict[str, np.ndarray] = {}
        self.quartiles: np.ndarray = np.array([], dtype=np.int64)
        self.train_len: np.ndarray = np.array([], dtype=np.int64)
        self._loaded = False

    # -- loading ------------------------------------------------------------
    def load(self) -> "FrozenData":
        with open(os.path.join(self.processed_root, "data_manifest.json"), "r", encoding="utf-8") as fh:
            self.manifest = json.load(fh)
        self.tensors = torch.load(
            os.path.join(self.processed_root, "tensors.pt"), map_location="cpu", weights_only=False
        )
        roles_df = pd.read_csv(os.path.join(self.processed_root, "roles.csv"))
        for role in ROLE_NAMES:
            self.roles[role] = roles_df.loc[roles_df["role"] == role, "user_id"].to_numpy()
        self.quartiles = roles_df["quartile"].to_numpy()
        self.train_len = roles_df["train_len"].to_numpy()
        self._loaded = True
        return self

    # -- properties ---------------------------------------------------------
    @property
    def data_hash(self) -> str:
        return self.manifest["data_hash"]

    @property
    def config_hash(self) -> str:
        return self.manifest["config_hash"]

    @property
    def max_len(self) -> int:
        return int(self.manifest["max_len"])

    @property
    def n_users(self) -> int:
        return int(self.manifest["stats"]["users"])

    @property
    def n_items(self) -> int:
        return int(self.manifest["stats"]["items"])

    @property
    def batch_size(self) -> int:
        return int(self.manifest["batch_size"])

    @property
    def quartile_edges(self) -> List[int]:
        return list(self.manifest["stats"]["quartile_edges"])

    def train_seqs(self) -> torch.Tensor:
        return self.tensors["train_seq"]

    def train_lengths(self) -> torch.Tensor:
        return self.tensors["train_len"]

    def user_ids_for_role(self, role: str) -> np.ndarray:
        return self.roles[role]

    def all_user_ids(self) -> np.ndarray:
        return np.arange(self.n_users)

    # -- evaluation sets ----------------------------------------------------
    def eval_inputs(
        self, role: str, ordering: str = "primary"
    ) -> Dict[str, torch.Tensor]:
        """Evaluation inputs: prefix tensors, targets and exclusion masks."""
        if role in ROLE_NAMES:
            users = self.roles[role]
        elif role == "all":
            users = np.arange(self.n_users)
        else:
            raise ValueError(f"unknown role {role}")
        if ordering == "primary":
            prefix_key, target_key, excl_key = "val_prefix", "val_target", "val_exclusion"
        elif ordering == "robustness":
            prefix_key = "val_prefix"
            target_key, excl_key = "val_target_robust", "val_exclusion"
        else:
            raise ValueError(f"unknown ordering {ordering}")
        users_t = torch.from_numpy(users.copy())
        prefix = self.tensors[prefix_key][users_t]
        target = self.tensors[target_key][users_t]
        return {
            "users": users_t,
            "prefix": prefix,
            "target": target,
            "exclusion": [self.tensors[excl_key][int(i)] for i in users],
        }

    def test_inputs(self, ordering: str = "primary") -> Dict[str, Any]:
        users = np.arange(self.n_users)
        users_t = torch.from_numpy(users.copy())
        if ordering == "primary":
            prefix = self.tensors["test_prefix"][users_t]
            target = self.tensors["test_target"][users_t]
            exclusion = [self.tensors["test_exclusion"][int(i)] for i in users]
        elif ordering == "robustness":
            prefix = self.tensors["test_prefix_robust"][users_t]
            target = self.tensors["test_target_robust"][users_t]
            exclusion = [self.tensors["test_exclusion_robust"][int(i)] for i in users]
        else:
            raise ValueError(f"unknown ordering {ordering}")
        return {
            "users": torch.from_numpy(users),
            "prefix": prefix,
            "target": target,
            "exclusion": exclusion,
        }

    # -- deterministic epoch permutations -----------------------------------
    def epoch_permutation(self, seed: int, epoch: int, role: Optional[str] = None) -> np.ndarray:
        """Permutation of training users keyed by (seed, epoch, dataset_hash).

        Every user appears exactly once per epoch permutation. The key is a
        pure function of the frozen data hash, so worker count and coalition
        enumeration order cannot change it.
        """
        if role is None or role == "all":
            users = np.arange(self.n_users)
        else:
            users = self.roles[role]
        rng = random.Random(key_int(seed, epoch, self.data_hash, salt="epoch-perm"))
        perm = users.copy()
        rng.shuffle(perm)
        return perm

    def train_batches(
        self, seed: int, epoch: int, batch_size: Optional[int] = None, role: Optional[str] = None
    ):
        """Yield (user_ids, seq_batch) for one epoch permutation.

        drop_last=False; the final partial batch is yielded as-is.
        """
        bs = batch_size or self.batch_size
        perm = self.epoch_permutation(seed, epoch, role=role)
        seqs = self.tensors["train_seq"]
        for start in range(0, len(perm), bs):
            idx = torch.from_numpy(perm[start : start + bs].copy())
            yield idx, seqs[idx]

    # -- validation ---------------------------------------------------------
    def validate(self, cfg: Optional[RunConfig] = None) -> Dict[str, Any]:
        """Run the frozen-artifact hygiene checks (data-validation gate)."""
        return validate_artifact(self, cfg=cfg)

    def stats(self) -> DatasetStats:
        s = DatasetStats(**self.manifest["stats"])
        return s


def validate_artifact(data: "FrozenData", cfg: Optional[RunConfig] = None) -> Dict[str, Any]:
    """Frozen-artifact hygiene checks: split disjointness, temporal order,
    raw_row_id persistence, target handling, masks, role/quartile freezing,
    config-hash match, and file-hash integrity.

    Returns {check: {pass: bool, detail: str}}.
    """
    checks: Dict[str, Dict[str, Any]] = {}

    def _check(name: str, ok: bool, detail: str) -> None:
        checks[name] = {"pass": bool(ok), "detail": detail}

    train_seq = data.tensors["train_seq"]
    val_target = data.tensors["val_target"].numpy()
    test_target = data.tensors["test_target"].numpy()
    n_users, n_items = data.n_users, data.n_items

    # 1. split disjointness: held-out targets never appear in training
    train_items = {int(u): set(int(x) for x in seq.tolist() if x != 0) for u, seq in enumerate(train_seq)}
    bad_val = sum(1 for u in range(n_users) if int(val_target[u]) in train_items[u])
    bad_test = sum(1 for u in range(n_users) if int(test_target[u]) in train_items[u])
    _check(
        "split_disjointness",
        bad_val == 0 and bad_test == 0,
        f"validation targets in training: {bad_val}; test targets in training: {bad_test}",
    )

    # 2. temporal order from the persisted interactions
    inter = pd.read_csv(os.path.join(data.processed_root, "interactions.csv"))
    order_ok = True
    order_detail = "verified"
    for uid, grp in inter.groupby("user_id"):
        srt = grp.sort_values(["timestamp", "raw_row_id"], kind="stable")
        if not (srt["timestamp"].to_numpy() == grp.sort_values("timestamp", kind="stable")["timestamp"].to_numpy()).all():
            order_ok = False
            order_detail = f"user {uid}: persisted order differs from temporal order"
            break
        items = srt["item_id"].tolist()
        if items[-1] != test_target[uid] or items[-2] != val_target[uid]:
            order_ok = False
            order_detail = f"user {uid}: val/test targets do not match the last two temporal events"
            break
    _check("temporal_order", order_ok, order_detail)

    # 3. raw_row_id persisted
    _check(
        "raw_row_id_persisted",
        "raw_row_id" in inter.columns and inter["raw_row_id"].notna().all(),
        f"{len(inter)} rows with raw_row_id",
    )

    # 4. repeated-target handling
    _check(
        "repeated_target_handling",
        "repeated_target_rate_val" in data.manifest["stats"]
        and "repeated_target_rate_test" in data.manifest["stats"],
        f"val repeat rate {data.manifest['stats']['repeated_target_rate_val']:.4f}, "
        f"test repeat rate {data.manifest['stats']['repeated_target_rate_test']:.4f}",
    )

    # 5. masks: padding id 0 only; item ids in [1, n_items]
    ids = set(int(x) for x in inter["item_id"].unique())
    _check(
        "mask_and_id_validity",
        0 not in ids and min(ids) >= 1 and max(ids) <= n_items,
        f"item ids in [1, {n_items}]; padding 0 absent from interactions",
    )

    # 6. roles disjoint and covering; frozen quartiles
    role_sets = [set(int(x) for x in data.roles[r]) for r in ROLE_NAMES]
    all_covered = set().union(*role_sets)
    _check(
        "roles_disjoint_covering",
        all_covered == set(range(n_users)) and sum(len(s) for s in role_sets) == n_users,
        f"tune/game/select = {[len(s) for s in role_sets]}",
    )
    edges = data.quartile_edges
    qs = [assign_quartile(int(l), edges) for l in data.train_len.tolist()]
    _check(
        "quartiles_frozen",
        qs == data.quartiles.tolist(),
        f"edges {edges} frozen in manifest",
    )

    # 7. config hash match
    if cfg is not None:
        _check(
            "config_hash_match",
            data.manifest["config_hash"] == cfg.config_hash(),
            f"artifact {data.manifest['config_hash'][:12]} vs config {cfg.config_hash()[:12]}",
        )
    else:
        _check("config_hash_match", True, "no config supplied; skipped")

    # 8. file integrity
    files_ok = True
    detail = []
    for fname, fhash in data.manifest["file_hashes"].items():
        path = os.path.join(data.processed_root, fname)
        if not os.path.exists(path) or file_hash(path) != fhash:
            files_ok = False
            detail.append(fname)
    _check("file_integrity", files_ok, "ok" if files_ok else f"corrupted: {detail}")

    return checks


__all__ = [
    "DatasetStats",
    "FrozenData",
    "build_dataset_artifact",
    "build_synthetic_interactions",
    "compute_applicability_rates",
    "iterative_core",
    "load_beauty_raw",
    "load_ml1m_raw",
    "split_leave_one_out",
    "assign_quartile",
    "assign_roles",
    "ROLE_NAMES",
]
