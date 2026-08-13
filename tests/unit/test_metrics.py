"""Unit tests: full-catalog ranking, deterministic ties, prefix filtering,
repeated-target handling, and metric identities (spec A.3/A.9 test 11)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import pytest

from shaper.metrics import EvaluationResult, evaluate_model, ndcg_at_k, rank_one_user


def test_rank_computation_hand_values():
    scores = torch.tensor([0.5, 0.1, 0.9, 0.2, 0.8])  # items 1..5
    r = rank_one_user(scores, target=3, exclusion=[], k=10)
    assert r.rank == 1  # item 3 has the highest score
    assert r.ndcg == pytest.approx(1.0)
    assert r.hr == 1.0
    assert r.mrr == pytest.approx(1.0)
    r2 = rank_one_user(scores, target=5, exclusion=[], k=10)
    assert r2.rank == 2
    assert r2.ndcg == pytest.approx(1 / math.log2(3))
    assert r2.mrr == pytest.approx(0.5)


def test_ties_broken_by_ascending_item_id():
    scores = torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])
    r = rank_one_user(scores, target=1, exclusion=[], k=10)
    assert r.rank == 1
    r5 = rank_one_user(scores, target=5, exclusion=[], k=10)
    assert r5.rank == 5  # tied scores -> smallest item id first


def test_prefix_filtering_and_repeated_target():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    # target 2 repeats in the prefix -> retained and counted
    r = rank_one_user(scores, target=2, exclusion=[2, 4], k=10)
    assert r.repeated_target is True
    assert r.rank == 2  # only item 4 excluded; item 1 (0.9) ranks above target
    # target 3 not in prefix, items 1 and 2 excluded -> target becomes rank 1
    r2 = rank_one_user(scores, target=3, exclusion=[1, 2], k=10)
    assert r2.rank == 1
    assert r2.repeated_target is False


def test_excluded_items_never_rank_above_target():
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
    r = rank_one_user(scores, target=4, exclusion=[1, 2, 3], k=10)
    assert r.rank == 1  # everything better is excluded


def test_ndcg_beyond_k_is_zero():
    scores = torch.tensor([0.5, 0.9, 0.4, 0.3, 0.2])
    r = rank_one_user(scores, target=5, exclusion=[], k=3)
    assert r.ndcg == 0.0 and r.hr == 0.0


def test_full_catalog_nll():
    scores = torch.tensor([0.0, 0.0, 5.0, 0.0, 0.0])
    r = rank_one_user(scores, target=3, exclusion=[], k=10)
    p = torch.softmax(scores - scores.max(), dim=0)
    assert r.nll == pytest.approx(float(-torch.log(p[2])), rel=1e-5)


def test_evaluate_model_end_to_end(tiny_data):
    from shaper.backbone import build_model
    from shaper.config import load_run_config

    cfg = load_run_config("synthetic")
    model = build_model(cfg, tiny_data.n_items)
    inputs = tiny_data.eval_inputs("game")
    res = evaluate_model(model, inputs, k=10)
    assert res.n_users == len(inputs["users"])
    assert len(res.per_user_ndcg) == res.n_users
    assert 0.0 <= res.ndcg <= 1.0
    assert 0.0 <= res.repeated_target_rate <= 1.0
    # HR@10 == mean of per-user HR
    assert res.hr == pytest.approx(sum(res.per_user_hr) / res.n_users)


def test_ndcg_at_k_helper():
    ranks = [1, 2, 15, 4]
    vals = ndcg_at_k(ranks, k=10)
    assert vals[0] == 1.0
    assert vals[2] == 0.0
    assert vals[1] == pytest.approx(1 / math.log2(3))


def test_evaluation_result_defaults():
    r = EvaluationResult()
    assert r.n_users == 0 and r.ndcg == 0.0
