"""Regression: baseline subtraction must use the RAW role metric, never the
(possibly unset) value fields. This pins a bug where every value collapsed to
zero because the value fields were read before being written."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from shaper.game import CoalitionValueRecord, apply_baseline_subtraction, subtract_baseline


def _record(coalition, ndcg, nll):
    return CoalitionValueRecord(
        dataset="synthetic", seed=2001, policy="game_a",
        coalition=tuple(coalition),
        metrics_by_role={"game": {"ndcg": ndcg, "nll": nll}},
    )


def test_baseline_subtraction_uses_raw_metrics():
    records = [
        _record([], 0.10, 1.0),
        _record(["crop"], 0.12, 0.9),
        _record(["mask"], 0.11, 0.95),
    ]
    values = apply_baseline_subtraction(records, metric="ndcg", role="game")
    assert values[()] == 0.0
    assert values[("crop",)] == pytest.approx(0.02)
    assert values[("mask",)] == pytest.approx(0.01)


def test_subtract_baseline_writes_value_fields():
    records = [
        _record([], 0.10, 1.0),
        _record(["crop"], 0.12, 0.9),
    ]
    subtract_baseline(records, metric="ndcg")
    assert records[0].value_ndcg == 0.0
    assert records[1].value_ndcg == pytest.approx(0.02)
    # NLL utility: v(C) = -NLL(C) + NLL(empty)
    subtract_baseline(records, metric="nll")
    assert records[1].value_nll == pytest.approx(0.1)
