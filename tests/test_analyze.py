"""Tests for the L3b Phase-2 read-only analyses (sync.analyze).

Use the shared ``FakeGraphClient`` (canned ``run_read`` keyed on the exact query
strings). NO live Neo4j is hit. Covers the calibration-report pipeline (randomized
PIT from result_percentile + surprisal, overall + by discipline, deterministic
under a fixed seed) and the weight-fit pipeline (rows → WeightSamples → fit_weights),
plus the graceful empty paths and the outcome allow-list.
"""

from __future__ import annotations

import math

import pytest

from sync.analyze import (
    CALIBRATION_QUERY,
    OUTCOME_FIELDS,
    WEIGHTFIT_QUERY,
    build_calibration_report,
    build_weightfit_report,
)
from tests.conftest import FakeGraphClient

# A spread of (result_percentile, surprisal) across two disciplines. surprisal is
# -log(point_mass); a moderate surprisal → a moderate point mass for the PIT jitter.
_CAL_ROWS = [
    {"result_percentile": (i + 0.5) / 12, "surprisal": 1.5, "discipline": "L" if i % 2 else "B"}
    for i in range(12)
]


def test_calibration_report_shape_and_breakdown() -> None:
    client = FakeGraphClient(read_results={CALIBRATION_QUERY: _CAL_ROWS})
    report = build_calibration_report(client, seed=7)

    overall = report["overall"]
    assert overall["n"] == 12
    assert 0.0 <= overall["mean"] <= 1.0
    assert overall["ks"] is not None and math.isfinite(overall["ks"])
    assert overall["ece"] is not None and math.isfinite(overall["ece"])
    # Both disciplines present, counts sum to the overall.
    assert set(report["by_discipline"]) == {"B", "L"}
    assert sum(b["n"] for b in report["by_discipline"].values()) == 12


def test_calibration_report_deterministic_under_seed() -> None:
    client = FakeGraphClient(read_results={CALIBRATION_QUERY: _CAL_ROWS})
    a = build_calibration_report(client, seed=42)
    b = build_calibration_report(client, seed=42)
    assert a == b


def test_calibration_report_empty_graceful() -> None:
    report = build_calibration_report(FakeGraphClient())  # no CALIBRATION_QUERY rows
    assert report["overall"]["n"] == 0
    assert report["by_discipline"] == {}


def test_weightfit_report_pipeline() -> None:
    # outcome = jr - tf (so rested_index correlates negatively at high w1) over a
    # small grid of independent (jr, tf); just exercises the rows → fit pipeline.
    rows = [
        {
            "jetlag_residual": a / 4 * 0.4,
            "travel_fatigue": b / 4 * 0.4,
            "outcome": (a / 4 * 0.4) - (b / 4 * 0.4),
            "discipline": "L",
            "travel_direction": "E",
        }
        for a in range(5)
        for b in range(5)
    ]
    client = FakeGraphClient(read_results={WEIGHTFIT_QUERY: rows})
    report = build_weightfit_report(client, outcome="elo_residual", grid_steps=21)

    assert report["n"] == len(rows)
    assert report["outcome_field"] == "elo_residual"
    assert {"w1", "w2", "pearson"} <= set(report["best"])
    assert {"w1", "w2", "pearson"} <= set(report["current"])
    assert len(report["curve"]) == 21
    # best is the most-negative point on the curve.
    pearsons = [e["pearson"] for e in report["curve"] if e["pearson"] is not None]
    assert report["best"]["pearson"] == min(pearsons)


def test_weightfit_report_empty_falls_back_to_prior() -> None:
    report = build_weightfit_report(FakeGraphClient(), outcome="result_percentile")
    assert report["n"] == 0
    assert report["best"]["w1"] == pytest.approx(0.7)
    assert report["best"]["pearson"] is None
    assert report["outcome_field"] == "result_percentile"


def test_weightfit_rejects_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="outcome must be one of"):
        build_weightfit_report(FakeGraphClient(), outcome="bogus")
    assert "elo_residual" in OUTCOME_FIELDS and "result_percentile" in OUTCOME_FIELDS
