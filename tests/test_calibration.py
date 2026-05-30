"""Unit tests for climber_network.elo.calibration.

Pure-Python only — no database, numpy, or external service. The PIT identities
are checked against the two stored MC scalars (``result_percentile`` and
``surprisal``); the KS / ECE diagnostics are exercised on uniform, degenerate,
seeded-random, and skewed samples.
"""

from __future__ import annotations

import math
import random

import pytest

from climber_network.elo.calibration import (
    calibration_report,
    expected_calibration_error,
    ks_uniform_statistic,
    point_mass_from_surprisal,
    randomized_pit,
    randomized_pit_rng,
    reliability_bins,
)

# ---------------------------------------------------------------------------
# point_mass_from_surprisal
# ---------------------------------------------------------------------------


def test_point_mass_surprisal_zero_is_one() -> None:
    assert point_mass_from_surprisal(0.0) == 1.0


def test_point_mass_large_surprisal_near_zero() -> None:
    assert point_mass_from_surprisal(50.0) == pytest.approx(0.0, abs=1e-9)


def test_point_mass_matches_exp() -> None:
    assert point_mass_from_surprisal(math.log(4.0)) == pytest.approx(0.25)


def test_point_mass_clamped_to_unit_interval() -> None:
    # A negative surprisal would give exp(-s) > 1; must clamp to 1.0.
    assert point_mass_from_surprisal(-5.0) == 1.0


# ---------------------------------------------------------------------------
# randomized_pit
# ---------------------------------------------------------------------------


def test_randomized_pit_u_zero_is_strict_below() -> None:
    # u = 0 -> P(X < actual) = result_percentile - point_mass.
    assert randomized_pit(0.7, 0.2, 0.0) == pytest.approx(0.5)


def test_randomized_pit_u_one_is_result_percentile() -> None:
    # u = 1 -> result_percentile (the inclusive CDF).
    assert randomized_pit(0.7, 0.2, 1.0) == pytest.approx(0.7)


def test_randomized_pit_interpolates() -> None:
    assert randomized_pit(0.7, 0.2, 0.5) == pytest.approx(0.6)


def test_randomized_pit_clamped() -> None:
    # Degenerate inputs (mass exceeding percentile) clamp into [0, 1].
    assert randomized_pit(0.1, 0.5, 0.0) == 0.0
    assert randomized_pit(0.9, 0.5, 5.0) == 1.0


def test_randomized_pit_rng_draws_from_rng() -> None:
    rng = random.Random(0)
    expected = randomized_pit(0.7, 0.2, random.Random(0).random())
    assert randomized_pit_rng(0.7, 0.2, rng=rng) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# ks_uniform_statistic / expected_calibration_error — known samples
# ---------------------------------------------------------------------------


def _uniform_grid(n: int) -> list[float]:
    """A perfectly uniform sample on [0, 1]: midpoints of n equal cells."""
    return [(i + 0.5) / n for i in range(n)]


def test_ks_perfectly_uniform_grid_near_zero() -> None:
    values = _uniform_grid(1000)
    assert ks_uniform_statistic(values) <= 1.0 / 1000 + 1e-9


def test_ece_perfectly_uniform_grid_near_zero() -> None:
    values = _uniform_grid(1000)
    assert expected_calibration_error(values, n_bins=10) == pytest.approx(0.0, abs=1e-9)


def test_ks_degenerate_all_half_is_about_half() -> None:
    values = [0.5] * 500
    # ECDF jumps from 0 to 1 at 0.5; the gap to the identity CDF is ~0.5.
    assert ks_uniform_statistic(values) == pytest.approx(0.5, abs=1e-3)


def test_ece_degenerate_all_half_is_large() -> None:
    values = [0.5] * 500
    # All mass in one bin: |1 - 0.1| + 9 * |0 - 0.1| = 1.8.
    assert expected_calibration_error(values, n_bins=10) == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# Seeded random uniform vs. clustered/skewed — contrast
# ---------------------------------------------------------------------------


def test_seeded_uniform_sample_small_ks_and_ece() -> None:
    rng = random.Random(0)
    values = [rng.random() for _ in range(5000)]
    assert ks_uniform_statistic(values) < 0.05
    assert expected_calibration_error(values, n_bins=10) < 0.1


def test_skewed_sample_worse_than_uniform() -> None:
    rng = random.Random(0)
    uniform = [rng.random() for _ in range(5000)]
    # Squaring pushes mass toward 0 -> clearly non-uniform.
    skewed = [rng.random() ** 2 for _ in range(5000)]

    assert ks_uniform_statistic(skewed) > ks_uniform_statistic(uniform)
    assert expected_calibration_error(skewed) > expected_calibration_error(uniform)
    # And the skew is substantial, not just marginally larger.
    assert ks_uniform_statistic(skewed) > 0.15


def test_clustered_sample_large_ece() -> None:
    # All values in [0.4, 0.5): two adjacent bins carry everything.
    rng = random.Random(1)
    values = [0.4 + 0.1 * rng.random() for _ in range(2000)]
    assert expected_calibration_error(values, n_bins=10) > 1.0


# ---------------------------------------------------------------------------
# reliability_bins
# ---------------------------------------------------------------------------


def test_reliability_bins_counts_sum_to_n() -> None:
    values = _uniform_grid(137)
    bins = reliability_bins(values, n_bins=10)
    assert sum(b["count"] for b in bins) == 137


def test_reliability_bins_freqs_sum_to_one() -> None:
    values = _uniform_grid(137)
    bins = reliability_bins(values, n_bins=10)
    assert sum(b["freq"] for b in bins) == pytest.approx(1.0)


def test_reliability_bins_edges_and_expected_freq() -> None:
    bins = reliability_bins(_uniform_grid(10), n_bins=10)
    assert len(bins) == 10
    assert bins[0]["lo"] == pytest.approx(0.0)
    assert bins[-1]["hi"] == pytest.approx(1.0)
    for b in bins:
        assert b["expected_freq"] == pytest.approx(0.1)
        assert set(b) == {"lo", "hi", "count", "freq", "expected_freq"}


def test_reliability_bins_value_one_goes_in_last_bin() -> None:
    bins = reliability_bins([1.0], n_bins=10)
    assert bins[-1]["count"] == 1.0
    assert sum(b["count"] for b in bins) == 1.0


# ---------------------------------------------------------------------------
# calibration_report
# ---------------------------------------------------------------------------


def test_calibration_report_bundles_all_keys() -> None:
    report = calibration_report(_uniform_grid(1000), n_bins=10)
    assert set(report) == {"n", "mean", "ks", "ece", "bins"}
    assert report["n"] == 1000
    assert len(report["bins"]) == 10


def test_calibration_report_uniform_mean_near_half() -> None:
    report = calibration_report(_uniform_grid(1000), n_bins=10)
    assert report["mean"] == pytest.approx(0.5, abs=1e-6)
    assert report["ks"] < 0.01
    assert report["ece"] == pytest.approx(0.0, abs=1e-9)


def test_calibration_report_seeded_random_mean_near_half() -> None:
    rng = random.Random(0)
    values = [rng.random() for _ in range(5000)]
    report = calibration_report(values)
    assert report["mean"] == pytest.approx(0.5, abs=0.02)


# ---------------------------------------------------------------------------
# Edge cases: empty input
# ---------------------------------------------------------------------------


def test_ks_empty_is_zero() -> None:
    assert ks_uniform_statistic([]) == 0.0


def test_ece_empty_is_zero() -> None:
    assert expected_calibration_error([]) == 0.0


def test_reliability_bins_empty_all_zero() -> None:
    bins = reliability_bins([], n_bins=10)
    assert len(bins) == 10
    assert all(b["count"] == 0.0 and b["freq"] == 0.0 for b in bins)


def test_calibration_report_empty() -> None:
    report = calibration_report([])
    assert report["n"] == 0
    assert report["mean"] is None
    assert report["ks"] is None
    assert report["ece"] is None
    assert len(report["bins"]) == 10


# ---------------------------------------------------------------------------
# End-to-end: PIT recovered from the two stored MC scalars is ~uniform
# ---------------------------------------------------------------------------


def test_pit_from_stored_scalars_is_calibrated() -> None:
    """A correctly-specified discrete model yields ~uniform randomized PIT.

    Simulate many rounds from a known PMF, derive the two stored scalars
    (``result_percentile`` = inclusive CDF, ``surprisal`` = -log point mass) the
    way montecarlo.py would, then check the randomized PIT is ~Uniform[0, 1].
    """
    rng = random.Random(7)
    pmf = [0.4, 0.3, 0.2, 0.1]  # 4 ranks
    cdf = [sum(pmf[: r + 1]) for r in range(len(pmf))]

    pit_values: list[float] = []
    for _ in range(8000):
        # Draw an actual rank from the true model.
        u = rng.random()
        actual = next(r for r, c in enumerate(cdf) if u <= c)
        result_percentile = cdf[actual]
        surprisal = -math.log(pmf[actual])
        point_mass = point_mass_from_surprisal(surprisal)
        pit_values.append(randomized_pit_rng(result_percentile, point_mass, rng=rng))

    report = calibration_report(pit_values)
    assert report["mean"] == pytest.approx(0.5, abs=0.02)
    assert report["ks"] < 0.05
    assert report["ece"] < 0.1
