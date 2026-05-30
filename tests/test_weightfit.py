"""Unit tests for climber_network.elo.weightfit.

All tests exercise pure-Python math with no database, graph, or network access.

Coverage:
- :func:`recompute_rested_index`: known values, clamping at 0 and 1.
- :func:`fit_weights`: recovery test (synthetic data with known true weights),
  ``current`` reports w1=0.7, ``curve`` has the right length and span, edge
  cases (empty samples, constant outcome / zero variance).
"""

from __future__ import annotations

import math
import random

import pytest

from climber_network.elo.weightfit import (
    WeightSample,
    fit_weights,
    recompute_rested_index,
)

# ---------------------------------------------------------------------------
# recompute_rested_index — unit tests
# ---------------------------------------------------------------------------


class TestRecomputeRestedIndex:
    def test_zero_components_is_one(self) -> None:
        """No jet lag, no fatigue → fully rested."""
        assert recompute_rested_index(0.0, 0.0, 0.7, 0.3) == pytest.approx(1.0)

    def test_both_max_components_is_zero(self) -> None:
        """Max jet lag + max fatigue, weights summing to 1 → clamped to 0."""
        # 1 - 0.7*1 - 0.3*1 = 0.0
        assert recompute_rested_index(1.0, 1.0, 0.7, 0.3) == pytest.approx(0.0)

    def test_weighted_mix(self) -> None:
        """Mid-way components → expected weighted average."""
        # 1 - 0.7*0.5 - 0.3*0.5 = 1 - 0.5 = 0.5
        assert recompute_rested_index(0.5, 0.5, 0.7, 0.3) == pytest.approx(0.5)

    def test_jetlag_only(self) -> None:
        """Only jet lag present → 1 - w1*jr."""
        # 1 - 0.7*1.0 - 0.3*0.0 = 0.3
        assert recompute_rested_index(1.0, 0.0, 0.7, 0.3) == pytest.approx(0.3)

    def test_fatigue_only(self) -> None:
        """Only travel fatigue present → 1 - w2*tf."""
        # 1 - 0.7*0.0 - 0.3*1.0 = 0.7
        assert recompute_rested_index(0.0, 1.0, 0.7, 0.3) == pytest.approx(0.7)

    def test_clamp_to_zero_large_components(self) -> None:
        """Penalty > 1 is clamped to 0 (never negative)."""
        # e.g. jr=2.0, tf=2.0, w1=0.7, w2=0.3 → penalty 2.0, clamped to 0
        assert recompute_rested_index(2.0, 2.0, 0.7, 0.3) == pytest.approx(0.0)
        # Same for individually large values
        assert recompute_rested_index(5.0, 0.0, 1.0, 0.0) == pytest.approx(0.0)

    def test_clamp_to_one_zero_penalty(self) -> None:
        """Zero penalty must give exactly 1.0, not above."""
        result = recompute_rested_index(0.0, 0.0, 0.0, 0.0)
        assert result == pytest.approx(1.0)

    def test_equal_weights(self) -> None:
        """With w1 = w2 = 0.5, both components contribute equally."""
        # 1 - 0.5*0.4 - 0.5*0.6 = 1 - 0.5 = 0.5
        assert recompute_rested_index(0.4, 0.6, 0.5, 0.5) == pytest.approx(0.5)

    def test_w1_zero_ignores_jetlag(self) -> None:
        """w1=0 means jet lag has no effect on restedness."""
        r_no_jl = recompute_rested_index(0.0, 0.5, 0.0, 1.0)
        r_full_jl = recompute_rested_index(1.0, 0.5, 0.0, 1.0)
        assert r_no_jl == pytest.approx(r_full_jl)

    def test_result_in_unit_interval(self) -> None:
        """Output is always in [0, 1] for any reasonable inputs."""
        for jr in [0.0, 0.3, 0.7, 1.0]:
            for tf in [0.0, 0.3, 0.7, 1.0]:
                r = recompute_rested_index(jr, tf, 0.6, 0.4)
                assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# fit_weights — edge cases
# ---------------------------------------------------------------------------


class TestFitWeightsEdgeCases:
    def test_empty_samples(self) -> None:
        """Empty input → n=0, both pearson None, best falls back to prior w1=0.7."""
        result = fit_weights([])
        assert result["n"] == 0
        assert result["best"]["w1"] == pytest.approx(0.7)
        assert result["best"]["w2"] == pytest.approx(0.3)
        assert result["best"]["pearson"] is None
        assert result["current"]["pearson"] is None

    def test_single_sample(self) -> None:
        """One sample → pearson undefined, falls back to prior."""
        samples = [WeightSample(jetlag_residual=0.5, travel_fatigue=0.3, outcome=-0.1)]
        result = fit_weights(samples)
        assert result["n"] == 1
        assert result["best"]["pearson"] is None
        assert result["best"]["w1"] == pytest.approx(0.7)

    def test_curve_length_equals_grid_steps(self) -> None:
        """curve list has exactly grid_steps entries."""
        samples = [
            WeightSample(jetlag_residual=0.2, travel_fatigue=0.3, outcome=-0.1),
            WeightSample(jetlag_residual=0.8, travel_fatigue=0.7, outcome=-0.9),
        ]
        for steps in [11, 51, 101]:
            result = fit_weights(samples, grid_steps=steps)
            assert len(result["curve"]) == steps

    def test_curve_spans_zero_to_one(self) -> None:
        """curve[0].w1 == 0 and curve[-1].w1 == 1 (default 101 steps)."""
        samples = [
            WeightSample(jetlag_residual=0.2, travel_fatigue=0.1, outcome=0.5),
            WeightSample(jetlag_residual=0.6, travel_fatigue=0.4, outcome=-0.5),
        ]
        result = fit_weights(samples)
        assert result["curve"][0]["w1"] == pytest.approx(0.0)
        assert result["curve"][-1]["w1"] == pytest.approx(1.0)

    def test_current_uses_prior_weights(self) -> None:
        """current always reports w1=0.7, w2=0.3."""
        samples = [
            WeightSample(jetlag_residual=0.1, travel_fatigue=0.2, outcome=0.8),
            WeightSample(jetlag_residual=0.9, travel_fatigue=0.8, outcome=-0.8),
        ]
        result = fit_weights(samples)
        assert result["current"]["w1"] == pytest.approx(0.7)
        assert result["current"]["w2"] == pytest.approx(0.3)

    def test_curve_w1_plus_w2_equals_one(self) -> None:
        """Every curve entry satisfies w1 + w2 == 1."""
        samples = [
            WeightSample(jetlag_residual=0.3, travel_fatigue=0.2, outcome=-0.5),
            WeightSample(jetlag_residual=0.7, travel_fatigue=0.6, outcome=-0.9),
        ]
        result = fit_weights(samples)
        for entry in result["curve"]:
            assert entry["w1"] + entry["w2"] == pytest.approx(1.0)

    def test_constant_outcome_returns_none_pearson(self) -> None:
        """Constant outcome → zero variance → pearson is None for all curve entries."""
        samples = [
            WeightSample(jetlag_residual=0.2, travel_fatigue=0.3, outcome=0.5),
            WeightSample(jetlag_residual=0.8, travel_fatigue=0.7, outcome=0.5),
            WeightSample(jetlag_residual=0.5, travel_fatigue=0.5, outcome=0.5),
        ]
        result = fit_weights(samples)
        assert all(e["pearson"] is None for e in result["curve"])
        assert result["best"]["pearson"] is None
        assert result["best"]["w1"] == pytest.approx(0.7)

    def test_n_reflects_sample_count(self) -> None:
        """n in result equals the number of samples passed."""
        samples = [
            WeightSample(
                jetlag_residual=float(i) / 10, travel_fatigue=float(i) / 10, outcome=-float(i)
            )
            for i in range(7)
        ]
        result = fit_weights(samples)
        assert result["n"] == 7


# ---------------------------------------------------------------------------
# fit_weights — recovery test (synthetic data with known true weights)
# ---------------------------------------------------------------------------


class TestFitWeightsRecovery:
    """Verify that fit_weights recovers a known w1 from synthetic data.

    The synthetic outcome is generated as::

        outcome = -5 * recompute_rested_index(jr, tf, W1_TRUE, 1 - W1_TRUE) + noise

    so the true signal is a *negative* linear function of rested_index at W1_TRUE.
    With enough samples and small noise the best w1 should be within ±0.10 of
    W1_TRUE and the correlation should be strongly negative (< -0.8).
    """

    W1_TRUE: float = 0.3
    N_SAMPLES: int = 200
    NOISE_SCALE: float = 0.05
    TOLERANCE: float = 0.10

    @pytest.fixture(autouse=True)
    def _seed_rng(self) -> None:
        """Pin RNG for deterministic test behaviour."""
        self._rng = random.Random(42)

    def _make_samples(self) -> list[WeightSample]:
        w1 = self.W1_TRUE
        w2 = 1.0 - w1
        samples: list[WeightSample] = []
        for _ in range(self.N_SAMPLES):
            # Spread inputs across the unit square.
            jr = self._rng.uniform(0.0, 1.0)
            tf = self._rng.uniform(0.0, 1.0)
            ri = recompute_rested_index(jr, tf, w1, w2)
            noise = self._rng.gauss(0.0, self.NOISE_SCALE)
            outcome = -5.0 * ri + noise
            samples.append(WeightSample(jetlag_residual=jr, travel_fatigue=tf, outcome=outcome))
        return samples

    def test_recovered_w1_close_to_true(self) -> None:
        """best.w1 must be within ±TOLERANCE of W1_TRUE."""
        result = fit_weights(self._make_samples())
        best_w1 = result["best"]["w1"]
        assert abs(best_w1 - self.W1_TRUE) <= self.TOLERANCE, (
            f"Recovered w1={best_w1:.3f} is more than {self.TOLERANCE} away "
            f"from the true value {self.W1_TRUE}"
        )

    def test_best_pearson_is_strongly_negative(self) -> None:
        """The best correlation must be strongly negative (< -0.8)."""
        result = fit_weights(self._make_samples())
        r = result["best"]["pearson"]
        assert r is not None
        assert r < -0.8, f"Expected strongly negative pearson, got {r:.4f}"

    def test_current_pearson_is_defined(self) -> None:
        """current (literature prior w1=0.7) must produce a defined Pearson value."""
        result = fit_weights(self._make_samples())
        assert result["current"]["pearson"] is not None
        assert math.isfinite(result["current"]["pearson"])

    def test_best_pearson_stronger_than_current(self) -> None:
        """The fitted best should have a stronger signal than the prior w1=0.7."""
        # Because W1_TRUE=0.3 is far from 0.7, the fitted weights should
        # show a notably stronger correlation than the prior.
        result = fit_weights(self._make_samples())
        best_r = result["best"]["pearson"]
        current_r = result["current"]["pearson"]
        assert best_r is not None and current_r is not None
        # abs(best) >= abs(current)  — best must be at least as strong.
        assert abs(best_r) >= abs(current_r)

    def test_curve_has_101_entries(self) -> None:
        """Default grid produces exactly 101 curve entries."""
        result = fit_weights(self._make_samples())
        assert len(result["curve"]) == 101

    def test_all_curve_pearsons_are_finite(self) -> None:
        """All 101 curve Pearson values must be finite (not None, not NaN/Inf)."""
        result = fit_weights(self._make_samples())
        for entry in result["curve"]:
            r = entry["pearson"]
            assert r is not None
            assert math.isfinite(r), f"Non-finite pearson at w1={entry['w1']}: {r}"


def test_objective_prefers_most_negative_over_larger_magnitude_positive() -> None:
    """The selection objective is most-negative, not largest-absolute.

    jr (small spread) and tf (large spread) are independent and outcome = jr - tf.
    At w1=0 the rested_index (1-tf) correlates strongly POSITIVE with the outcome;
    at w1=1 (1-jr) it correlates weakly NEGATIVE. A largest-absolute objective
    would wrongly pick the big positive; most-negative must pick the negative one.
    """
    samples = [
        WeightSample(
            jetlag_residual=a / 9 * 0.3,
            travel_fatigue=b / 9 * 1.0,
            outcome=(a / 9 * 0.3) - (b / 9 * 1.0),
        )
        for a in range(10)
        for b in range(10)
    ]
    result = fit_weights(samples, grid_steps=11)
    pearsons = [e["pearson"] for e in result["curve"] if e["pearson"] is not None]

    assert result["best"]["pearson"] == min(pearsons)  # most-negative objective
    assert result["best"]["pearson"] < 0.0  # we selected a negative correlation
    # A larger-magnitude POSITIVE correlation existed but was (correctly) not chosen.
    assert max(pearsons) > abs(result["best"]["pearson"])
