"""Unit tests for climber_network.travel.formulas.

All tests exercise pure-Python math against hand-computed fixtures. No
database, graph, or network access required. Default constants come from
``TRAVEL_PARAMS`` (cruise 800 km/h, overhead 1.5 h, full-fatigue 12 h,
decay 4 days, recovery cap 5 days, w1 0.7 / w2 0.3, arrive 2 days before).
"""

from __future__ import annotations

import math

import pytest

from climber_network.config import TRAVEL_PARAMS
from climber_network.travel.formulas import (
    compute_restedness,
    direction,
    est_flight_h,
    haversine_km,
    jetlag_residual,
    recovery_days_needed,
    rested_index,
    travel_fatigue,
    tz_delta_h,
)

# ---------------------------------------------------------------------------
# haversine_km
# ---------------------------------------------------------------------------


class TestHaversineKm:
    def test_identical_points_is_zero(self) -> None:
        assert haversine_km(48.85, 2.35, 48.85, 2.35) == pytest.approx(0.0)

    def test_one_degree_latitude_is_about_111km(self) -> None:
        # One degree of latitude is ~111.19 km on a 6371.0088 km-radius sphere.
        assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)

    def test_known_paris_to_tokyo(self) -> None:
        # Paris (48.8566, 2.3522) -> Tokyo (35.6762, 139.6503): ~9714 km.
        d = haversine_km(48.8566, 2.3522, 35.6762, 139.6503)
        assert d == pytest.approx(9714.0, abs=30.0)

    def test_symmetric(self) -> None:
        a = haversine_km(10.0, 20.0, -30.0, 100.0)
        b = haversine_km(-30.0, 100.0, 10.0, 20.0)
        assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# est_flight_h
# ---------------------------------------------------------------------------


class TestEstFlightH:
    def test_zero_distance_is_overhead_only(self) -> None:
        assert est_flight_h(0.0) == pytest.approx(TRAVEL_PARAMS.flight_overhead_h)

    def test_8000km(self) -> None:
        # 8000 / 800 + 1.5 = 11.5 h
        assert est_flight_h(8000.0) == pytest.approx(11.5)

    def test_uses_params(self) -> None:
        # 800 km / 800 kmh = 1.0 h cruise + 1.5 overhead = 2.5 h
        assert est_flight_h(800.0) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# tz_delta_h + direction
# ---------------------------------------------------------------------------


class TestTzDeltaAndDirection:
    def test_eastward_is_positive(self) -> None:
        # Paris (+1) -> Tokyo (+9): venue - origin = +8 (eastward).
        assert tz_delta_h(1.0, 9.0) == pytest.approx(8.0)

    def test_westward_is_negative(self) -> None:
        # Paris (+1) -> Denver (-7): venue - origin = -8 (westward).
        assert tz_delta_h(1.0, -7.0) == pytest.approx(-8.0)

    def test_no_change(self) -> None:
        assert tz_delta_h(2.0, 2.0) == pytest.approx(0.0)

    def test_direction_east(self) -> None:
        assert direction(8.0) == "E"

    def test_direction_west(self) -> None:
        assert direction(-3.0) == "W"

    def test_direction_none(self) -> None:
        assert direction(0.0) == "none"


# ---------------------------------------------------------------------------
# recovery_days_needed — the 1-day-east / 0.5-day-west rule
# ---------------------------------------------------------------------------


class TestRecoveryDaysNeeded:
    def test_eastward_3tz_is_3_days(self) -> None:
        # 1.0 day/tz eastward * 3 tz = 3.0 days.
        assert recovery_days_needed(3.0) == pytest.approx(3.0)

    def test_westward_3tz_is_1_5_days(self) -> None:
        # 0.5 day/tz westward * 3 tz = 1.5 days.
        assert recovery_days_needed(-3.0) == pytest.approx(1.5)

    def test_east_vs_west_asymmetry(self) -> None:
        # Same magnitude, eastward needs twice the westward recovery.
        assert recovery_days_needed(4.0) == pytest.approx(2.0 * recovery_days_needed(-4.0))

    def test_zero_tz_is_zero_days(self) -> None:
        assert recovery_days_needed(0.0) == pytest.approx(0.0)

    def test_8tz_east_capped_at_5(self) -> None:
        # 1.0 * 8 = 8 days, capped at recovery_cap_days (5.0).
        assert TRAVEL_PARAMS.recovery_cap_days == 5.0
        assert recovery_days_needed(8.0) == pytest.approx(5.0)

    def test_large_west_capped_at_5(self) -> None:
        # 0.5 * 12 = 6 days, capped at 5.0.
        assert recovery_days_needed(-12.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# jetlag_residual
# ---------------------------------------------------------------------------


class TestJetlagResidual:
    def test_zero_recovery_is_zero(self) -> None:
        # No timezone change -> no jet lag, regardless of days elapsed.
        assert jetlag_residual(0.0, 0.0) == pytest.approx(0.0)
        assert jetlag_residual(5.0, 0.0) == pytest.approx(0.0)

    def test_fresh_arrival_is_one(self) -> None:
        assert jetlag_residual(0.0, 3.0) == pytest.approx(1.0)

    def test_halfway_recovered(self) -> None:
        # 1.5 of 3 recovery days -> residual 0.5.
        assert jetlag_residual(1.5, 3.0) == pytest.approx(0.5)

    def test_fully_recovered_days_equal_recovery(self) -> None:
        # days_since_arrival >= recovery_days_needed -> residual 0.
        assert jetlag_residual(3.0, 3.0) == pytest.approx(0.0)

    def test_overshoot_clamped_to_zero(self) -> None:
        assert jetlag_residual(10.0, 3.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# travel_fatigue
# ---------------------------------------------------------------------------


class TestTravelFatigue:
    def test_full_flight_fresh_arrival_is_one(self) -> None:
        # 12 h flight == fatigue_full_h, 0 days elapsed -> 1.0.
        assert travel_fatigue(12.0, 0.0) == pytest.approx(1.0)

    def test_long_flight_intensity_saturates(self) -> None:
        # Beyond fatigue_full_h the intensity clamps at 1.
        assert travel_fatigue(20.0, 0.0) == pytest.approx(1.0)

    def test_half_flight_fresh(self) -> None:
        # 6 h / 12 h = 0.5 intensity, no decay.
        assert travel_fatigue(6.0, 0.0) == pytest.approx(0.5)

    def test_decay_halfway(self) -> None:
        # 12 h flight, 2 of 4 decay days -> 1.0 * (1 - 2/4) = 0.5.
        assert travel_fatigue(12.0, 2.0) == pytest.approx(0.5)

    def test_fully_decayed(self) -> None:
        # days >= fatigue_decay_days -> 0.
        assert travel_fatigue(12.0, 4.0) == pytest.approx(0.0)
        assert travel_fatigue(12.0, 10.0) == pytest.approx(0.0)

    def test_positive_even_with_zero_tz(self) -> None:
        # Travel fatigue is incurred regardless of timezone change.
        assert travel_fatigue(8.0, 0.0) > 0.0


# ---------------------------------------------------------------------------
# rested_index
# ---------------------------------------------------------------------------


class TestRestedIndex:
    def test_no_penalty_is_one(self) -> None:
        assert rested_index(0.0, 0.0) == pytest.approx(1.0)

    def test_both_terms_live(self) -> None:
        # 1 - 0.7*1.0 - 0.3*1.0 = 0.0 (both maxed).
        assert rested_index(1.0, 1.0) == pytest.approx(0.0)

    def test_weighted_mix(self) -> None:
        # 1 - 0.7*0.5 - 0.3*0.5 = 1 - 0.5 = 0.5.
        assert rested_index(0.5, 0.5) == pytest.approx(0.5)

    def test_jetlag_dominates_via_w1(self) -> None:
        # 1 - 0.7*1.0 - 0.3*0.0 = 0.3.
        assert rested_index(1.0, 0.0) == pytest.approx(0.3)

    def test_clamped_to_zero(self) -> None:
        # Penalty can never push below 0 even with weird inputs.
        assert rested_index(2.0, 2.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_restedness — end-to-end
# ---------------------------------------------------------------------------


class TestComputeRestedness:
    def test_report_keys_present(self) -> None:
        report = compute_restedness(
            distance_km=8000.0,
            origin_offset_h=1.0,
            venue_offset_h=9.0,
            days_since_arrival=0.0,
        )
        expected_keys = {
            "distance_km",
            "est_flight_h",
            "tz_delta_h",
            "direction",
            "recovery_days_needed",
            "days_since_arrival",
            "jetlag_residual",
            "travel_fatigue",
            "rested_index",
            "model_version",
        }
        assert set(report.keys()) == expected_keys

    def test_eastward_8tz_fresh_arrival(self) -> None:
        # Paris (+1) -> Tokyo (+9), 9714 km-ish, day 0.
        report = compute_restedness(
            distance_km=9714.0,
            origin_offset_h=1.0,
            venue_offset_h=9.0,
            days_since_arrival=0.0,
        )
        assert report["direction"] == "E"
        assert report["tz_delta_h"] == pytest.approx(8.0)
        # 8 tz east -> 8 days, capped at 5.
        assert report["recovery_days_needed"] == pytest.approx(5.0)
        # Fresh arrival -> full jet lag.
        assert report["jetlag_residual"] == pytest.approx(1.0)
        # est_flight_h = 9714/800 + 1.5 = 13.6425 h -> intensity clamps to 1.
        assert report["est_flight_h"] == pytest.approx(9714.0 / 800.0 + 1.5)
        assert report["travel_fatigue"] == pytest.approx(1.0)
        # rested_index = 1 - 0.7*1 - 0.3*1 = 0.0.
        assert report["rested_index"] == pytest.approx(0.0)
        assert report["model_version"] == TRAVEL_PARAMS.model_version

    def test_rested_index_in_unit_interval_both_terms_live(self) -> None:
        # Both jet lag and fatigue are non-trivial at the default
        # arrive_days_before; rested_index must land strictly inside (0, 1).
        days = float(TRAVEL_PARAMS.arrive_days_before)  # 2 days
        report = compute_restedness(
            distance_km=6000.0,
            origin_offset_h=0.0,
            venue_offset_h=5.0,  # 5 tz east -> 5 recovery days
            days_since_arrival=days,
        )
        assert report["direction"] == "E"
        # 5 tz east -> 5 recovery days; 2 of 5 elapsed -> residual 0.6.
        assert report["recovery_days_needed"] == pytest.approx(5.0)
        assert report["jetlag_residual"] == pytest.approx(1.0 - 2.0 / 5.0)
        # est_flight_h = 6000/800 + 1.5 = 9.0 h; intensity = 9/12 = 0.75;
        # decay = 1 - 2/4 = 0.5 -> fatigue 0.375.
        assert report["travel_fatigue"] == pytest.approx(0.75 * 0.5)
        # Both terms strictly positive -> both "live".
        assert report["jetlag_residual"] > 0.0
        assert report["travel_fatigue"] > 0.0
        assert 0.0 < report["rested_index"] < 1.0
        # rested = 1 - 0.7*0.6 - 0.3*0.375 = 1 - 0.42 - 0.1125 = 0.4675.
        assert report["rested_index"] == pytest.approx(0.4675)

    def test_zero_tz_has_no_jetlag_but_has_fatigue(self) -> None:
        # Same offset -> no timezone change. Jet lag must be 0, but a real
        # flight still produces travel fatigue on arrival.
        report = compute_restedness(
            distance_km=4000.0,
            origin_offset_h=2.0,
            venue_offset_h=2.0,
            days_since_arrival=0.0,
        )
        assert report["direction"] == "none"
        assert report["tz_delta_h"] == pytest.approx(0.0)
        assert report["recovery_days_needed"] == pytest.approx(0.0)
        assert report["jetlag_residual"] == pytest.approx(0.0)
        assert report["travel_fatigue"] > 0.0
        assert math.isfinite(report["rested_index"])

    def test_fully_recovered_after_recovery_window(self) -> None:
        # Arrive far enough ahead that both jet lag and fatigue have resolved.
        report = compute_restedness(
            distance_km=9000.0,
            origin_offset_h=0.0,
            venue_offset_h=6.0,  # 6 tz east -> 6 days -> capped at 5
            days_since_arrival=6.0,  # >= recovery (5) and >= decay (4)
        )
        assert report["jetlag_residual"] == pytest.approx(0.0)
        assert report["travel_fatigue"] == pytest.approx(0.0)
        assert report["rested_index"] == pytest.approx(1.0)

    def test_westward_recovers_faster_than_eastward(self) -> None:
        # Same |tz| magnitude and arrival timing; westward should be rester.
        east = compute_restedness(
            distance_km=8000.0,
            origin_offset_h=0.0,
            venue_offset_h=4.0,
            days_since_arrival=1.0,
        )
        west = compute_restedness(
            distance_km=8000.0,
            origin_offset_h=0.0,
            venue_offset_h=-4.0,
            days_since_arrival=1.0,
        )
        assert east["direction"] == "E"
        assert west["direction"] == "W"
        assert west["jetlag_residual"] < east["jetlag_residual"]
        assert west["rested_index"] > east["rested_index"]
