"""climber_network.travel.formulas — Pure travel / circadian-load math.

The "L3" layer of the model: great-circle distance, estimated flight time,
timezone deltas, jet-lag residual, raw travel fatigue, and a composite
``rested_index`` in ``[0, 1]`` (1 = fully rested).

Everything here is a **pure function**. There is no graph, database, numpy, or
external-service dependency — only :mod:`math` and the frozen
:class:`~climber_network.config.TravelParams` constants.

Circadian recovery rule
------------------------
Eastward travel (advancing the body clock) is harder to adjust to than
westward travel. We use the widely cited heuristic that the body re-entrains
at roughly **1 timezone (hour) per day after eastward flights** and about
**0.5 timezones per day after westward flights** (equivalently ~1.5 h/day
westward). See :func:`recovery_days_needed`.
"""

from __future__ import annotations

import math
from typing import Literal, TypedDict

from climber_network.config import TRAVEL_PARAMS, TravelParams

__all__ = [
    "Direction",
    "RestednessReport",
    "compute_restedness",
    "direction",
    "est_flight_h",
    "haversine_km",
    "jetlag_residual",
    "recovery_days_needed",
    "rested_index",
    "travel_fatigue",
    "tz_delta_h",
]

Direction = Literal["E", "W", "none"]

#: Small epsilon to guard against division by zero.
_EPS: float = 1e-9

#: Mean Earth radius in kilometres (WGS-84 mean radius).
_EARTH_RADIUS_KM: float = 6371.0088

#: Recovery rate (days per timezone) for eastward travel.
_RATE_EAST: float = 1.0

#: Recovery rate (days per timezone) for westward travel.
_RATE_WEST: float = 0.5


class RestednessReport(TypedDict):
    """Full set of intermediates returned by :func:`compute_restedness`."""

    distance_km: float
    est_flight_h: float
    tz_delta_h: float
    direction: Direction
    recovery_days_needed: float
    days_since_arrival: float
    jetlag_residual: float
    travel_fatigue: float
    rested_index: float
    model_version: str


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp ``value`` into the closed interval ``[lo, hi]``."""
    return max(lo, min(hi, value))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometres.

    Args:
        lat1: Latitude of the first point, in degrees.
        lon1: Longitude of the first point, in degrees.
        lat2: Latitude of the second point, in degrees.
        lon2: Longitude of the second point, in degrees.

    Returns:
        The great-circle (haversine) distance in kilometres. ``0.0`` when the
        two points coincide.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return _EARTH_RADIUS_KM * c


def est_flight_h(distance_km: float, params: TravelParams = TRAVEL_PARAMS) -> float:
    """Estimate door-to-door flight time (hours) from great-circle distance.

    Flight time is modelled as cruise time plus a fixed overhead (check-in,
    boarding, taxi, deplaning)::

        distance_km / cruise_kmh + flight_overhead_h

    Args:
        distance_km: Great-circle distance in kilometres (>= 0).
        params: Travel model constants.

    Returns:
        Estimated flight time in hours.
    """
    return distance_km / params.cruise_kmh + params.flight_overhead_h


def tz_delta_h(origin_offset_h: float, venue_offset_h: float) -> float:
    """Signed timezone delta between origin and venue, in hours.

    Args:
        origin_offset_h: UTC offset (hours) of the origin.
        venue_offset_h: UTC offset (hours) of the venue.

    Returns:
        ``venue_offset_h - origin_offset_h``. A **positive** value means the
        venue is *east* of the origin (the body clock must advance).
    """
    return venue_offset_h - origin_offset_h


def direction(tz_delta_h: float) -> Direction:
    """Classify the travel direction implied by a signed timezone delta.

    Args:
        tz_delta_h: Signed timezone delta (see :func:`tz_delta_h`).

    Returns:
        ``"E"`` for eastward (positive delta), ``"W"`` for westward (negative
        delta), or ``"none"`` when there is no timezone change.
    """
    if tz_delta_h > 0:
        return "E"
    if tz_delta_h < 0:
        return "W"
    return "none"


def recovery_days_needed(tz_delta_h: float, params: TravelParams = TRAVEL_PARAMS) -> float:
    """Days of circadian recovery needed for a given timezone delta.

    Applies the standard jet-lag heuristic: the body re-entrains at about
    **1 timezone per day eastward** and **0.5 timezones per day westward**.
    Concretely, the number of recovery days is ``rate * |tz_delta_h|`` where
    the rate is ``1.0`` for eastward travel and ``0.5`` for westward travel,
    capped at ``recovery_cap_days``. Returns ``0.0`` when there is no timezone
    change.

    Args:
        tz_delta_h: Signed timezone delta (see :func:`tz_delta_h`).
        params: Travel model constants (supplies ``recovery_cap_days``).

    Returns:
        Recovery days in ``[0, recovery_cap_days]``.
    """
    if tz_delta_h == 0:
        return 0.0
    rate = _RATE_EAST if tz_delta_h > 0 else _RATE_WEST
    raw = rate * abs(tz_delta_h)
    return min(raw, params.recovery_cap_days)


def jetlag_residual(days_since_arrival: float, recovery_days_needed: float) -> float:
    """Fraction of jet lag still unresolved, in ``[0, 1]``.

    Models recovery as a linear ramp from full jet lag on arrival to zero once
    ``days_since_arrival`` reaches ``recovery_days_needed``::

        clamp(1 - days_since_arrival / max(recovery_days_needed, eps), 0, 1)

    With ``recovery_days_needed == 0`` (no timezone change) the residual is
    ``0`` immediately.

    Args:
        days_since_arrival: Days elapsed since landing at the venue (>= 0).
        recovery_days_needed: Output of :func:`recovery_days_needed`.

    Returns:
        Residual jet-lag fraction in ``[0, 1]`` (1 = freshly arrived, fully
        jet-lagged; 0 = recovered or no timezone change).
    """
    if recovery_days_needed <= 0:
        return 0.0
    return _clamp(1.0 - days_since_arrival / max(recovery_days_needed, _EPS))


def travel_fatigue(
    est_flight_h: float,
    days_since_arrival: float,
    params: TravelParams = TRAVEL_PARAMS,
) -> float:
    """Raw travel fatigue from flight duration, decaying after arrival.

    Combines a flight-length term (longer flights → more fatigue, saturating
    at ``fatigue_full_h``) with a linear post-arrival decay over
    ``fatigue_decay_days``::

        clamp(est_flight_h / fatigue_full_h, 0, 1)
            * max(0, 1 - days_since_arrival / fatigue_decay_days)

    Unlike jet lag, travel fatigue is incurred even with no timezone change.

    Args:
        est_flight_h: Estimated flight time in hours (see :func:`est_flight_h`).
        days_since_arrival: Days elapsed since landing (>= 0).
        params: Travel model constants (``fatigue_full_h``,
            ``fatigue_decay_days``).

    Returns:
        Travel-fatigue fraction in ``[0, 1]``.
    """
    intensity = _clamp(est_flight_h / params.fatigue_full_h)
    decay = max(0.0, 1.0 - days_since_arrival / params.fatigue_decay_days)
    return intensity * decay


def rested_index(
    jetlag_residual: float,
    travel_fatigue: float,
    params: TravelParams = TRAVEL_PARAMS,
) -> float:
    """Composite restedness score in ``[0, 1]`` (1 = fully rested).

    Penalises a baseline of full restedness by a weighted sum of the jet-lag
    residual and travel fatigue::

        clamp(1 - w1 * jetlag_residual - w2 * travel_fatigue, 0, 1)

    Args:
        jetlag_residual: Output of :func:`jetlag_residual`, in ``[0, 1]``.
        travel_fatigue: Output of :func:`travel_fatigue`, in ``[0, 1]``.
        params: Travel model constants (weights ``w1``, ``w2``).

    Returns:
        Restedness in ``[0, 1]``.
    """
    penalty = params.w1 * jetlag_residual + params.w2 * travel_fatigue
    return _clamp(1.0 - penalty)


def compute_restedness(
    distance_km: float,
    origin_offset_h: float,
    venue_offset_h: float,
    days_since_arrival: float,
    params: TravelParams = TRAVEL_PARAMS,
) -> RestednessReport:
    """End-to-end restedness computation with all intermediates.

    Chains the L3 formulas: distance → flight time, offsets → timezone delta →
    direction → recovery days → jet-lag residual, flight time → travel fatigue,
    then composes the final ``rested_index``.

    Args:
        distance_km: Great-circle distance in kilometres (>= 0).
        origin_offset_h: UTC offset (hours) of the origin.
        venue_offset_h: UTC offset (hours) of the venue.
        days_since_arrival: Days elapsed since landing at the venue (>= 0).
        params: Travel model constants.

    Returns:
        A :class:`RestednessReport` with every intermediate, the final
        ``rested_index``, and the stamped ``model_version``.
    """
    flight_h = est_flight_h(distance_km, params)
    delta = tz_delta_h(origin_offset_h, venue_offset_h)
    dir_ = direction(delta)
    recovery = recovery_days_needed(delta, params)
    jr = jetlag_residual(days_since_arrival, recovery)
    tf = travel_fatigue(flight_h, days_since_arrival, params)
    ri = rested_index(jr, tf, params)

    return RestednessReport(
        distance_km=distance_km,
        est_flight_h=flight_h,
        tz_delta_h=delta,
        direction=dir_,
        recovery_days_needed=recovery,
        days_since_arrival=days_since_arrival,
        jetlag_residual=jr,
        travel_fatigue=tf,
        rested_index=ri,
        model_version=params.model_version,
    )
