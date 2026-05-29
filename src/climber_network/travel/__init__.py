"""climber_network.travel — Pure travel / circadian-load formulas (no graph, no DB).

The "L3" layer: great-circle distance, estimated flight time, timezone deltas,
the eastward-1-day / westward-0.5-day jet-lag recovery rule, raw travel
fatigue, and a composite ``rested_index`` in ``[0, 1]``. Every function is pure
and takes the frozen :class:`~climber_network.config.TravelParams` constants.
"""

from __future__ import annotations

from climber_network.travel.formulas import (
    Direction,
    RestednessReport,
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
